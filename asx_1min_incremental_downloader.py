"""
ASX 1-minute historical bar downloader with incremental backfill/update.

Purpose
-------
This script downloads 1-minute RTH historical bars for ASX-listed symbols from
Interactive Brokers, keeps one rolling CSV file per symbol, and updates it in a
safe incremental way:

1. If a symbol has no local CSV yet, the script starts from the target trading
   date and backfills historical data in fixed-size windows.
2. If a symbol already has a CSV, the script first tries to extend the history
   further backward, then appends missing recent trading days forward.
3. After every fetch/merge, the script removes after-close minute bars that IB
   may include in the RTH response.

Market-close cleanup rules
--------------------------
- Normal ASX trading days: keep all bars before 16:00 and keep only 16:10 among
  bars at or after 16:00.
- Early-close days: for the effective ASX early-close dates around Christmas Eve
  and New Year's Eve, keep all bars before 14:00 and keep only 14:10 among bars
  at or after 14:00.

Notes
-----
- This script assumes TWS or IB Gateway is running locally.
- File paths are configured for the author's local machine. Change the path
  constants before running on another computer.
- IB historical data availability and pacing limits can vary by account, symbol,
  and market-data subscription.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date as Date
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Contract, Stock, util
from zoneinfo import ZoneInfo


# =============================================================================
# Configuration
# =============================================================================

TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 17

# Excel file containing ASX ticker symbols in column A.
# The first row is treated as a header and skipped.
EXCEL_FILE = "/Users/richman/量化交易文件/ASX200000.xlsx"

SAVE_DIR = Path("/Users/richman/量化交易文件/股票数据(每日更新)")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

EXCHANGE = "ASX"
PRIMARY_EXCHANGE = "ASX"
CURRENCY = "AUD"

BAR_SIZE = "1 min"
WHAT_TO_SHOW = "TRADES"
USE_RTH = True

# Optional pause after each IB request. Keep at 0 for fastest runs, increase if
# your account hits pacing limits frequently.
INTER_REQUEST_SLEEP = 0.0

SYDNEY_TZ = ZoneInfo("Australia/Sydney")
UTC_TZ = ZoneInfo("UTC")

# Set to None to backfill until IB returns an empty window. Otherwise the
# backfill stops once this inclusive lower date is reached.
BACKFILL_STOP_DATE_STR = "2015-01-01"
BACKFILL_STOP_DT: Optional[datetime] = (
    datetime.strptime(BACKFILL_STOP_DATE_STR, "%Y-%m-%d").replace(tzinfo=SYDNEY_TZ)
    if BACKFILL_STOP_DATE_STR
    else None
)

# Number of calendar days per backward historical-data request.
BACKFILL_WINDOW_DAYS = 30

# CSV filename format:
# CBA_ASX_1min_20200101_1000_20251001_1610.csv
FNAME_RE = re.compile(
    r"^(?P<sym>[A-Z0-9]+)_ASX_1min_(?P<start>\d{8}_\d{4})_(?P<end>\d{8}_\d{4})\.csv$"
)


# =============================================================================
# IB request helpers
# =============================================================================

def ib_call_with_retry(
    call: Callable[..., Any],
    *args: Any,
    retry: int = 10,
    backoff_pacing: int = 120,
    backoff_other: int = 10,
    **kwargs: Any,
) -> Any:
    """
    Execute an Interactive Brokers API call with retry handling.

    The downloader treats "HMDS query returned no data" as a normal empty result,
    because older windows or illiquid symbols can legitimately have no minute
    bars. Pacing violations receive a longer backoff; other errors receive a
    shorter backoff and are retried before finally being raised.
    """
    for attempt in range(retry):
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - IB exceptions are heterogeneous.
            msg = str(exc).lower()
            if "hmds query returned no data" in msg:
                return None

            wait_seconds = backoff_pacing if ("pacing" in msg or "violation" in msg) else backoff_other
            if attempt == retry - 1:
                raise

            print(f"⚠️ IB request failed ({attempt + 1}/{retry}): {exc}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def sleep_after_request() -> None:
    """Apply optional request-level throttling."""
    if INTER_REQUEST_SLEEP > 0:
        time.sleep(INTER_REQUEST_SLEEP)


# =============================================================================
# Symbol and contract helpers
# =============================================================================

def load_symbols_from_excel(xlsx_path: str) -> List[str]:
    """Load ticker symbols from column A of the configured Excel file."""
    df = pd.read_excel(xlsx_path, header=None, usecols=[0], skiprows=1)
    return df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()


def make_asx_contract(symbol: str) -> Contract:
    """Create an ASX stock contract for IB historical-data requests."""
    return Stock(symbol.strip().upper(), EXCHANGE, CURRENCY, primaryExchange=PRIMARY_EXCHANGE)


def qualify_or_skip(ib: IB, contract: Contract, symbol: str) -> Optional[Contract]:
    """
    Ask IB to qualify a contract. Return None when the symbol cannot be used.

    Qualifying early avoids wasting historical-data requests on invalid or
    ambiguous ticker symbols.
    """
    try:
        qualified = ib.qualifyContracts(contract)
    except Exception as exc:  # noqa: BLE001
        print(f"[{symbol}] Contract qualification failed: {exc}. Skipped.")
        return None

    if not qualified:
        print(f"[{symbol}] Contract is invalid or not recognised. Skipped.")
        return None

    return qualified[0]


# =============================================================================
# Trading-date detection
# =============================================================================

def _series_to_sydney_midnight(date_series: pd.Series) -> pd.Series:
    """Convert an IB date column to Sydney-local midnight timestamps."""
    s = pd.to_datetime(date_series, errors="coerce")
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(SYDNEY_TZ).dt.normalize()


def get_latest_trading_date(ib: IB, contract: Contract) -> datetime:
    """Return the latest available daily trading date as Sydney-local midnight."""
    bars = ib_call_with_retry(
        ib.reqHistoricalData,
        contract,
        endDateTime="",
        durationStr="3 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )
    sleep_after_request()

    if not bars:
        bars = ib_call_with_retry(
            ib.reqHistoricalData,
            contract,
            endDateTime=datetime.now(UTC_TZ) - timedelta(days=1),
            durationStr="3 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
            keepUpToDate=False,
        )
        sleep_after_request()

    if not bars:
        raise RuntimeError("Unable to identify the latest trading date from daily bars.")

    df = util.df(bars)
    return _series_to_sydney_midnight(df["date"]).max().to_pydatetime()


def get_prev_trading_date(ib: IB, contract: Contract, ref_date_syd: datetime) -> Optional[datetime]:
    """Return the nearest trading date before ``ref_date_syd``."""
    end_dt_syd = ref_date_syd.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=1)
    bars = ib_call_with_retry(
        ib.reqHistoricalData,
        contract,
        endDateTime=end_dt_syd.astimezone(UTC_TZ),
        durationStr="10 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        keepUpToDate=False,
    )
    sleep_after_request()
    if not bars:
        return None

    df = util.df(bars)
    s = _series_to_sydney_midnight(df["date"])
    candidates = s[s < ref_date_syd.replace(hour=0, minute=0, second=0, microsecond=0)]
    return None if candidates.empty else candidates.max().to_pydatetime()


# =============================================================================
# Historical bar fetching
# =============================================================================

def fetch_intraday_one_day(ib: IB, contract: Contract, date_syd: datetime) -> pd.DataFrame:
    """Fetch one Sydney calendar day of 1-minute bars and crop precisely to that date."""
    end_dt_syd = datetime(date_syd.year, date_syd.month, date_syd.day, 23, 59, 59, tzinfo=SYDNEY_TZ)
    bars = ib_call_with_retry(
        ib.reqHistoricalData,
        contract,
        endDateTime=end_dt_syd.astimezone(UTC_TZ),
        durationStr="1 D",
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        formatDate=1,
        keepUpToDate=False,
    )
    sleep_after_request()
    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(SYDNEY_TZ)
    df = df[df["date"].dt.date == date_syd.date()].copy()
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def fetch_intraday_multi_days(
    ib: IB,
    contract: Contract,
    end_date_syd: datetime,
    days: int = BACKFILL_WINDOW_DAYS,
) -> pd.DataFrame:
    """Fetch a multi-day backward window ending on ``end_date_syd`` inclusive."""
    start_date_syd = (end_date_syd - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt_syd = end_date_syd.replace(hour=23, minute=59, second=59, microsecond=0)

    bars = ib_call_with_retry(
        ib.reqHistoricalData,
        contract,
        endDateTime=end_dt_syd.astimezone(UTC_TZ),
        durationStr=f"{days} D",
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        formatDate=1,
        keepUpToDate=False,
    )
    sleep_after_request()
    if not bars:
        return pd.DataFrame()

    df = util.df(bars)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(SYDNEY_TZ)
    df = df[(df["date"] >= start_date_syd) & (df["date"] <= end_dt_syd)].copy()
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


# =============================================================================
# File naming and parsing
# =============================================================================

def format_fname(symbol: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    """Build the canonical CSV filename for a symbol and date range."""
    return f"{symbol}_ASX_1min_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%Y%m%d_%H%M')}.csv"


def find_existing_file(symbol: str) -> Optional[Path]:
    """Find the latest local CSV for ``symbol`` according to filename date range."""
    files = list(SAVE_DIR.glob(f"{symbol}_ASX_1min_*.csv"))
    if not files:
        return None

    def sort_key(path: Path) -> Tuple[str, str]:
        match = FNAME_RE.match(path.name)
        return (match.group("start"), match.group("end")) if match else ("", "")

    return sorted(files, key=sort_key)[-1]


def parse_fname_dates(path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Parse start/end timestamps from a canonical CSV filename."""
    match = FNAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filename format: {path.name}")

    start = pd.to_datetime(match.group("start"), format="%Y%m%d_%H%M").tz_localize(SYDNEY_TZ)
    end = pd.to_datetime(match.group("end"), format="%Y%m%d_%H%M").tz_localize(SYDNEY_TZ)
    return start, end


# =============================================================================
# Market-close cleanup
# =============================================================================

def _is_weekend(d: Date) -> bool:
    return d.weekday() >= 5


def _prev_business_day(d: Date) -> Date:
    """Move weekend dates back to the previous weekday."""
    while _is_weekend(d):
        d = Date.fromordinal(d.toordinal() - 1)
    return d


def _early_close_days_in_year(year: int) -> set[Date]:
    """Return effective early-close dates for Christmas Eve and New Year's Eve."""
    return {_prev_business_day(Date(year, 12, 24)), _prev_business_day(Date(year, 12, 31))}


def _collect_early_close_dates_for_df(df: pd.DataFrame) -> set[Date]:
    years = sorted(set(df["date"].dt.year.tolist()))
    early_dates: set[Date] = set()
    for year in years:
        early_dates |= _early_close_days_in_year(year)
    return early_dates


def clean_intraday_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted post-close minute bars according to normal/early close rules."""
    if df.empty:
        return df

    work = df.copy()
    work["_day"] = work["date"].dt.date
    work["_hour"] = work["date"].dt.hour
    work["_minute"] = work["date"].dt.minute

    early_close_dates = _collect_early_close_dates_for_df(work)
    is_early = work["_day"].isin(early_close_dates)

    normal_keep = (work["_hour"] < 16) | ((work["_hour"] == 16) & (work["_minute"] == 10))
    early_keep = (work["_hour"] < 14) | ((work["_hour"] == 14) & (work["_minute"] == 10))

    out = work.loc[(~is_early & normal_keep) | (is_early & early_keep)].drop(columns=["_day", "_hour", "_minute"])
    return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def already_latest_by_filename_relaxed(path: Path, target_end_date: datetime) -> bool:
    """Check whether the local CSV appears to cover the target trading date."""
    try:
        _, end_dt = parse_fname_dates(path)
    except Exception:  # noqa: BLE001
        return False

    if end_dt.date() != target_end_date.date():
        return False

    return (end_dt.hour == 16 and end_dt.minute >= 10) or (end_dt.hour == 14 and end_dt.minute >= 10)


def next_day_start(d: datetime) -> datetime:
    return (d + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


# =============================================================================
# Incremental update routines
# =============================================================================

def _read_existing_csv(path: Path) -> pd.DataFrame:
    """Read an existing symbol CSV and normalise its timestamp column."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(SYDNEY_TZ)
    return df.dropna(subset=["date"])


def _replace_existing_csv(existing: Path, new_path: Path) -> None:
    """Delete the old CSV unless the new output overwrote the same path."""
    try:
        if existing.resolve() != new_path.resolve():
            existing.unlink()
        else:
            print(f"ℹ️ Reused filename {existing.name}; old file deletion skipped.")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ Failed to delete old file {existing.name}: {exc}")


def backfill_backward_by_windows(
    ib: IB,
    symbol: str,
    contract: Contract,
    start_from_date: datetime,
    days_per_chunk: int = BACKFILL_WINDOW_DAYS,
    stop_at: Optional[datetime] = None,
) -> None:
    """Extend a symbol's local CSV backward in fixed-size historical windows."""
    existing = find_existing_file(symbol)
    cursor_end = start_from_date.replace(hour=0, minute=0, second=0, microsecond=0)

    while True:
        if stop_at is not None and cursor_end < stop_at:
            print(f"[{symbol}] Reached configured earliest date {stop_at.date()}; backfill stopped.")
            break

        days_this_round = days_per_chunk
        if stop_at is not None:
            planned_start = cursor_end - timedelta(days=days_per_chunk - 1)
            if planned_start < stop_at:
                days_this_round = (cursor_end - stop_at).days + 1
                if days_this_round <= 0:
                    break

        window_df = fetch_intraday_multi_days(ib, contract, cursor_end, days=days_this_round)
        if window_df.empty:
            print(f"[{symbol}] Empty window ending {cursor_end.date()} ({days_this_round} days); backfill stopped.")
            break

        window_df = clean_intraday_df(window_df)
        if window_df.empty:
            print(f"[{symbol}] Window ending {cursor_end.date()} became empty after cleanup; backfill stopped.")
            break

        if existing is None:
            out_path = SAVE_DIR / format_fname(symbol, window_df["date"].min(), window_df["date"].max())
            window_df.to_csv(out_path, index=False)
            existing = out_path
            print(f"[{symbol}] Initialised history: {out_path.name} ({len(window_df)} rows)")
        else:
            old_df = _read_existing_csv(existing)
            merged = pd.concat([window_df, old_df], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
            merged = clean_intraday_df(merged).reset_index(drop=True)

            new_path = SAVE_DIR / format_fname(symbol, merged["date"].min(), merged["date"].max())
            merged.to_csv(new_path, index=False)
            _replace_existing_csv(existing, new_path)
            existing = new_path
            print(f"[{symbol}] Backfilled to {cursor_end.date()}: {new_path.name} ({len(merged)} rows)")

        cursor_end = (cursor_end - timedelta(days=days_this_round)).replace(hour=0, minute=0, second=0, microsecond=0)


def append_forward_one_day(ib: IB, symbol: str, contract: Contract, target_end_date: datetime) -> None:
    """Append the next missing trading day until the CSV reaches ``target_end_date``."""
    existing = find_existing_file(symbol)
    if existing is None or already_latest_by_filename_relaxed(existing, target_end_date):
        return

    _, end_dt = parse_fname_dates(existing)
    cursor_day = next_day_start(end_dt)

    while cursor_day <= target_end_date:
        day_df = fetch_intraday_one_day(ib, contract, cursor_day)
        if day_df.empty:
            cursor_day = next_day_start(cursor_day)
            continue

        day_df = clean_intraday_df(day_df)
        old_df = _read_existing_csv(existing)
        merged = pd.concat([old_df, day_df], ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        merged = clean_intraday_df(merged).reset_index(drop=True)

        new_path = SAVE_DIR / format_fname(symbol, merged["date"].min(), merged["date"].max())
        merged.to_csv(new_path, index=False)
        _replace_existing_csv(existing, new_path)
        print(f"[{symbol}] Appended {cursor_day.date()}: {new_path.name} ({len(merged)} rows)")
        return

    print(f"[{symbol}] No new minute bars found up to {target_end_date.date()}.")


# =============================================================================
# Main orchestration
# =============================================================================

def determine_target_end_date(ib: IB, probe_contract: Contract) -> datetime:
    """Choose the latest safe trading date to download based on Sydney time."""
    latest_trade_date = get_latest_trading_date(ib, probe_contract)
    now_syd = datetime.now(SYDNEY_TZ)
    today_syd = now_syd.replace(hour=0, minute=0, second=0, microsecond=0)

    print(
        f"🔎 Latest trading date: {latest_trade_date.date()} | "
        f"Today Sydney: {today_syd.date()} | Now: {now_syd.strftime('%H:%M:%S')}"
    )

    if today_syd == latest_trade_date and now_syd < now_syd.replace(hour=17, minute=0, second=0, microsecond=0):
        prev_trading_date = get_prev_trading_date(ib, probe_contract, ref_date_syd=today_syd)
        if prev_trading_date is not None:
            print(f"⏪ Before 17:00 Sydney; target date changed to previous trading day: {prev_trading_date.date()}")
            return prev_trading_date

    print(f"✅ Target update date: {latest_trade_date.date()}")
    return latest_trade_date


def run_update_once() -> None:
    """Run one full update pass across all configured symbols."""
    symbols = load_symbols_from_excel(EXCEL_FILE)
    print(f"✅ Loaded {len(symbols)} symbols from {EXCEL_FILE}.")

    ib = IB()
    print(f"Connecting to IB at {TWS_HOST}:{TWS_PORT}, clientId={CLIENT_ID}...")
    ib.connect(TWS_HOST, TWS_PORT, clientId=CLIENT_ID)

    try:
        probe = qualify_or_skip(ib, make_asx_contract(symbols[0]), symbols[0])
        if probe is None:
            print("❌ Cannot detect latest trading date because the first symbol is invalid.")
            sys.exit(1)

        target_end_date = determine_target_end_date(ib, probe)

        for i, symbol in enumerate(symbols, start=1):
            print(f"\n=== {symbol} | ASX 1-min TRADES | RTH={USE_RTH} | {i}/{len(symbols)} ===")
            contract = qualify_or_skip(ib, make_asx_contract(symbol), symbol)
            if contract is None:
                continue

            try:
                existing = find_existing_file(symbol)
                if existing is None:
                    backfill_backward_by_windows(
                        ib,
                        symbol,
                        contract,
                        start_from_date=target_end_date,
                        days_per_chunk=BACKFILL_WINDOW_DAYS,
                        stop_at=BACKFILL_STOP_DT,
                    )
                else:
                    start_dt, _ = parse_fname_dates(existing)
                    backfill_backward_by_windows(
                        ib,
                        symbol,
                        contract,
                        start_from_date=start_dt.to_pydatetime(),
                        days_per_chunk=BACKFILL_WINDOW_DAYS,
                        stop_at=BACKFILL_STOP_DT,
                    )
                    append_forward_one_day(ib, symbol, contract, target_end_date)
            except Exception as exc:  # noqa: BLE001
                print(f"[{symbol}] Update failed: {exc}")
    finally:
        ib.disconnect()

    print("\n✅ Update pass completed.")


def wait_until_next_1700_sydney() -> None:
    """Sleep until the next scheduled daily run time in Sydney."""
    now = datetime.now(SYDNEY_TZ)
    target = now.replace(hour=17, minute=5, second=0, microsecond=0)
    if now >= target:
        target = (now + timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)

    print(f"⏱ Next run: {target.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    time.sleep((target - now).total_seconds())


def main() -> None:
    run_update_once()
    while True:
        wait_until_next_1700_sydney()
        run_update_once()


if __name__ == "__main__":
    main()
