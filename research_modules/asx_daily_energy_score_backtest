"""
ASX daily-bar energy-score trend backtester.

This script tests a simple state-based trend strategy across many ASX daily CSV
files. The signal is derived from an "energy score" built from neighbouring EMA
curves. A fast EMA and a slow EMA of that score define the trading state:

- Fast score EMA > slow score EMA: enter or remain long.
- Fast score EMA < slow score EMA: exit the full position.

Design choices
--------------
- Signal calculation uses PRICE_COL_FOR_SCORE, default "average".
- Trade execution uses PRICE_COL_FOR_TRADE, default "close".
- Each entry uses a fixed maximum notional amount; quantity is floored so the
  gross position value does not exceed MIN_NOTIONAL.
- Fees are charged on both buy and sell notional.
- Work is parallelised per CSV file.
- Per-stock charts and a per-stock summary CSV are generated for inspection.
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

CSV_DIR = "/Users/richman/量化交易文件/1day股票数据(每日更新)"
CSV_GLOB = "*.csv"

# None means use all CPU cores. Set an integer when you want to leave cores free.
N_WORKERS: Optional[int] = 11

MIN_NOTIONAL = 10000.0
FEE_RATE = 8.8 / 10000.0

PRICE_COL_FOR_SCORE = "average"
PRICE_COL_FOR_TRADE = "close"

# Energy score is computed by summing weighted adjacent EMA differences for all
# spans in [ENERGY_EMA_MIN, ENERGY_EMA_MAX].
ENERGY_EMA_MIN = 180
ENERGY_EMA_MAX = 200

# The trading state is based on fast/slow EMAs of the energy score.
SCORE_EMA_FAST = 140
SCORE_EMA_SLOW = 150


# =============================================================================
# Utility functions
# =============================================================================

def calc_fee(notional: float) -> float:
    """Calculate transaction cost for a trade notional."""
    if not np.isfinite(notional) or notional <= 0:
        return 0.0
    return float(notional * FEE_RATE)


def fixed_notional_buy_qty(price: float, notional: float = MIN_NOTIONAL) -> int:
    """Return the largest integer share quantity whose value is <= notional."""
    if not np.isfinite(price) or price <= 0:
        return 0
    return max(int(np.floor(notional / price)), 0)


def compute_energy_score(
    df: pd.DataFrame,
    ema_min: int,
    ema_max: int,
    price_col: str = PRICE_COL_FOR_SCORE,
) -> pd.Series:
    """
    Compute the EMA-curve energy score.

    For each row, the score sums adjacent EMA percentage differences:

        score += ((EMA_p - EMA_{p+1}) / abs(EMA_{p+1})) * log(p + 1)

    Intuition: when shorter EMAs consistently sit above longer EMAs, the score
    becomes more positive; when shorter EMAs sit below longer EMAs, it becomes
    more negative. Using a range of EMA spans makes the signal smoother than a
    single moving-average crossover.
    """
    price = pd.Series(df[price_col].values, index=df.index, dtype=float)
    spans = list(range(ema_min, ema_max + 1))

    ema_data = {
        span: price.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()
        for span in spans
    }
    ema_df = pd.DataFrame(ema_data, index=df.index)
    valid = ema_df.notna().all(axis=1).to_numpy()

    score = np.full(len(df), np.nan, dtype=float)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return pd.Series(score, index=df.index)

    acc = np.zeros(len(valid_indices), dtype=np.float64)
    eps = 1e-12
    for span in range(ema_min, ema_max):
        fast_curve = ema_df.loc[valid_indices, span].to_numpy()
        slow_curve = ema_df.loc[valid_indices, span + 1].to_numpy()
        growth = (fast_curve - slow_curve) / (np.abs(slow_curve) + eps)
        acc += growth * np.log(span + 1.0)

    score[valid_indices] = acc
    return pd.Series(score, index=df.index)


def norm_day_key(ts: pd.Timestamp) -> str:
    """Convert a timestamp to a stable YYYY-MM-DD dictionary key."""
    return pd.Timestamp(ts).normalize().strftime("%Y-%m-%d")


def safe_symbol_from_path(path: str) -> str:
    """Use the CSV filename without extension as the symbol label."""
    return os.path.splitext(os.path.basename(path))[0]


def add_day_value(store: Dict[str, float], date_ts: pd.Timestamp, delta: float) -> None:
    """Accumulate a daily value in a dictionary."""
    key = norm_day_key(date_ts)
    store[key] = store.get(key, 0.0) + float(delta)


# =============================================================================
# Charting
# =============================================================================

def save_stock_chart(
    symbol: str,
    df: pd.DataFrame,
    buy_marks: List[tuple],
    sell_marks: List[tuple],
    stock_net_pnl: float,
    stock_fee: float,
    trades_closed: int,
    out_dir: str,
) -> Optional[str]:
    """Save a price chart with buy/sell markers for one stock."""
    if df.empty:
        return None

    dates = df["date"].dt.tz_convert("Australia/Sydney")
    plt.figure(figsize=(16, 7))
    plt.plot(dates, df[PRICE_COL_FOR_TRADE], label=PRICE_COL_FOR_TRADE.capitalize(), linewidth=1.2)

    if buy_marks:
        idxs = [mark[0] for mark in buy_marks]
        prices = [mark[1] for mark in buy_marks]
        plt.scatter(dates.iloc[idxs], prices, s=80, marker="^", c="green", edgecolors="black", linewidths=0.6, label="Buy: Fast > Slow", zorder=6)

    if sell_marks:
        idxs = [mark[0] for mark in sell_marks]
        prices = [mark[1] for mark in sell_marks]
        plt.scatter(dates.iloc[idxs], prices, s=80, marker="v", c="black", edgecolors="black", linewidths=0.6, label="Sell: Fast < Slow", zorder=6)

    plt.title(
        f"{symbol} Backtest Trades\n"
        f"Signal price: {PRICE_COL_FOR_SCORE} | Trade price: {PRICE_COL_FOR_TRADE}\n"
        f"Net PnL after fees: {stock_net_pnl:,.2f} | Fees: {stock_fee:,.2f} | Closed trades: {trades_closed}"
    )
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{symbol}.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


# =============================================================================
# Single-file backtest
# =============================================================================

def _load_and_clean_daily_csv(path: str) -> pd.DataFrame:
    """Load one daily CSV, validate required columns, and remove bad rows."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "open", "high", "low", "close", "average", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing_cols: {missing}")

    dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.loc[dt.notna()].copy()
    df["date"] = dt.loc[dt.notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "average", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close", "average", "volume"]).reset_index(drop=True)


def backtest_one_file(path: str, charts_dir: str) -> dict:
    """Backtest one stock CSV and return daily and per-stock performance data."""
    symbol = safe_symbol_from_path(path)
    try:
        df = _load_and_clean_daily_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {"skip": True, "path": path, "reason": f"load_failed: {type(exc).__name__}: {exc}"}

    min_len_need = max(ENERGY_EMA_MAX, SCORE_EMA_SLOW, SCORE_EMA_FAST) + 10
    if len(df) < min_len_need:
        return {"skip": True, "path": path, "reason": f"too_short_after_clean: len={len(df)} < {min_len_need}"}

    df["score"] = compute_energy_score(df, ENERGY_EMA_MIN, ENERGY_EMA_MAX, price_col=PRICE_COL_FOR_SCORE)
    df["score_ema_fast"] = df["score"].ewm(span=SCORE_EMA_FAST, adjust=False, min_periods=SCORE_EMA_FAST).mean()
    df["score_ema_slow"] = df["score"].ewm(span=SCORE_EMA_SLOW, adjust=False, min_periods=SCORE_EMA_SLOW).mean()

    pnl_by_day: Dict[str, float] = {}
    fee_by_day: Dict[str, float] = {}
    buy_marks: List[tuple] = []
    sell_marks: List[tuple] = []

    qty = 0
    entry_px = np.nan

    entries = exits = trades = fast_lt_slow_sell_count = 0
    stock_net_pnl = 0.0
    stock_fee = 0.0

    start_i = max(ENERGY_EMA_MAX, SCORE_EMA_SLOW, SCORE_EMA_FAST) + 2
    for i in range(start_i, len(df)):
        trade_date = df.at[i, "date"]
        close_i = float(df.at[i, PRICE_COL_FOR_TRADE])
        if not np.isfinite(close_i) or close_i <= 0:
            continue

        fast_val = float(df.at[i, "score_ema_fast"]) if np.isfinite(df.at[i, "score_ema_fast"]) else np.nan
        slow_val = float(df.at[i, "score_ema_slow"]) if np.isfinite(df.at[i, "score_ema_slow"]) else np.nan
        if not (np.isfinite(fast_val) and np.isfinite(slow_val)):
            continue

        # Enter long when the fast energy trend is above the slow energy trend.
        if qty <= 0 and fast_val > slow_val:
            buy_qty = fixed_notional_buy_qty(close_i, MIN_NOTIONAL)
            if buy_qty <= 0:
                continue

            buy_notional = buy_qty * close_i
            fee_buy = calc_fee(buy_notional)
            add_day_value(pnl_by_day, trade_date, -fee_buy)
            add_day_value(fee_by_day, trade_date, fee_buy)

            stock_net_pnl -= fee_buy
            stock_fee += fee_buy
            qty = buy_qty
            entry_px = close_i
            entries += 1
            buy_marks.append((i, close_i, "BUY"))

        # Exit the full position when the fast energy trend crosses below slow.
        elif qty > 0 and fast_val < slow_val:
            sell_notional = qty * close_i
            fee_sell = calc_fee(sell_notional)
            realized = qty * (close_i - entry_px) - fee_sell

            add_day_value(pnl_by_day, trade_date, realized)
            add_day_value(fee_by_day, trade_date, fee_sell)

            stock_net_pnl += realized
            stock_fee += fee_sell
            exits += 1
            trades += 1
            fast_lt_slow_sell_count += 1
            sell_marks.append((i, close_i, "FAST_LT_SLOW"))

            qty = 0
            entry_px = np.nan

    chart_path = None
    if entries > 0 or exits > 0 or trades > 0:
        chart_path = save_stock_chart(symbol, df, buy_marks, sell_marks, stock_net_pnl, stock_fee, trades, charts_dir)

    return {
        "path": path,
        "pnl_by_day": pnl_by_day,
        "fee_by_day": fee_by_day,
        "entries": entries,
        "exits": exits,
        "trades": trades,
        "fast_lt_slow_sell_count": fast_lt_slow_sell_count,
        "stock_net_pnl": stock_net_pnl,
        "stock_fee": stock_fee,
        "chart": chart_path,
    }


# =============================================================================
# Portfolio aggregation
# =============================================================================

def aggregate_results(results: List[dict], charts_dir: str, script_dir: str) -> Optional[pd.DataFrame]:
    """Aggregate all per-stock backtests into portfolio daily PnL and summaries."""
    pnl_by_day: Dict[str, float] = {}
    fee_by_day: Dict[str, float] = {}
    per_stock_summary = []

    total_entries = total_exits = total_trades = total_fast_lt_slow_sells = 0
    files_ok = 0

    for res in results:
        if not res:
            continue
        if res.get("skip"):
            print(f"[SKIP] {os.path.basename(res['path'])} -> {res['reason']}")
            continue

        files_ok += 1
        total_entries += res["entries"]
        total_exits += res["exits"]
        total_trades += res["trades"]
        total_fast_lt_slow_sells += res["fast_lt_slow_sell_count"]

        for day_key, value in res["pnl_by_day"].items():
            pnl_by_day[day_key] = pnl_by_day.get(day_key, 0.0) + float(value)
        for day_key, value in res["fee_by_day"].items():
            fee_by_day[day_key] = fee_by_day.get(day_key, 0.0) + float(value)

        per_stock_summary.append({
            "symbol": safe_symbol_from_path(res["path"]),
            "net_pnl": float(res.get("stock_net_pnl", 0.0)),
            "fee": float(res.get("stock_fee", 0.0)),
            "trades": int(res.get("trades", 0)),
            "fast_lt_slow_sells": int(res.get("fast_lt_slow_sell_count", 0)),
            "chart": res.get("chart"),
        })

    all_days = sorted(set(pnl_by_day) | set(fee_by_day))
    if not all_days:
        print("No trades / no PnL generated. Check signal conditions or input data.")
        return None

    out = pd.DataFrame({"date": pd.to_datetime(all_days)})
    out["pnl_delta"] = out["date"].dt.strftime("%Y-%m-%d").map(lambda d: pnl_by_day.get(d, 0.0))
    out["fee_delta"] = out["date"].dt.strftime("%Y-%m-%d").map(lambda d: fee_by_day.get(d, 0.0))
    out = out.sort_values("date").reset_index(drop=True)
    out["cum_pnl"] = out["pnl_delta"].cumsum()
    out["cum_fee"] = out["fee_delta"].cumsum()

    net_pnl = float(out["cum_pnl"].iloc[-1])
    total_fee = float(out["cum_fee"].iloc[-1])
    avg_pnl_per_trade = net_pnl / total_trades if total_trades > 0 else 0.0

    print("========== RESULT ==========")
    print(f"Files used:                {files_ok}")
    print(f"Charts saved to:           {charts_dir}")
    print(f"Signal price column:       {PRICE_COL_FOR_SCORE}")
    print(f"Trade price column:        {PRICE_COL_FOR_TRADE}")
    print(f"Entries:                   {total_entries}")
    print(f"Exits:                     {total_exits}")
    print(f"Closed trades:             {total_trades}")
    print(f"Fast<Slow sells:           {total_fast_lt_slow_sells}")
    print(f"Net PnL after fees:        {net_pnl:.2f}")
    print(f"Total fees:                {total_fee:.2f}")
    print(f"Average PnL / trade:       {avg_pnl_per_trade:.2f}")
    print("Strategy: buy/hold when fast score EMA > slow score EMA; sell all when fast score EMA < slow score EMA.")

    if per_stock_summary:
        save_per_stock_summary(per_stock_summary, script_dir)

    return out


def save_per_stock_summary(per_stock_summary: List[dict], script_dir: str) -> None:
    """Print and save per-stock performance sorted by average PnL per trade."""
    ps = pd.DataFrame(per_stock_summary)
    ps = ps[ps["trades"] > 0].copy()

    print("\n========== PER-STOCK SUMMARY ==========")
    if ps.empty:
        print("No stocks had closed trades.")
        return

    ps["avg_pnl_per_trade"] = ps.apply(
        lambda row: float(row["net_pnl"]) / int(row["trades"]) if int(row["trades"]) > 0 else np.nan,
        axis=1,
    )
    ps = ps.sort_values(["avg_pnl_per_trade", "net_pnl", "trades"], ascending=[False, False, False]).reset_index(drop=True)

    display_cols = ["symbol", "avg_pnl_per_trade", "net_pnl", "fee", "trades", "fast_lt_slow_sells"]
    show_df = ps[display_cols].copy()
    for col in ["avg_pnl_per_trade", "net_pnl", "fee"]:
        show_df[col] = show_df[col].map(lambda x: f"{x:,.2f}")

    pd.set_option("display.max_rows", 5000)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.width", 240)
    print(show_df.to_string(index=False))

    summary_csv = os.path.join(script_dir, "per_stock_summary.csv")
    ps[["symbol", "avg_pnl_per_trade", "net_pnl", "fee", "trades", "fast_lt_slow_sells", "chart"]].to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"\nPer-stock summary saved to: {summary_csv}")


def plot_portfolio_result(out: pd.DataFrame) -> None:
    """Show cumulative portfolio PnL and cumulative fees."""
    dates = out["date"].dt.tz_localize("UTC").dt.tz_convert("Australia/Sydney")

    plt.figure(figsize=(16, 7))
    plt.plot(dates, out["cum_pnl"], label="Cumulative Net PnL after fees")
    plt.plot(dates, out["cum_fee"], label="Cumulative Fees")
    plt.title(
        "Backtest Result: Cumulative Net PnL and Fees\n"
        f"Signal price={PRICE_COL_FOR_SCORE}, Trade price={PRICE_COL_FOR_TRADE}"
    )
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    paths = sorted(glob.glob(os.path.join(CSV_DIR, CSV_GLOB)))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR} with pattern {CSV_GLOB}")

    workers = N_WORKERS if N_WORKERS is not None and N_WORKERS > 0 else os.cpu_count()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    charts_dir = os.path.join(script_dir, "per_stock_charts")
    os.makedirs(charts_dir, exist_ok=True)

    print(f"Files scanned: {len(paths)}")
    print(f"Workers:       {workers}")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(backtest_one_file, path, charts_dir) for path in paths]
        for future in as_completed(futures):
            results.append(future.result())

    out = aggregate_results(results, charts_dir, script_dir)
    if out is not None:
        plot_portfolio_result(out)


if __name__ == "__main__":
    main()
