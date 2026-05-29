"""
Multi-stock Transformer-XL training pipeline for ASX 1-minute bars.

The script trains a shared sequence model across many ASX stocks on a unified
minute-level timeline. It combines three learning objectives:

1. Past-window reconstruction: the model reconstructs the input window, which
   encourages it to learn compact market-state representations.
2. Future-window prediction: the model predicts the next FUTURE_HORIZON bars,
   which encourages representations that contain forward-looking dynamics.
3. Buy/Sell classification: labels are generated from a future-window trend rule
   based on EMA movement. Only Buy and Sell labels participate in the
   classification loss; Hold labels are created for analysis but excluded from
   the classification mask in this version.

Important assumptions
---------------------
- Input CSV files are created by the ASX minute downloader and contain:
  date, open, high, low, close, volume, average, barcount.
- The ticker Excel file defines the stock order. That order is saved to JSON and
  must be reused during inference.
- Existing model weights can be reused as a backbone. The classification head is
  reinitialised so that label-rule changes do not inherit an incompatible head.
"""

from __future__ import annotations

import glob
import json
import math
import multiprocessing as mp
import os
import random
import re
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler, Subset


# =============================================================================
# Paths
# =============================================================================

DATA_DIR = "C:/Users/richman/Desktop/量化交易文件/股票数据(每日更新)"
TICKERS_XLSX = "C:/Users/richman/Desktop/量化交易文件/ASX200000.xlsx"
SAVE_DIR = "C:/Users/richman/Desktop/量化交易文件"

SCALER_PATH = os.path.join(SAVE_DIR, "scaler_multi.joblib")
TRAIN_TICKERS_JSON = os.path.join(SAVE_DIR, "train_tickers.json")

START_DATE = "2020-01-01"
END_DATE = "2025-10-01"


# =============================================================================
# Model and training hyperparameters
# =============================================================================

SEQ_LEN = 64
BATCH_SIZE = 4
LR = 1e-4
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 10
MEM_LEN = 128
TIME_DIM = 64
STOCK_EMB = 64
DROPOUT = 0.1
SEED = 42
MAX_STOCKS: Optional[int] = None

NUM_COLS = ["open", "high", "low", "close", "volume", "average", "barcount"]

EARLY_STOP_PATIENCE = 10
MIN_IMPROVE = 0.0

# Classification classes: 0=Hold, 1=Buy, 2=Sell.
NUM_CLASSES = 3

# Label generation settings. The label at time t is determined by how many of
# the next SUPER_K_HORIZON EMA values move beyond a threshold relative to the
# current close.
SUPER_K_HORIZON = 100
BUY_THRESHOLD = 0.003
SELL_THRESHOLD = 0.003

# Number of future bars the auxiliary prediction head tries to forecast.
FUTURE_HORIZON = 30

# Loss weights.
CLS_ALPHA = 1.0
RECON_PAST_BETA = 1.0
RECON_FUT_BETA = 1.0


# =============================================================================
# Label generation
# =============================================================================

def compute_superk_labels(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    horizon: int = SUPER_K_HORIZON,
    ema_span: int = 5,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Generate trend labels from the future EMA path.

    For each time t, the function looks at the next ``horizon`` bars. If enough
    future EMA values are above the current close by BUY_THRESHOLD, t is labelled
    as Buy. If enough future EMA values are below the current close by
    SELL_THRESHOLD, t is labelled as Sell. Otherwise it is Hold.

    The open/high/low arguments are intentionally kept in the signature so the
    label rule can be extended later without changing the multiprocessing call.
    """
    del open_arr, high_arr, low_arr

    close_arr = np.asarray(close_arr, dtype=float)
    total_len = len(close_arr)
    horizon = int(horizon)

    ema = pd.Series(close_arr).ewm(span=ema_span, adjust=False).mean().to_numpy()
    labels = np.full(total_len, -1, dtype=np.int64)

    # A lower ratio creates more labels. Increase it when you want only stronger
    # future trends to be labelled.
    threshold_ratio = 1 / 4
    required_hits = int(np.ceil(horizon * threshold_ratio))

    for t in range(total_len):
        if t + horizon >= total_len:
            continue

        current_close = close_arr[t]
        if not np.isfinite(current_close):
            continue

        future_ema = ema[t + 1 : t + 1 + horizon]
        up_hits = np.sum(future_ema >= current_close * (1 + BUY_THRESHOLD))
        down_hits = np.sum(future_ema <= current_close * (1 - SELL_THRESHOLD))

        if up_hits >= required_hits:
            labels[t] = 1
        elif down_hits >= required_hits:
            labels[t] = 2
        else:
            labels[t] = 0

    buy_points = np.where(labels == 1)[0].tolist()
    sell_points = np.where(labels == 2)[0].tolist()
    return labels, buy_points, sell_points


def compute_labels_for_one_stock(args: Tuple[str, pd.DataFrame, Sequence[Tuple[date, date]]]):
    """Multiprocessing worker that labels one stock over the requested date ranges."""
    code, df, train_ranges = args

    date_values = df["date"].dt.date.to_numpy()
    range_mask = np.zeros(len(df), dtype=bool)
    for start_date, end_date in train_ranges:
        range_mask |= (date_values >= start_date) & (date_values <= end_date)

    df_sub = df.loc[range_mask].sort_values("date").reset_index(drop=True)
    if df_sub.empty:
        return code, None

    required = ["open", "high", "low", "close"]
    if any(col not in df_sub.columns for col in required):
        return code, None

    labels, _, _ = compute_superk_labels(
        df_sub["open"].to_numpy(float),
        df_sub["high"].to_numpy(float),
        df_sub["low"].to_numpy(float),
        df_sub["close"].to_numpy(float),
        horizon=SUPER_K_HORIZON,
        ema_span=1,
    )

    valid_mask = labels >= 0
    if valid_mask.any():
        counts = np.bincount(labels[valid_mask], minlength=NUM_CLASSES)
        print(f"[{code}] labels: Hold={counts[0]} Buy={counts[1]} Sell={counts[2]} Total={counts.sum()}")
    else:
        print(f"[{code}] No valid labels generated.")

    df_sub["dp_label"] = labels
    return code, df_sub[["date", "dp_label"]]


# =============================================================================
# Model
# =============================================================================

class Time2Vec(nn.Module):
    """Time2Vec embedding: one linear component plus periodic components."""

    def __init__(self, k: int):
        super().__init__()
        if k < 2:
            raise ValueError("Time2Vec dimension k must be at least 2.")
        self.lin = nn.Linear(1, 1)
        self.per = nn.Linear(1, k - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.lin(t), torch.sin(self.per(t))], dim=-1)


class MultiStockTXLAligned(nn.Module):
    """Transformer encoder shared across all stocks on a unified timeline."""

    def __init__(
        self,
        in_feats: int,
        num_stocks: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 8,
        mem_len: int = 0,
        time_dim: int = 32,
        stock_emb: int = 32,
        dropout: float = 0.1,
        pred_len: int = 0,
    ):
        super().__init__()
        self.mem_len = mem_len
        self.num_stocks = num_stocks
        self.in_feats = in_feats
        self.pred_len = pred_len

        self.time2vec = Time2Vec(time_dim)
        self.stock_emb = nn.Embedding(num_stocks, stock_emb)

        # Input features are market features + time embedding + stock embedding
        # + one missing-data indicator channel.
        self.input_fc = nn.Linear(in_feats + time_dim + stock_emb + 1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.proj = nn.Linear(d_model, d_model)
        self.recon_head = nn.Linear(d_model, in_feats)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.future_head = (
            nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, in_feats * pred_len))
            if pred_len > 0
            else None
        )

    @torch.no_grad()
    def _update_mem(self, mem: Optional[torch.Tensor], z: torch.Tensor) -> Optional[torch.Tensor]:
        del mem
        if self.mem_len <= 0:
            return None
        return z[-self.mem_len :].detach()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        mem: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        x: [S, B, N, F]
            Historical input window.
        t: [S, B, 1]
            Time feature for each minute.
        mask: [S, B, N]
            1 for available stock bars, 0 for missing bars.
        mem: optional [M, B*N, D]
            Cached Transformer-XL memory from previous stream segment.
        """
        seq_len, batch_size, num_stocks, num_feats = x.shape
        if num_stocks != self.num_stocks:
            raise ValueError("Input stock dimension does not match model.num_stocks.")

        x_flat = x.view(seq_len, batch_size * num_stocks, num_feats)
        m_flat = mask.view(seq_len, batch_size * num_stocks)

        t2v = self.time2vec(t).unsqueeze(2).expand(-1, -1, num_stocks, -1).reshape(seq_len, batch_size * num_stocks, -1)
        sid = torch.arange(num_stocks, device=x.device).view(1, 1, num_stocks).expand(seq_len, batch_size, num_stocks)
        stock_vec = self.stock_emb(sid).view(seq_len, batch_size * num_stocks, -1)

        h = self.input_fc(torch.cat([x_flat, t2v, stock_vec, m_flat.unsqueeze(-1)], dim=-1))
        if mem is not None and mem.numel() > 0 and mem.shape[1] == batch_size * num_stocks:
            h = torch.cat([mem, h], dim=0)

        z_all = self.proj(self.encoder(h))
        z_hist = z_all[-seq_len:]

        rec = self.recon_head(z_hist).view(seq_len, batch_size, num_stocks, num_feats)
        logits = self.cls_head(z_hist).view(seq_len, batch_size, num_stocks, -1)

        fut_pred = None
        if self.future_head is not None and self.pred_len > 0:
            fut_flat = self.future_head(z_hist[-1])
            fut_pred = fut_flat.view(self.pred_len, batch_size, num_stocks, num_feats)

        return z_hist, rec, logits, fut_pred, self._update_mem(mem, z_all)


# =============================================================================
# Dataset and sampler
# =============================================================================

class TXLStreamSampler(Sampler[List[int]]):
    """Yield sequential batches that preserve Transformer-XL stream continuity."""

    def __init__(self, n_steps: int, batch_size: int, seq_len: int, drop_last: bool = True):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.drop_last = drop_last
        self.max_start = n_steps - 1 - (batch_size - 1) * seq_len

    def __iter__(self):
        if self.max_start < 0:
            if not self.drop_last:
                batch = [j * self.seq_len for j in range(self.batch_size) if j * self.seq_len < self.n_steps]
                if batch:
                    yield batch
            return

        start = 0
        while start <= self.max_start:
            yield [start + j * self.seq_len for j in range(self.batch_size)]
            start += self.seq_len

    def __len__(self) -> int:
        if self.max_start < 0:
            return 0 if self.drop_last else 1
        return self.max_start // self.seq_len + 1


class AlignedDataset(Dataset):
    """Windowed dataset backed by aligned [time, stock, feature] arrays."""

    def __init__(
        self,
        full_X: np.ndarray,
        full_mask: np.ndarray,
        full_T: np.ndarray,
        seq_len: int,
        full_label_raw: np.ndarray,
        fut_len: int,
    ):
        self.full_X = full_X
        self.full_mask = full_mask
        self.full_T = full_T
        self.seq_len = seq_len
        self.fut_len = fut_len
        self.full_label = full_label_raw.astype(np.int64)

        # Current design trains classification only on Buy/Sell events. Hold can
        # be included later by changing this mask to labels >= 0.
        self.full_label_mask = ((self.full_label == 1) | (self.full_label == 2)).astype(np.float32)
        self.n_steps = max(0, full_X.shape[0] - seq_len - fut_len + 1)

    def __len__(self) -> int:
        return self.n_steps

    def __getitem__(self, idx: int):
        a = idx
        b = idx + self.seq_len
        c = b + self.fut_len
        return (
            torch.tensor(self.full_X[a:b], dtype=torch.float32),
            torch.tensor(self.full_T[a:b], dtype=torch.float32),
            torch.tensor(self.full_mask[a:b], dtype=torch.float32),
            torch.tensor(self.full_label[b - 1], dtype=torch.long),
            torch.tensor(self.full_label_mask[b - 1], dtype=torch.float32),
            torch.tensor(self.full_X[b:c], dtype=torch.float32),
            torch.tensor(self.full_mask[b:c], dtype=torch.float32),
        )


def collate_fn(batch):
    """Convert dataset samples to Transformer layout [S, B, ...]."""
    X_hist = torch.stack([item[0] for item in batch], dim=1)
    T_hist = torch.stack([item[1] for item in batch], dim=1)
    M_hist = torch.stack([item[2] for item in batch], dim=1)
    label_last = torch.stack([item[3] for item in batch], dim=0)
    mask_last = torch.stack([item[4] for item in batch], dim=0)
    X_fut = torch.stack([item[5] for item in batch], dim=1)
    M_fut = torch.stack([item[6] for item in batch], dim=1)
    return X_hist, T_hist, M_hist, label_last, mask_last, X_fut, M_fut


# =============================================================================
# General utilities
# =============================================================================

def read_tickers_xlsx(path: str, sheet_name=0, col_index=0) -> List[str]:
    """Read and validate ASX ticker symbols from Excel."""
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    tickers = df.iloc[:, col_index].dropna().astype(str).str.strip().str.upper()
    return [ticker for ticker in tickers if re.fullmatch(r"[A-Z0-9]{2,6}", ticker)]


def find_stock_files(data_dir: str, tickers: Iterable[str]) -> List[str]:
    ticker_set = set(tickers)
    files = []
    for path in glob.glob(os.path.join(data_dir, "*_ASX_1min_*.csv")):
        code = os.path.basename(path).split("_")[0].upper()
        if code in ticker_set:
            files.append(path)
    return sorted(files)


def load_csv_df(path: str) -> pd.DataFrame:
    """Load one stock CSV and enforce required columns and datetime sorting."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in NUM_COLS + ["date"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def intersect_range(r: Tuple[date, date], data_min: date, data_max: date) -> Optional[Tuple[date, date]]:
    start, end = max(r[0], data_min), min(r[1], data_max)
    return None if start > end else (start, end)


def find_existing_model(save_dir: str):
    """Return the latest model path and its filename date coverage."""
    files = sorted(glob.glob(os.path.join(save_dir, "txl_model_multi_*.pth")))
    if not files:
        return None, None, None
    path = files[-1]
    match = re.search(r"txl_model_multi_(\d{8})_(\d{8})\.pth", os.path.basename(path))
    if not match:
        return None, None, None
    start = datetime.strptime(match.group(1), "%Y%m%d").date()
    end = datetime.strptime(match.group(2), "%Y%m%d").date()
    return path, start, end


def choose_device() -> torch.device:
    """Select the best available PyTorch device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# Training and validation loops
# =============================================================================

def _loss_step(
    model: nn.Module,
    batch,
    device: torch.device,
    mem: Optional[torch.Tensor],
    use_cls_loss: bool,
    cls_only: bool,
    class_weights: Optional[torch.Tensor],
):
    X, T, M, label_last, mask_last, X_fut, M_fut = [item.to(device) for item in batch]
    _, rec, logits, fut_pred, mem = model(X, T, M, mem=mem)

    diff_past = (rec - X) * M.unsqueeze(-1)
    recon_past = F.smooth_l1_loss(diff_past, torch.zeros_like(diff_past), reduction="sum", beta=1.0) / (M.sum() * X.shape[-1] + 1e-8)

    if fut_pred is not None:
        diff_fut = (fut_pred - X_fut) * M_fut.unsqueeze(-1)
        recon_fut = F.smooth_l1_loss(diff_fut, torch.zeros_like(diff_fut), reduction="sum", beta=1.0) / (M_fut.sum() * X_fut.shape[-1] + 1e-8)
    else:
        recon_fut = torch.tensor(0.0, device=device)

    recon_loss = RECON_PAST_BETA * recon_past + RECON_FUT_BETA * recon_fut

    if use_cls_loss:
        _, B, N, C = logits.shape
        logits_flat = logits[-1].view(B * N, C)
        labels_flat = label_last.view(-1)
        valid = (mask_last.view(-1) > 0.5) & (labels_flat >= 0)
        cls_loss = F.cross_entropy(logits_flat[valid], labels_flat[valid], weight=class_weights) if valid.any() else torch.tensor(0.0, device=device)
    else:
        cls_loss = torch.tensor(0.0, device=device)

    if use_cls_loss:
        total_loss = cls_loss if cls_only else CLS_ALPHA * cls_loss + recon_loss
    else:
        total_loss = recon_loss

    return total_loss, recon_past, recon_fut, cls_loss, mem


def train_loop(model, dataloader, optimizer, device, epoch_idx, use_cls_loss, cls_only=False, class_weights=None):
    model.train()
    class_weights = class_weights.to(device) if class_weights is not None else None
    totals = np.zeros(4, dtype=float)
    mem = None

    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        loss, recon_past, recon_fut, cls_loss, mem = _loss_step(model, batch, device, mem, use_cls_loss, cls_only, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        values = [loss.item(), recon_past.item(), recon_fut.item(), cls_loss.item()]
        totals += np.array(values)
        print(
            f"\r🚀 Epoch {epoch_idx} | Train {step}/{len(dataloader)} | "
            f"ReconPast={values[1]:.6f} ReconFut={values[2]:.6f} Cls={values[3]:.6f} Total={values[0]:.6f}",
            end="",
        )
    print()
    return tuple((totals / max(1, len(dataloader))).tolist())


@torch.no_grad()
def validate_loop(model, dataloader, device, epoch_idx, use_cls_loss, cls_only=False, class_weights=None):
    model.eval()
    class_weights = class_weights.to(device) if class_weights is not None else None
    totals = np.zeros(4, dtype=float)
    mem = None

    for step, batch in enumerate(dataloader, start=1):
        loss, recon_past, recon_fut, cls_loss, mem = _loss_step(model, batch, device, mem, use_cls_loss, cls_only, class_weights)
        values = [loss.item(), recon_past.item(), recon_fut.item(), cls_loss.item()]
        totals += np.array(values)
        print(
            f"\r🧪 Epoch {epoch_idx} | Valid {step}/{len(dataloader)} | "
            f"ReconPast={values[1]:.6f} ReconFut={values[2]:.6f} Cls={values[3]:.6f} Total={values[0]:.6f}",
            end="",
        )
    print()
    return tuple((totals / max(1, len(dataloader))).tolist())


# =============================================================================
# Main training pipeline
# =============================================================================

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    target_s = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    target_e = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    tickers = sorted(read_tickers_xlsx(TICKERS_XLSX))
    if MAX_STOCKS is not None:
        tickers = tickers[:MAX_STOCKS]
    if not tickers:
        print("❌ Ticker list is empty.")
        return

    num_stocks = len(tickers)
    print(f"✅ Training stocks: {num_stocks}")
    with open(TRAIN_TICKERS_JSON, "w", encoding="utf-8") as f:
        json.dump(tickers, f, ensure_ascii=False, indent=2)

    stock_df_map = {}
    data_min = data_max = None
    for path in find_stock_files(DATA_DIR, tickers):
        code = os.path.basename(path).split("_")[0].upper()
        df = load_csv_df(path)
        if df.empty:
            continue
        stock_df_map[code] = df
        dmin, dmax = df["date"].dt.date.min(), df["date"].dt.date.max()
        data_min = dmin if data_min is None or dmin < data_min else data_min
        data_max = dmax if data_max is None or dmax > data_max else data_max

    if data_min is None or data_max is None:
        print("❌ No usable CSV data found.")
        return

    inter = intersect_range((target_s, target_e), data_min, data_max)
    if inter is None:
        print(f"⚠️ Target range {target_s}~{target_e} does not overlap data range {data_min}~{data_max}.")
        return
    inter_s, inter_e = inter
    train_ranges = [inter]
    print(f"🧩 Training date intersection: {inter_s} ~ {inter_e}")

    existing_model, model_s, model_e = find_existing_model(SAVE_DIR)
    if existing_model:
        print(f"📁 Existing model found: {existing_model} ({model_s} ~ {model_e})")

    tasks = [(code, df, train_ranges) for code, df in stock_df_map.items()]
    n_workers = min(len(tasks), mp.cpu_count() or 1)
    print(f"⚙️ Label generation workers: {n_workers}")

    label_map = {}
    with mp.Pool(processes=n_workers) as pool:
        for code, df_label in pool.imap_unordered(compute_labels_for_one_stock, tasks):
            if df_label is not None:
                label_map[code] = df_label
    print(f"🧠 Generated labels for {len(label_map)} stocks.")

    global_times = []
    for start_date, end_date in train_ranges:
        bucket = set()
        for df in stock_df_map.values():
            mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
            if mask.any():
                bucket.update(df.loc[mask, "date"].tolist())
        global_times.extend(sorted(bucket))

    global_times = sorted(set(global_times))
    if len(global_times) < SEQ_LEN:
        print(f"⚠️ Unified timeline length {len(global_times)} is shorter than SEQ_LEN={SEQ_LEN}.")
        return
    print(f"🕒 Unified timeline length: {len(global_times)} minutes")

    time_index = pd.Index(global_times)
    t_len = len(global_times)
    f_dim = len(NUM_COLS)
    full_X_raw = np.full((t_len, num_stocks, f_dim), np.nan, dtype=np.float32)
    full_mask = np.zeros((t_len, num_stocks), dtype=np.float32)
    full_label_raw = np.full((t_len, num_stocks), -1, dtype=np.int64)

    for stock_idx, code in enumerate(tickers):
        df = stock_df_map.get(code)
        if df is None or df.empty:
            continue
        aligned = df.set_index("date").reindex(time_index)
        full_mask[:, stock_idx] = (~aligned["close"].isna()).to_numpy(dtype=np.float32)
        full_X_raw[:, stock_idx, :] = aligned[NUM_COLS].to_numpy(dtype=np.float32)

        if code in label_map:
            lab = label_map[code].set_index("date").reindex(time_index)["dp_label"].to_numpy(dtype=float)
            full_label_raw[:, stock_idx] = np.where(np.isnan(lab), -1, lab).astype(np.int64)

    labels_flat = full_label_raw.reshape(-1)
    valid_label_mask = labels_flat > 0
    class_weights = None
    if valid_label_mask.any():
        counts = np.bincount(labels_flat[valid_label_mask], minlength=NUM_CLASSES)
        total = counts.sum()
        safe_counts = np.maximum(counts.astype(np.float64), 1e-6)
        class_weights = torch.tensor((total / (NUM_CLASSES * safe_counts)).astype(np.float32))
        print(f"📊 Classification labels used in loss: Buy={counts[1]} Sell={counts[2]}")
        print(f"📊 Class weights: {class_weights.numpy()}")
    else:
        print("⚠️ No Buy/Sell labels found. Classification loss will be zero.")

    local_time = pd.DatetimeIndex(global_times).tz_convert("Australia/Sydney")
    anchor = local_time.normalize() + pd.Timedelta(hours=9, minutes=59)
    full_T = ((local_time - anchor) / pd.Timedelta(hours=1)).to_numpy(np.float32).reshape(-1, 1)

    total_steps = max(0, t_len - SEQ_LEN - FUTURE_HORIZON + 1)
    if total_steps < 2:
        print("⚠️ Not enough samples for train/validation split.")
        return

    n_tr_steps = int(0.8 * total_steps)
    train_last_time_idx = n_tr_steps + SEQ_LEN + FUTURE_HORIZON - 2

    X_flat = full_X_raw.reshape(-1, f_dim)
    rows_mask_time = np.repeat(np.arange(t_len) <= train_last_time_idx, repeats=num_stocks)
    train_rows = X_flat[rows_mask_time]
    valid_rows = ~np.isnan(train_rows).any(axis=1)

    if os.path.exists(SCALER_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        print(f"📦 Reusing scaler: {SCALER_PATH}")
    else:
        if valid_rows.sum() == 0:
            print("❌ No valid rows available to fit scaler.")
            return
        scaler = StandardScaler().fit(train_rows[valid_rows])
        joblib.dump(scaler, SCALER_PATH)
        print(f"📊 Saved scaler: {SCALER_PATH}")

    X_std = (X_flat - scaler.mean_.reshape(1, f_dim)) / scaler.scale_.reshape(1, f_dim)
    np.nan_to_num(X_std, copy=False, nan=0.0)
    full_X = X_std.reshape(full_X_raw.shape).astype(np.float32)

    dataset = AlignedDataset(full_X, full_mask, full_T, SEQ_LEN, full_label_raw, fut_len=FUTURE_HORIZON)
    ds_tr = Subset(dataset, list(range(n_tr_steps)))
    ds_va = Subset(dataset, list(range(n_tr_steps, len(dataset))))

    ld_tr = DataLoader(ds_tr, batch_sampler=TXLStreamSampler(len(ds_tr), BATCH_SIZE, SEQ_LEN, drop_last=True), collate_fn=collate_fn, pin_memory=True)
    ld_va = DataLoader(ds_va, batch_sampler=TXLStreamSampler(len(ds_va), BATCH_SIZE, SEQ_LEN, drop_last=True), collate_fn=collate_fn, pin_memory=True)

    device = choose_device()
    print("Using device:", device)

    model = MultiStockTXLAligned(
        in_feats=f_dim,
        num_stocks=num_stocks,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        mem_len=MEM_LEN,
        time_dim=TIME_DIM,
        stock_emb=STOCK_EMB,
        dropout=DROPOUT,
        pred_len=FUTURE_HORIZON,
    ).to(device)

    if existing_model:
        try:
            state = torch.load(existing_model, map_location=device)
            model_dict = model.state_dict()
            compatible = {k: v for k, v in state.items() if k in model_dict and not k.startswith("cls_head")}
            model_dict.update(compatible)
            model.load_state_dict(model_dict)
            nn.init.xavier_uniform_(model.cls_head.weight)
            if model.cls_head.bias is not None:
                nn.init.zeros_(model.cls_head.bias)
            print(f"✅ Loaded compatible backbone weights from {existing_model}; classification head reinitialised.")
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Failed to partially load existing model; training from scratch: {exc}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    best_cls = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    try:
        epoch = 0
        while True:
            epoch += 1
            tr_total, tr_recon_past, tr_recon_fut, tr_cls = train_loop(
                model, ld_tr, optimizer, device, epoch, True, False, class_weights
            )
            va_total, va_recon_past, va_recon_fut, va_cls = validate_loop(
                model, ld_va, device, epoch, True, False, class_weights
            )

            print(
                f"Joint Epoch {epoch} — "
                f"Train recon_past={tr_recon_past:.6f}, recon_fut={tr_recon_fut:.6f}, cls={tr_cls:.6f} | "
                f"Valid recon_past={va_recon_past:.6f}, recon_fut={va_recon_fut:.6f}, cls={va_cls:.6f}"
            )

            if va_cls < best_cls - MIN_IMPROVE:
                best_cls = va_cls
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                print(f"⚠️ Validation cls did not improve: current={va_cls:.6f}, best={best_cls:.6f}, patience={no_improve}/{EARLY_STOP_PATIENCE}")

            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"⛔ Early stopping. Best epoch={best_epoch}, best val cls={best_cls:.6f}")
                break
    except KeyboardInterrupt:
        print(f"\n🛑 Interrupted. Saving best available classification state from epoch {best_epoch}.")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"✅ Restored best classification state from epoch {best_epoch}.")

    if existing_model and model_s is not None and model_e is not None:
        new_s, new_e = min(model_s, inter_s), max(model_e, inter_e)
    else:
        new_s, new_e = inter_s, inter_e

    final_path = os.path.join(SAVE_DIR, f"txl_model_multi_{new_s.strftime('%Y%m%d')}_{new_e.strftime('%Y%m%d')}.pth")
    torch.save(model.state_dict(), final_path)

    for old_path in glob.glob(os.path.join(SAVE_DIR, "txl_model_multi_*.pth")):
        if os.path.abspath(old_path) != os.path.abspath(final_path):
            try:
                os.remove(old_path)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Failed to remove old model {old_path}: {exc}")

    print(f"💾 Saved final model: {final_path}")


if __name__ == "__main__":
    main()
