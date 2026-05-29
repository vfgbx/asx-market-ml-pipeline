# ASX Market ML Pipeline

A Python-based research pipeline for collecting ASX stock market data, training a multi-stock Transformer-XL model, and testing quantitative trading ideas through feature engineering and backtesting.

This repository demonstrates an end-to-end workflow that connects:

1. Historical ASX market data collection
2. Incremental data updating
3. Multi-stock time-series alignment
4. Machine learning model training
5. Signal generation
6. Quantitative backtesting
7. Engineering-oriented automation

The project is mainly designed as a personal research and engineering portfolio project. It focuses on automation, data processing, model training, and experimental trading-signal research rather than financial advice.

---

## Project Structure

```text
asx-market-ml-pipeline/
├── README.md
├── requirements.txt
├── data/
│   └── ASX200000.xlsx
├── data_collection/
│   └── asx_1min_incremental_downloader.py
├── model_training/
│   └── txl_multi_train_aligned_with_actions.py
└── backtesting/
    └── asx_daily_energy_score_backtest.py
```

---

## 1. Data Collection

### File

```text
data_collection/asx_1min_incremental_downloader.py
```

This script downloads and updates historical 1-minute K-line data for ASX stocks.

It is designed to work with an ASX stock list stored in:

```text
data/ASX200000.xlsx
```

The Excel file should contain stock symbols in the first column, for example:

```text
CBA
BHP
WBC
NAB
CSL
```

---

## Important Requirement: Interactive Brokers API

This data downloader requires access to the **Interactive Brokers API**.

To use the downloader, you must have:

- An Interactive Brokers account
- Trader Workstation, also known as **TWS**, or IB Gateway installed
- Market data permissions for ASX stocks
- The Python package `ib_insync`
- TWS API access enabled
- TWS or IB Gateway running locally while the script is executed

The script connects to Interactive Brokers through:

```python
TWS_HOST = "127.0.0.1"
TWS_PORT = 7497
CLIENT_ID = 17
```

Typical IBKR ports are:

| Environment | Port |
|---|---:|
| TWS Paper Trading | 7497 |
| TWS Live Trading | 7496 |
| IB Gateway Paper Trading | 4002 |
| IB Gateway Live Trading | 4001 |

Before running the script, make sure the port in the code matches your TWS or IB Gateway setting.

---

## Data Collection Logic

The downloader uses an incremental update design.

### If no existing file is found

The script starts from the target update date and backfills historical 1-minute bars in rolling windows.

For example:

```text
Target date → previous 30 days → previous 30 days → previous 30 days ...
```

The process stops when:

- No more historical minute data is returned
- The configured stop date is reached
- The contract cannot be qualified through IBKR

The default stop date is:

```python
BACKFILL_STOP_DATE_STR = "2015-01-01"
```

---

### If an existing file is found

The script performs two actions:

1. Backward fill  
   It checks whether older missing data can be added before the current earliest timestamp.

2. Forward update  
   It appends new trading days until the file reaches the latest target trading date.

This avoids re-downloading the full dataset every time.

---

## File Naming Convention

Downloaded files are saved using the following pattern:

```text
SYMBOL_ASX_1min_START_END.csv
```

Example:

```text
CBA_ASX_1min_20200102_1000_20251001_1610.csv
```

This makes it easy to identify:

- Stock symbol
- Exchange
- Bar interval
- Start time
- End time

---

## ASX Market Closing-Time Cleaning

The downloader also includes a cleaning rule for ASX market close data.

### Regular trading days

For normal trading days:

- Keep all bars before 16:00
- Keep the 16:10 bar
- Remove other bars at or after 16:00

Conceptually:

```text
Keep:    09:59 → 15:59, 16:10
Remove: 16:00, 16:01, 16:02, ..., except 16:10
```

### Early close days

For early close days, such as Christmas Eve and New Year’s Eve, adjusted to the previous business day if they fall on a weekend:

- Keep all bars before 14:00
- Keep the 14:10 bar
- Remove other bars at or after 14:00

Conceptually:

```text
Keep:    09:59 → 13:59, 14:10
Remove: 14:00, 14:01, 14:02, ..., except 14:10
```

This cleaning step is useful because exchange data may contain post-close settlement or auction-related bars that can distort intraday model training.

---

## 2. Multi-Stock Transformer-XL Training

### File

```text
model_training/txl_multi_train_aligned_with_actions.py
```

This script trains a multi-stock time-series model using a Transformer-XL-style architecture.

The model is designed to process many ASX stocks on a unified time axis.

---

## Model Training Objective

The training pipeline combines three learning tasks:

1. Past-window reconstruction
2. Future-window prediction
3. Buy / sell classification

The model therefore learns both:

- General market structure from price-volume sequences
- Forward-looking movement patterns
- Classification signals for buy and sell zones

---

## Input Features

Each stock uses the following numerical features:

```python
NUM_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "average",
    "barcount"
]
```

For each time step, the input tensor is shaped as:

```text
[time, stock, feature]
```

After batching, the model receives:

```text
[S, B, N, F]
```

Where:

| Symbol | Meaning |
|---|---|
| `S` | Sequence length |
| `B` | Batch size |
| `N` | Number of stocks |
| `F` | Number of features |

---

## Unified Multi-Stock Time Axis

Different ASX stocks may have missing bars or different available histories.

To train them together, the script builds a global timestamp index:

```text
global_times = union of all stock timestamps
```

Then each stock is reindexed onto this shared timeline.

If a stock has no data at a timestamp, that value is treated as missing and replaced with zero after standardisation. A separate mask is used to tell the model whether the data is real or missing.

This creates two key matrices:

```text
full_X     → price and volume features
full_mask  → whether each stock has valid data at each timestamp
```

The mask prevents the model from treating missing values as real market observations.

---

## Time Encoding: Time2Vec

The model uses a Time2Vec module to encode intraday time.

The time feature is measured relative to the market opening anchor:

```text
09:59 Australia/Sydney time
```

The Time2Vec layer creates both linear and periodic time features.

The output is:

```python
Time2Vec(t) = [linear(t), sin(periodic_1(t)), ..., sin(periodic_k(t))]
```

This helps the model understand repeating intraday patterns, such as:

- Opening volatility
- Midday quiet periods
- Closing auction effects
- Time-of-day liquidity behaviour

---

## Stock Embedding

Each stock receives its own learnable embedding vector.

This allows the model to learn stock-specific behaviour, for example:

- Different volatility profiles
- Different liquidity levels
- Different price movement patterns
- Sector-specific behaviour

The stock embedding is concatenated with the input features and time encoding.

The final model input contains:

```text
market features + time encoding + stock embedding + missing-data mask
```

---

## Transformer-XL Style Memory

The model supports memory from previous segments.

Instead of treating each sequence independently, the model can carry hidden states from the previous window into the next one.

This helps preserve temporal context beyond the fixed sequence length.

Important parameters:

```python
SEQ_LEN = 64
MEM_LEN = 128
```

This means:

- The model directly sees 64 recent time steps
- It can also keep up to 128 previous hidden states as memory

Conceptually:

```text
previous memory + current sequence → Transformer encoder → updated memory
```

---

## Classification Labels

The training script creates three possible labels:

```text
0 = Hold
1 = Buy
2 = Sell
```

However, the current classification loss only focuses on buy and sell labels.

Hold labels are generated but not used in the classification loss in the current training setup.

---

## Super-K Labeling Logic

The model uses a future-window labeling method.

For each time step `t`, the script looks ahead over a future window:

```python
SUPER_K_HORIZON = 100
```

It calculates a future EMA over the close price and checks whether enough future EMA values are above or below the current price threshold.

### Buy condition

A buy label is assigned if enough future EMA values are higher than:

```text
current_close × (1 + BUY_THRESHOLD)
```

In formula form:

```text
future_ema >= current_close × (1 + BUY_THRESHOLD)
```

If this happens at least a required number of times inside the future horizon, the point is labelled as:

```text
Buy = 1
```

---

### Sell condition

A sell label is assigned if enough future EMA values are lower than:

```text
current_close × (1 - SELL_THRESHOLD)
```

In formula form:

```text
future_ema <= current_close × (1 - SELL_THRESHOLD)
```

If this happens often enough inside the future horizon, the point is labelled as:

```text
Sell = 2
```

---

### Hold condition

If neither the buy nor sell condition is met, the label is:

```text
Hold = 0
```

---

## Reconstruction Loss

The model reconstructs the past input window.

This is similar to asking:

```text
Can the model understand and rebuild the recent market structure?
```

The reconstruction target is the original feature matrix.

The loss is calculated only on valid, non-missing market data using the mask.

Smooth L1 loss is used:

```python
F.smooth_l1_loss(...)
```

Conceptually:

```text
reconstruction_loss = SmoothL1(predicted_past, real_past)
```

---

## Future Prediction Loss

The model also predicts future market features for the next `FUTURE_HORIZON` steps.

```python
FUTURE_HORIZON = 30
```

The model uses the final hidden state of the historical window to predict the next 30 bars.

Conceptually:

```text
last_hidden_state → future_head → predicted_future_window
```

The future prediction loss is also calculated using Smooth L1 loss and the valid-data mask.

```text
future_loss = SmoothL1(predicted_future, real_future)
```

---

## Classification Loss

For buy and sell prediction, the model uses cross-entropy loss.

```python
F.cross_entropy(...)
```

Only valid buy and sell labels are included in the classification loss.

Class weighting is also calculated to reduce the effect of label imbalance.

Conceptually:

```text
classification_loss = CrossEntropy(predicted_action, buy_or_sell_label)
```

---

## Total Training Loss

The total loss combines reconstruction, future prediction, and classification:

```text
total_loss =
    CLS_ALPHA × classification_loss
  + RECON_PAST_BETA × past_reconstruction_loss
  + RECON_FUT_BETA × future_prediction_loss
```

In code:

```python
loss = CLS_ALPHA * cls_loss + recon_loss
```

Where:

```text
recon_loss =
    RECON_PAST_BETA × recon_past
  + RECON_FUT_BETA × recon_fut
```

This multi-task setup encourages the model to learn both:

- Market representation
- Future movement structure
- Trading-action classification

---

## Early Stopping

The model uses early stopping based on validation classification loss.

The logic is:

```text
If validation classification loss does not improve for N epochs, stop training.
```

The default patience is:

```python
EARLY_STOP_PATIENCE = 10
```

This helps avoid overfitting.

---

## 3. Energy-Score Backtesting

### File

```text
backtesting/asx_daily_energy_score_backtest.py
```

This script tests a separate rule-based trading idea using daily ASX stock data.

It does not use the Transformer model directly. Instead, it calculates a custom trend-energy score from EMA relationships and backtests a simple state-based strategy.

---

## Energy Score Formula

The energy score is based on the difference between neighbouring exponential moving averages.

For each EMA span `p`, the score adds:

```text
(EMA(p) - EMA(p + 1)) / abs(EMA(p + 1)) × ln(p + 1)
```

The full score is:

```text
score =
Σ from p = EMA_MIN to EMA_MAX - 1
[
    (EMA_p - EMA_(p+1)) / |EMA_(p+1)| × ln(p + 1)
]
```

In code, the default EMA range is:

```python
ENERGY_EMA_MIN = 180
ENERGY_EMA_MAX = 200
```

This formula attempts to measure the slope and separation structure of a group of long-range EMAs.

---

## Intuition Behind the Energy Score

If shorter EMAs are consistently above longer EMAs, the score tends to be positive.

This suggests upward trend pressure.

If shorter EMAs are consistently below longer EMAs, the score tends to be negative.

This suggests downward trend pressure.

The logarithmic weight:

```text
ln(p + 1)
```

gives slightly different importance to different EMA spans.

---

## Score EMA Fast / Slow Lines

After calculating the raw energy score, the script applies two EMAs to the score itself:

```python
SCORE_EMA_FAST = 140
SCORE_EMA_SLOW = 150
```

This creates:

```text
score_ema_fast
score_ema_slow
```

These two smoothed lines are used to generate trading states.

---

## Backtesting Strategy

The strategy is state-based.

### Buy rule

If there is no current position and:

```text
score_ema_fast > score_ema_slow
```

Then the strategy buys the stock.

The script buys the maximum integer number of shares not exceeding:

```python
MIN_NOTIONAL = 10000.0
```

Quantity is calculated as:

```text
quantity = floor(10000 / close_price)
```

---

### Sell rule

If there is an existing position and:

```text
score_ema_fast < score_ema_slow
```

Then the strategy sells the full position.

---

## Transaction Fee

The backtest includes transaction fees.

The default fee rate is:

```python
FEE_RATE = 8.8 / 10000.0
```

That is equivalent to:

```text
0.00088
```

For each transaction:

```text
fee = notional × fee_rate
```

---

## Profit and Loss Calculation

When buying, the fee is deducted immediately.

When selling, realised profit is calculated as:

```text
realised_pnl =
    quantity × (sell_price - entry_price) - sell_fee
```

Total net PnL includes both buy-side and sell-side fees.

---

## Output

The backtesting script prints:

- Number of files scanned
- Number of files used
- Number of entries
- Number of exits
- Number of closed trades
- Net profit and loss after fees
- Total fees
- Average PnL per trade

It also generates:

```text
per_stock_summary.csv
per_stock_charts/
```

Each stock chart marks:

- Buy points
- Sell points
- Close-price movement
- Net PnL
- Number of closed trades

---

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib torch scikit-learn joblib openpyxl ib_insync
```

For the data downloader, you also need Interactive Brokers TWS or IB Gateway installed and running.

---

## Example Usage

### 1. Download or update ASX 1-minute data

Make sure TWS or IB Gateway is running first.

Then run:

```bash
python data_collection/asx_1min_incremental_downloader.py
```

---

### 2. Train the Transformer-XL model

After enough 1-minute CSV data has been downloaded, run:

```bash
python model_training/txl_multi_train_aligned_with_actions.py
```

---

### 3. Run the energy-score backtest

After preparing daily stock CSV files, run:

```bash
python backtesting/asx_daily_energy_score_backtest.py
```

---

## Configuration Notes

The scripts currently use local absolute paths.

Before running the project on another machine, update paths such as:

```python
DATA_DIR
SAVE_DIR
TICKERS_XLSX
EXCEL_FILE
CSV_DIR
```

For example:

```python
DATA_DIR = "path/to/your/data"
TICKERS_XLSX = "path/to/ASX200000.xlsx"
SAVE_DIR = "path/to/output"
```

---

## Research Notes

This repository contains experimental research code.

The purpose is to explore:

- Automated market data engineering
- Incremental historical data collection
- Multi-stock time-series modelling
- Transformer-based representation learning
- Rule-based trend-energy signal design
- Backtesting and performance visualisation

The scripts are designed to be understandable and modifiable rather than packaged as a production trading system.

---

## Limitations

This project has several important limitations:

1. It is not financial advice.
2. Backtest results do not guarantee future performance.
3. Market data quality depends on Interactive Brokers data availability and permissions.
4. The current model training pipeline is experimental.
5. The strategy does not currently include advanced portfolio-level risk management.
6. Slippage, liquidity constraints, tax, and order execution uncertainty may not be fully modelled.
7. Local absolute paths need to be changed before running on another machine.

---

## Disclaimer

This repository is for educational, research, and portfolio demonstration purposes only.

Nothing in this project should be interpreted as financial advice, investment recommendation, or a production-ready trading system.

Use at your own risk.

---
