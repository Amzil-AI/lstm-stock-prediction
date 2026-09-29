# LSTM Stock Price Prediction (AAPL)

Pedagogical solution for **Master 2 Finance, Data et IA** — Deep Learning course.

**Repo:** https://github.com/LikhitaYerra/lstm-stock-prediction  
*(intended home under `Amzil-AI/lstm-stock-prediction` after org transfer)*

Predict Apple's next-day adjusted close with a small LSTM, using a chronological split and a train-only scaler. Compare against a naive baseline (yesterday's close).

> This is a teaching lab, not a trading system. A clean protocol matters more than beating the market.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_lstm.py
```

## What the script does

1. Downloads (or reloads) daily AAPL OHLCV via `yfinance` (2018–2025)
2. Builds sliding windows of 60 days → target = next-day Close
3. Splits **chronologically** 70% / 15% / 15% (never shuffle)
4. Fits `MinMaxScaler` on the **train period only**
5. Trains a 2-layer LSTM (PyTorch) with early stopping on validation MSE
6. Reports RMSE, MAE, MAPE, directional accuracy vs naive baseline
7. Writes `figures/` and `outputs/metrics.json`

## Outputs

| Path | Content |
|---|---|
| `data/AAPL.csv` | Cached prices |
| `figures/training_curves.png` | Train / val loss |
| `figures/test_predictions.png` | True vs LSTM vs naive |
| `outputs/metrics.json` | Test metrics |
| `outputs/lstm_aapl.pt` | Weights + scaler bounds |

## Course context

Warm-up before the **Signal Desk** case. Same discipline (no leakage, chronological evaluation), simpler question (price level, one ticker, one architecture).
