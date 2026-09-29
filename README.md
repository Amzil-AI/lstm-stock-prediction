# LSTM Stock Price Prediction (AAPL)

Pedagogical solution for **Master 2 Finance, Data et IA** — Deep Learning course.

**Repo:** https://github.com/Amzil-AI/lstm-stock-prediction

Two tasks on Apple daily data:

- **Part A** — next-day **Close** (OHLCV windows) vs naive yesterday close  
- **Part B** — next-day **return** with engineered features + **3-fold walk-forward** vs zero-return baseline  

> Teaching lab, not a trading system. Clean protocol matters more than beating the market.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_lstm.py
```

## What the script does

1. Downloads / reloads AAPL OHLCV (`yfinance`, 2018–2025)
2. **Part A:** 60-day windows → next Close; chronological 70/15/15; MinMax on train only; LSTM vs naive
3. **Part B:** features `ret_1`, `ret_5`, `vol_20`, `hl_range`, `vol_chg` → next return; 3 expanding walk-forward folds
4. Writes figures + `outputs/metrics.json`

## Outputs

| Path | Content |
|---|---|
| `figures/test_predictions.png` | Part A: true vs LSTM vs naive |
| `figures/test_returns.png` | Part B: last fold returns |
| `figures/training_curves*.png` | Loss curves |
| `outputs/metrics.json` | All metrics |

## Course context

Warm-up before **Signal Desk** (direction + costs). On Close level, naive often wins; on returns, directional accuracy stays near chance — that is an acceptable, well-documented result.
