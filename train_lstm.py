"""LSTM lab on AAPL — Master 2 Finance, Data et IA (solution).

Part A — next-day Close (raw OHLCV) vs naive yesterday close.
Part B — next-day return with engineered features + 3-fold walk-forward.

Not a trading system. Chronological splits only. Scalers fit on train only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

TICKER = "AAPL"
START = "2018-01-01"
END = "2025-12-31"
WINDOW = 60
OHLCV = ["Close", "High", "Low", "Open", "Volume"]
BATCH_SIZE = 64
EPOCHS = 35
LR = 1e-3
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
SEED = 42
VOL_WINDOW = 20
N_FOLDS = 3


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_prices() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{TICKER}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    else:
        raw = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[OHLCV].dropna().copy()
        df.index.name = "Date"
        df.to_csv(path)
    return df[OHLCV].dropna().astype(float)


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Engineered features known at close J (no J+1 leakage)."""
    close = prices["Close"]
    feat = pd.DataFrame(index=prices.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_5"] = close.pct_change(5)
    feat["vol_20"] = feat["ret_1"].rolling(VOL_WINDOW).std()
    feat["hl_range"] = (prices["High"] - prices["Low"]) / close
    feat["vol_chg"] = prices["Volume"].pct_change(1)
    feat["target_ret"] = close.pct_change(1).shift(-1)  # return from J to J+1
    feat["target_close"] = close.shift(-1)
    feat["prev_close"] = close
    return feat.dropna()


def make_xy(
    feat: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int = WINDOW,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    values = feat[feature_cols].values.astype(np.float64)
    target = feat[target_col].values.astype(np.float64)
    dates = feat.index
    x, y, d = [], [], []
    for i in range(len(feat) - window):
        # window ends at day i+window-1; target is that day's target_* (J+1 from last day)
        end = i + window - 1
        x.append(values[i : i + window])
        y.append(target[end])
        d.append(dates[end])
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        pd.DatetimeIndex(d),
    )


def walk_forward_slices(n: int, n_folds: int = N_FOLDS) -> list[tuple[slice, slice, slice]]:
    """Expanding train, then val, then test — three chronological folds."""
    min_train = int(n * 0.45)
    block = (n - min_train) // (n_folds * 2)  # each fold: val block + test block roughly
    if block < 40:
        block = max(30, (n - min_train) // (n_folds * 2))
    folds = []
    cursor = min_train
    for _ in range(n_folds):
        val_start = cursor
        val_end = min(val_start + block, n)
        test_start = val_end
        test_end = min(test_start + block, n)
        if test_end - test_start < 20:
            break
        folds.append((slice(0, val_start), slice(val_start, val_end), slice(test_start, test_end)))
        cursor = test_end
    return folds


class PriceLSTM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=DROPOUT if LAYERS > 1 else 0.0,
        )
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, wait = math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tr = []
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr.append(loss.item())
        model.eval()
        va = []
        with torch.no_grad():
            for xb, yb in val_loader:
                va.append(loss_fn(model(xb), yb).item())
        tr_m, va_m = float(np.mean(tr)), float(np.mean(va))
        history["train_loss"].append(tr_m)
        history["val_loss"].append(va_m)
        if va_m < best_val - 1e-8:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 7:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model, loader) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for xb, _ in loader:
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def metrics_level(y_true, y_pred) -> dict:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    if len(y_true) > 1:
        true_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
        directional = float(np.mean(true_dir == pred_dir) * 100)
    else:
        directional = float("nan")
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "directional_acc_pct": directional}


def metrics_return(y_true, y_pred) -> dict:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # Direction of return vs zero (up / down day)
    directional = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    # Naive return baseline is 0
    naive = np.zeros_like(y_true)
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, naive)))
    return {
        "rmse": rmse,
        "mae": mae,
        "directional_acc_pct": directional,
        "naive_zero_rmse": naive_rmse,
    }


def scale_windows(X, y, train_sl, val_sl, test_sl):
    """Scale features with train rows only; scale target with train targets only."""
    n_feat = X.shape[-1]
    flat_train = X[train_sl].reshape(-1, n_feat)
    x_scaler = StandardScaler().fit(flat_train)
    y_scaler = StandardScaler().fit(y[train_sl])

    def transform_x(sl):
        shape = X[sl].shape
        return x_scaler.transform(X[sl].reshape(-1, n_feat)).reshape(shape).astype(np.float32)

    def transform_y(sl):
        return y_scaler.transform(y[sl]).astype(np.float32)

    def loader(sl, shuffle=False):
        xt = torch.tensor(transform_x(sl))
        yt = torch.tensor(transform_y(sl))
        return DataLoader(TensorDataset(xt, yt), batch_size=BATCH_SIZE, shuffle=shuffle)

    return loader(train_sl), loader(val_sl), loader(test_sl), x_scaler, y_scaler


def invert_y(arr, y_scaler):
    return y_scaler.inverse_transform(arr)


def plot_curves(history, path, title):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_level(dates, y_true, y_pred, y_naive, path):
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(dates, y_true, label="True close", lw=1.5)
    ax.plot(dates, y_pred, label="LSTM", lw=1.3)
    ax.plot(dates, y_naive, label="Naive (yesterday)", lw=1.1, alpha=0.85)
    ax.set_title("Part A — AAPL next-day close (hold-out test)")
    ax.set_ylabel("USD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_returns(dates, y_true, y_pred, path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(dates, y_true, label="True return", lw=1.0, alpha=0.85)
    ax.plot(dates, y_pred, label="LSTM return", lw=1.0, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Part B — next-day return (last walk-forward fold)")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_part_a(feat: pd.DataFrame) -> dict:
    print("\n=== PART A: next-day Close (raw-style levels via prev_close path) ===")
    # Use OHLCV-like engineered stand-ins: we rebuild from prices for clarity
    prices = download_prices()
    # Align to feat index for consistency
    prices = prices.loc[feat.index.min() : feat.index.max()].copy()
    # Build level windows from OHLCV
    from sklearn.preprocessing import MinMaxScaler

    raw = prices[OHLCV].values.astype(np.float64)
    close_idx = OHLCV.index("Close")
    X, y, dates = [], [], []
    for i in range(len(prices) - WINDOW):
        X.append(raw[i : i + WINDOW])
        y.append(raw[i + WINDOW, close_idx])
        dates.append(prices.index[i + WINDOW])
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32).reshape(-1, 1)
    dates = pd.DatetimeIndex(dates)

    n = len(X)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    train_sl, val_sl, test_sl = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)

    # MinMax on train period rows of price table
    train_end_row = WINDOW + n_train
    scaler = MinMaxScaler().fit(prices.iloc[:train_end_row].values)
    scaled = scaler.transform(prices.values)
    Xs, ys = [], []
    for i in range(len(prices) - WINDOW):
        Xs.append(scaled[i : i + WINDOW])
        ys.append(scaled[i + WINDOW, close_idx])
    Xs = np.asarray(Xs, np.float32)
    ys = np.asarray(ys, np.float32).reshape(-1, 1)

    def pack(sl):
        return DataLoader(
            TensorDataset(torch.tensor(Xs[sl]), torch.tensor(ys[sl])),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    model = PriceLSTM(n_features=len(OHLCV))
    hist = train_model(model, pack(train_sl), pack(val_sl))
    print(f"Part A epochs run: {len(hist['train_loss'])}")

    cmin, cmax = scaler.data_min_[close_idx], scaler.data_max_[close_idx]

    def inv(a):
        return a * (cmax - cmin) + cmin

    y_true = inv(ys[test_sl])
    y_pred = inv(predict(model, pack(test_sl)))
    test_dates = dates[test_sl]
    y_naive = prices["Close"].shift(1).loc[test_dates].values.reshape(-1, 1)
    mask = ~np.isnan(y_naive.ravel())
    y_true, y_pred, y_naive, test_dates = y_true[mask], y_pred[mask], y_naive[mask], test_dates[mask]

    lstm_m = metrics_level(y_true, y_pred)
    naive_m = metrics_level(y_true, y_naive)
    print("LSTM :", lstm_m)
    print("Naive:", naive_m)

    plot_curves(hist, FIG_DIR / "training_curves.png", "Part A — LSTM training (Close)")
    plot_level(test_dates, y_true, y_pred, y_naive, FIG_DIR / "test_predictions.png")
    torch.save({"model_state": model.state_dict(), "task": "close"}, OUT_DIR / "lstm_aapl_close.pt")
    return {
        "task": "next_day_close",
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": int(mask.sum()),
        "lstm": lstm_m,
        "naive_yesterday": naive_m,
    }


def run_part_b(feat: pd.DataFrame) -> dict:
    print("\n=== PART B: next-day return + engineered features + walk-forward ===")
    feature_cols = ["ret_1", "ret_5", "vol_20", "hl_range", "vol_chg"]
    X, y, dates = make_xy(feat, feature_cols, "target_ret", WINDOW)
    folds = walk_forward_slices(len(X), N_FOLDS)
    fold_reports = []

    for fi, (train_sl, val_sl, test_sl) in enumerate(folds, 1):
        print(f"\n--- Fold {fi}/{len(folds)}  train={train_sl.stop}  val={val_sl.stop - val_sl.start}  test={test_sl.stop - test_sl.start}")
        train_loader, val_loader, test_loader, _, y_scaler = scale_windows(X, y, train_sl, val_sl, test_sl)
        model = PriceLSTM(n_features=len(feature_cols))
        hist = train_model(model, train_loader, val_loader, epochs=EPOCHS)
        y_true = y[test_sl]
        y_pred = invert_y(predict(model, test_loader), y_scaler)
        m = metrics_return(y_true, y_pred)
        print("fold metrics:", m)
        fold_reports.append(
            {
                "fold": fi,
                "train_end": str(dates[train_sl][-1].date()),
                "test_start": str(dates[test_sl][0].date()),
                "test_end": str(dates[test_sl][-1].date()),
                "n_test": int(test_sl.stop - test_sl.start),
                "metrics": m,
                "epochs": len(hist["train_loss"]),
            }
        )
        if fi == len(folds):
            plot_curves(hist, FIG_DIR / "training_curves_returns.png", "Part B — last fold training (return)")
            plot_returns(dates[test_sl], y_true, y_pred, FIG_DIR / "test_returns.png")
            torch.save({"model_state": model.state_dict(), "task": "return"}, OUT_DIR / "lstm_aapl_return.pt")

    # Aggregate
    dir_acc = [f["metrics"]["directional_acc_pct"] for f in fold_reports]
    rmse = [f["metrics"]["rmse"] for f in fold_reports]
    naive_rmse = [f["metrics"]["naive_zero_rmse"] for f in fold_reports]
    summary = {
        "task": "next_day_return_walk_forward",
        "features": feature_cols,
        "n_folds": len(fold_reports),
        "mean_directional_acc_pct": float(np.mean(dir_acc)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_naive_zero_rmse": float(np.mean(naive_rmse)),
        "folds": fold_reports,
    }
    print("\nWalk-forward summary:", {k: summary[k] for k in summary if k != "folds"})
    return summary


def main() -> None:
    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prices = download_prices()
    print(f"AAPL rows={len(prices)}  {prices.index.min().date()} → {prices.index.max().date()}")
    feat = build_feature_frame(prices)
    print(f"Feature rows after lag/roll drop={len(feat)}")

    part_a = run_part_a(feat)
    part_b = run_part_b(feat)

    report = {
        "ticker": TICKER,
        "window": WINDOW,
        "part_a_close": part_a,
        "part_b_return_walk_forward": part_b,
        "teaching_note": (
            "On Close level, naive yesterday often wins. On returns, compare to zero-return "
            "baseline and directional accuracy near 50%. Clean protocol > claimed edge."
        ),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_DIR / 'metrics.json'}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
