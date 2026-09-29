"""LSTM next-day close prediction on Apple (AAPL) — course solution.

Pedagogical demo for Master 2 Finance, Data et IA.
Not a trading system. Chronological split only. Scaling fit on train only.
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

TICKER = "AAPL"
START = "2018-01-01"
END = "2025-12-31"
WINDOW = 60
FEATURES = ["Close", "High", "Low", "Open", "Volume"]
TARGET_COL = "Close"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.2
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def download_prices(ticker: str = TICKER, start: str = START, end: str = END) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{ticker}.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    else:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[FEATURES].dropna().copy()
        df.index.name = "Date"
        df.to_csv(path)
    df = df[FEATURES].dropna().astype(float)
    return df


def make_windows(values: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """X[t] = values[t:t+window], y[t] = Close at t+window (next day after window)."""
    x, y = [], []
    close_idx = FEATURES.index(TARGET_COL)
    for i in range(len(values) - window):
        x.append(values[i : i + window])
        y.append(values[i + window, close_idx])
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32).reshape(-1, 1)


def chronological_split(n: int) -> tuple[slice, slice, slice]:
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)
    return train, val, test


class PriceLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, layers: int = LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> dict:
    device = torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val = math.inf
    best_state = None
    patience, wait = 8, 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"epoch {epoch:02d}  train={tr:.6f}  val={va:.6f}")

        if va < best_val - 1e-6:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    # Directional: sign of day-to-day change vs previous true close
    if len(y_true) > 1:
        true_dir = np.sign(y_true[1:] - y_true[:-1])
        pred_dir = np.sign(y_pred[1:] - y_true[:-1])
        directional = float(np.mean(true_dir == pred_dir) * 100)
    else:
        directional = float("nan")
    return {"rmse": rmse, "mae": mae, "mape_pct": mape, "directional_acc_pct": directional}


def plot_curves(history: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled)")
    ax.set_title("LSTM training curves — AAPL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(dates, y_true, y_pred, y_naive, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(dates, y_true, label="True close", linewidth=1.6)
    ax.plot(dates, y_pred, label="LSTM", linewidth=1.4)
    ax.plot(dates, y_naive, label="Naive (yesterday)", linewidth=1.2, alpha=0.8)
    ax.set_title("AAPL next-day close — test set")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading / loading AAPL…")
    prices = download_prices()
    print(f"rows={len(prices)}  from={prices.index.min().date()}  to={prices.index.max().date()}")

    # Build windows on raw values first to keep date alignment, then scale with train-only scaler.
    raw = prices.values.astype(np.float64)
    X_raw, y_raw = make_windows(raw, WINDOW)
    dates = prices.index[WINDOW:]  # target date = day after window
    assert len(dates) == len(y_raw)

    train_sl, val_sl, test_sl = chronological_split(len(X_raw))

    # Fit scaler on TRAIN windows only (all features flattened across time of train samples' source rows).
    # Safer: fit on train period rows of the price table.
    train_end_row = WINDOW + train_sl.stop  # exclusive end index in prices for last train target
    scaler = MinMaxScaler()
    scaler.fit(prices.iloc[:train_end_row].values)

    scaled = scaler.transform(prices.values)
    X, y = make_windows(scaled, WINDOW)

    def pack(sl: slice):
        xt = torch.tensor(X[sl])
        yt = torch.tensor(y[sl])
        return DataLoader(TensorDataset(xt, yt), batch_size=BATCH_SIZE, shuffle=False)

    train_loader = pack(train_sl)
    val_loader = pack(val_sl)
    test_loader = pack(test_sl)

    model = PriceLSTM(n_features=len(FEATURES))
    print("Training LSTM…")
    history = train_model(model, train_loader, val_loader)

    # Predictions in scaled space → invert Close only
    close_min = scaler.data_min_[FEATURES.index(TARGET_COL)]
    close_max = scaler.data_max_[FEATURES.index(TARGET_COL)]

    def invert_close(arr: np.ndarray) -> np.ndarray:
        return arr * (close_max - close_min) + close_min

    y_test_true = invert_close(y[test_sl])
    y_test_pred = invert_close(predict(model, test_loader))

    # Naive baseline: predict previous day's true close (in original dollars)
    # For target at index i (in windowed series), previous close is prices Close at dates[i]-1 trading day
    # = prices.loc[dates[i]] previous row close ≈ values at WINDOW-1 offset within each window's last close
    # Equivalent: y_true shifted by 1 within test — use actual previous close from price table.
    test_dates = dates[test_sl]
    prev_close = prices["Close"].shift(1).loc[test_dates].values.reshape(-1, 1)
    # First test point may be nan if alignment fails — drop if needed
    mask = ~np.isnan(prev_close.ravel())
    y_true_m = y_test_true[mask]
    y_pred_m = y_test_pred[mask]
    y_naive_m = prev_close[mask]
    dates_m = test_dates[mask]

    lstm_m = metrics(y_true_m, y_pred_m)
    naive_m = metrics(y_true_m, y_naive_m)

    print("\n=== Test metrics (USD) ===")
    print("LSTM :", lstm_m)
    print("Naive:", naive_m)

    plot_curves(history, FIG_DIR / "training_curves.png")
    plot_predictions(dates_m, y_true_m, y_pred_m, y_naive_m, FIG_DIR / "test_predictions.png")

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_min": scaler.data_min_.tolist(),
            "scaler_max": scaler.data_max_.tolist(),
            "features": FEATURES,
            "window": WINDOW,
        },
        OUT_DIR / "lstm_aapl.pt",
    )

    report = {
        "ticker": TICKER,
        "window": WINDOW,
        "train_samples": int(train_sl.stop - train_sl.start),
        "val_samples": int(val_sl.stop - val_sl.start),
        "test_samples": int(len(y_true_m)),
        "lstm": lstm_m,
        "naive_yesterday": naive_m,
        "note": (
            "Pedagogical exercise. Beating or losing to the naive baseline on price level "
            "is both acceptable if the protocol is clean. Do not treat this as a trading edge."
        ),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved figures → {FIG_DIR}")
    print(f"Saved metrics → {OUT_DIR / 'metrics.json'}")
    print(f"Saved model  → {OUT_DIR / 'lstm_aapl.pt'}")


if __name__ == "__main__":
    main()
