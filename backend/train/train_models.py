"""
TCN training for NEXT-DAY BTC direction prediction.
Trained on 3 years of daily OHLCV data from CryptoCompare.

Split:
  Train: Apr 2023 – Apr 2025 (2 years, ~730 days)
  Val:   Apr 2025 – Oct 2025 (6 months, ~180 days)
  Test:  Oct 2025 – Apr 2026 (6 months, ~180 days)

Prediction: "BTC will likely go UP/DOWN tomorrow"

Usage: python train/train_models.py
"""

import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_tcn import DirectionTCN
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR

SEQ_LEN = 30  # 30 days lookback


def build_features(prices, opens, highs, lows, volumes, timestamps):
    """10 daily features from OHLCV data."""
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values

    price_pos = np.array([
        (lambda w: (prices[i] - w.min()) / (w.max() - w.min()) if w.max() > w.min() else 0.5)(prices[max(0, i-20):i+1])
        for i in range(n)
    ])
    mom_5 = np.concatenate([np.zeros(5), [(prices[i] - prices[i-5]) / prices[i-5] for i in range(5, n)]])
    mean_rev = np.array([
        (lambda w: (prices[i] - np.mean(w)) / np.std(w) if np.std(w) > 0 else 0)(prices[max(0, i-20):i+1])
        for i in range(n)
    ])
    body_ratio = np.array([(prices[i] - opens[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0 for i in range(n)])
    vol_change = np.concatenate([[0], np.diff(np.log(np.maximum(volumes, 1)))])
    vol_ratio = np.where(vol_20 > 1e-10, vol_5 / vol_20, 1.0)
    dow = pd.to_datetime(timestamps).dt.dayofweek.values / 6.0

    return np.column_stack([log_ret, vol_5, vol_20, price_pos, mom_5, mean_rev, body_ratio, vol_change, vol_ratio, dow])


def create_nextday_sequences(features, prices, seq_len=SEQ_LEN):
    """Next-day direction labels. No consensus filter — predict every day."""
    X, Y = [], []
    for i in range(seq_len, len(features) - 1):
        X.append(features[i - seq_len:i])
        Y.append(1.0 if prices[i + 1] > prices[i] else 0.0)
    return np.array(X), np.array(Y)


def train_tcn(X_train, Y_train, X_val, Y_val, save_name, epochs=300):
    """Train TCN for next-day prediction."""
    # Normalize on training data
    flat = X_train.reshape(-1, X_train.shape[-1])
    means, stds = flat.mean(0), flat.std(0)
    stds[stds == 0] = 1

    X_t = torch.FloatTensor((X_train - means) / stds)
    Y_t = torch.FloatTensor(Y_train)
    X_v = torch.FloatTensor((X_val - means) / stds)
    Y_v = torch.FloatTensor(Y_val)

    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=32, shuffle=True)
    model = DirectionTCN(input_size=10, num_channels=48, num_layers=4, kernel_size=3, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2)
    criterion = nn.BCELoss()

    best_val_acc, patience = 0, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_v)
            va = ((vp > 0.5).float() == Y_v).float().mean().item()

        if (epoch + 1) % 50 == 0:
            print(f"    Ep {epoch+1}: val_acc={va:.1%}")

        if va > best_val_acc:
            best_val_acc = va
            patience = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}.pt"))
        else:
            patience += 1
            if patience >= 60:
                print(f"    Early stop at ep {epoch+1}")
                break

    with open(os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}_norm.json"), "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

    # Final accuracies
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(X_t) > 0.5).float() == Y_t).float().mean().item()

    return {"train_acc": tr_acc, "val_acc": best_val_acc, "train_n": len(X_train), "val_n": len(X_val), "means": means, "stds": stds}


def main():
    print("=" * 60)
    print("TCN Training — Next-Day BTC Direction (3 Years Daily)")
    print("=" * 60)

    # Load 3-year data
    with open(os.path.join(CACHE_DIR, "btc_3Y_1d.json")) as f:
        pts = json.load(f)["results"][0]["points"]

    df = pd.DataFrame([{
        "timestamp": p["t"], "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  {len(df)} daily candles: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")

    features = build_features(df["price"].values, df["open"].values, df["high"].values,
                               df["low"].values, df["volume"].values, df["timestamp"])
    prices = df["price"].values
    X, Y = create_nextday_sequences(features, prices, seq_len=SEQ_LEN)
    dates = df["timestamp"].values[SEQ_LEN:-1]
    print(f"  {len(X)} sequences, class balance: {Y.mean():.1%} up")

    # Split by date
    dates_naive = pd.to_datetime(dates).tz_localize(None).values
    train_end = pd.Timestamp("2025-04-15")
    val_end = pd.Timestamp("2025-10-15")

    train_mask = dates_naive < train_end
    val_mask = (dates_naive >= train_end) & (dates_naive < val_end)
    test_mask = dates_naive >= val_end

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    print(f"  Train: {len(X_train)} (Apr 2023 – Apr 2025)")
    print(f"  Val:   {len(X_val)} (Apr 2025 – Oct 2025)")
    print(f"  Test:  {len(X_test)} (Oct 2025 – Apr 2026)")

    # Train
    print("\n  Training TCN for next-day prediction...")
    metrics = train_tcn(X_train, Y_train, X_val, Y_val, "direction_tcn", epochs=300)

    # Test
    model = DirectionTCN(input_size=10, num_channels=48, num_layers=4, kernel_size=3, dropout=0.2)
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        X_test_norm = torch.FloatTensor((X_test - metrics["means"]) / metrics["stds"])
        test_preds = model(X_test_norm)
        test_acc = ((test_preds > 0.5).float() == torch.FloatTensor(Y_test)).float().mean().item()

    print(f"\n{'='*60}")
    print(f"  RESULTS (Next-Day BTC Direction)")
    print(f"{'='*60}")
    print(f"  Train: {metrics['train_acc']:.1%} ({metrics['train_n']} days)")
    print(f"  Val:   {metrics['val_acc']:.1%} ({metrics['val_n']} days)")
    print(f"  Test:  {test_acc:.1%} ({len(X_test)} days, Oct 2025 – Apr 2026)")
    print(f"{'='*60}")

    # Save metrics
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "backtest_results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "backtest_results", "training_metrics.json"), "w") as f:
        json.dump({"train_acc": metrics["train_acc"], "val_acc": metrics["val_acc"],
                    "test_acc": test_acc, "train_n": metrics["train_n"],
                    "val_n": metrics["val_n"], "test_n": len(X_test)}, f, indent=2)


if __name__ == "__main__":
    main()
