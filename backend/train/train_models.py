"""
TCN training and evaluation on 3 years of daily BTC + 7 days hourly.

Split:
  Daily:  Train (Apr 2023 – Oct 2025) | Test (Oct 2025 – Apr 2026)
  Hourly: Train (first 80%) | Test (last 20%)

Also retrains on latest hourly data (called by app/main.py on every launch).

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
from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH
from app.data.truemarkets_mcp import CACHE_DIR
import asyncio


# ─── Feature engineering ──────────────────────────────────

def build_features(prices, timestamps=None, has_ohlcv=False, opens=None, highs=None, lows=None, volumes=None):
    """
    10 features for TCN. Works for both daily (OHLCV) and hourly (price-only) data.
    """
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

    if has_ohlcv and opens is not None:
        # Daily: candle body ratio + volume change
        body_ratio = np.array([(prices[i] - opens[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0 for i in range(n)])
        vol_change = np.concatenate([[0], np.diff(np.log(np.maximum(volumes, 1)))])
    else:
        # Hourly: acceleration + time encoding
        body_ratio = np.concatenate([np.zeros(2), [log_ret[i] - log_ret[i-1] for i in range(2, n)]])
        if timestamps is not None:
            hours = pd.to_datetime(timestamps).hour
            vol_change = np.sin(2 * np.pi * hours / 24)
        else:
            vol_change = np.zeros(n)

    vol_ratio = np.where(vol_20 > 1e-10, vol_5 / vol_20, 1.0)

    if has_ohlcv:
        dow = pd.to_datetime(timestamps).dt.dayofweek.values / 6.0 if timestamps is not None else np.zeros(n)
    else:
        if timestamps is not None:
            dow = np.cos(2 * np.pi * pd.to_datetime(timestamps).hour / 24)
        else:
            dow = np.zeros(n)

    return np.column_stack([log_ret, vol_5, vol_20, price_pos, mom_5, mean_rev, body_ratio, vol_change, vol_ratio, dow])


def create_consensus_sequences(features, prices, seq_len, horizons):
    """Multi-horizon consensus: only keep clear trend signals."""
    X, Y, indices = [], [], []
    max_h = max(horizons)
    for i in range(seq_len, len(features) - max_h):
        dirs = [prices[min(i + h, len(prices) - 1)] > prices[i] for h in horizons]
        up = sum(dirs)
        if all(dirs):       label = 0.95
        elif not any(dirs): label = 0.05
        elif up >= len(dirs) - 1: label = 0.85
        elif up <= 1:       label = 0.15
        else: continue
        X.append(features[i - seq_len:i])
        Y.append(label)
        indices.append(i)
    return np.array(X), np.array(Y), np.array(indices)


# ─── Training ────────────────────────────────────────────

def train_tcn(X_train, Y_train, X_val, Y_val, save_name, epochs=200, channels=32, layers=3, dropout=0.2):
    """Train TCN, save best model by validation accuracy. Returns val accuracy."""
    # Light augmentation
    X_aug = np.vstack([X_train, X_train + np.random.randn(*X_train.shape) * 0.002])
    Y_aug = np.hstack([Y_train, Y_train])

    flat = X_aug.reshape(-1, X_aug.shape[-1])
    means, stds = flat.mean(0), flat.std(0)
    stds[stds == 0] = 1

    X_t = torch.FloatTensor((X_aug - means) / stds)
    Y_t = torch.FloatTensor(Y_aug)
    X_v = torch.FloatTensor((X_val - means) / stds)
    Y_v_hard = (torch.FloatTensor(Y_val) > 0.5).float()

    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=16, shuffle=True)
    model = DirectionTCN(input_size=10, num_channels=channels, num_layers=layers, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.BCELoss()

    best_acc, patience = 0, 0
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
            va = ((model(X_v) > 0.5).float() == Y_v_hard).float().mean().item()

        if (epoch + 1) % 50 == 0:
            print(f"    Ep {epoch+1}: val_acc={va:.1%}")

        if va > best_acc:
            best_acc = va
            patience = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}.pt"))
        else:
            patience += 1
            if patience >= 40:
                print(f"    Early stop at ep {epoch+1}")
                break

    with open(os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}_norm.json"), "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

    # Final test accuracy
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(torch.FloatTensor((X_train - means) / stds)) > 0.5).float() == (torch.FloatTensor(Y_train) > 0.5).float()).float().mean().item()

    return {"train_acc": tr_acc, "val_acc": best_acc, "train_n": len(X_train), "val_n": len(X_val)}


# ─── Main ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TCN Training Pipeline")
    print("=" * 60)

    # ── Load 3-year daily data ────────────────────────────
    print("\n[1] Loading 3-year daily data...")
    with open(os.path.join(CACHE_DIR, "btc_3Y_1d.json")) as f:
        pts = json.load(f)["results"][0]["points"]

    df_daily = pd.DataFrame([{
        "timestamp": p["t"], "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])
    df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
    print(f"    {len(df_daily)} daily candles: {df_daily['timestamp'].iloc[0].date()} to {df_daily['timestamp'].iloc[-1].date()}")

    # ── Build daily features + consensus sequences ────────
    feat_d = build_features(
        df_daily["price"].values, df_daily["timestamp"],
        has_ohlcv=True, opens=df_daily["open"].values,
        highs=df_daily["high"].values, lows=df_daily["low"].values,
        volumes=df_daily["volume"].values,
    )
    X_d, Y_d, idx_d = create_consensus_sequences(feat_d, df_daily["price"].values, seq_len=30, horizons=[1, 2, 3])
    print(f"    {len(X_d)} consensus sequences")

    # ── Split: Train (Apr 2023 – Oct 2025) | Test (Oct 2025 – Apr 2026) ──
    dates = pd.to_datetime(df_daily["timestamp"].values[idx_d]).tz_localize(None).values
    test_start = pd.Timestamp("2025-10-15")
    train_mask = dates < test_start
    test_mask = dates >= test_start

    X_train_d, Y_train_d = X_d[train_mask], Y_d[train_mask]
    X_test_d, Y_test_d = X_d[test_mask], Y_d[test_mask]
    print(f"    Train: {len(X_train_d)} | Test: {len(X_test_d)}")

    # ── Train daily TCN ───────────────────────────────────
    print("\n[2] Training daily TCN (trend model)...")
    val_n = max(int(len(X_train_d) * 0.1), 20)
    daily_metrics = train_tcn(
        X_train_d[:-val_n], Y_train_d[:-val_n],
        X_train_d[-val_n:], Y_train_d[-val_n:],
        "direction_tcn_daily", epochs=200, channels=16, layers=3, dropout=0.35,
    )

    # Test accuracy
    model = DirectionTCN(input_size=10, num_channels=16, num_layers=3, dropout=0.35)
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily.pt"), weights_only=True))
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily_norm.json")) as f:
        norm = json.load(f)
    means, stds = np.array(norm["means"]), np.array(norm["stds"])
    stds[stds == 0] = 1
    model.eval()
    with torch.no_grad():
        test_acc_d = ((model(torch.FloatTensor((X_test_d - means) / stds)) > 0.5).float() == (torch.FloatTensor(Y_test_d) > 0.5).float()).float().mean().item()
    daily_metrics["test_acc"] = test_acc_d
    daily_metrics["test_n"] = len(X_test_d)

    # ── Load hourly data ──────────────────────────────────
    print("\n[3] Loading 7-day hourly data...")
    from app.data.truemarkets_mcp import fetch_historical_prices
    df_hourly = asyncio.run(fetch_historical_prices("BTC", days=7))
    print(f"    {len(df_hourly)} hourly candles")

    feat_h = build_features(df_hourly["price"].values, df_hourly.get("timestamp", pd.Series()).values)
    X_h, Y_h, _ = create_consensus_sequences(feat_h, df_hourly["price"].values, seq_len=SEQUENCE_LENGTH, horizons=[1, 2, 3, 4, 6])
    print(f"    {len(X_h)} consensus sequences")

    # ── Train hourly TCN (80/20 split) ────────────────────
    print("\n[4] Training hourly TCN (production model)...")
    split_h = int(len(X_h) * 0.8)
    hourly_metrics = train_tcn(
        X_h[:split_h], Y_h[:split_h],
        X_h[split_h:], Y_h[split_h:],
        "direction_tcn", epochs=250, channels=32, layers=3, dropout=0.2,
    )

    # ── Results ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Daily TCN (3-year trend model):")
    print(f"    Train: {daily_metrics['train_acc']:.1%} ({daily_metrics['train_n']} samples)")
    print(f"    Val:   {daily_metrics['val_acc']:.1%}")
    print(f"    Test:  {daily_metrics['test_acc']:.1%} ({daily_metrics['test_n']} samples, Oct 2025 – Apr 2026)")
    print(f"  Hourly TCN (production model):")
    print(f"    Train: {hourly_metrics['train_acc']:.1%} ({hourly_metrics['train_n']} samples)")
    print(f"    Val:   {hourly_metrics['val_acc']:.1%} ({hourly_metrics['val_n']} samples)")
    print(f"{'=' * 60}")

    # Save metrics
    metrics = {"daily": daily_metrics, "hourly": hourly_metrics}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "backtest_results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "backtest_results", "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
