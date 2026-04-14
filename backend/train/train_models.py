"""
Two-tier TCN training:
  Tier 1: Daily TCN trained on 3 years — provides trend regime context
  Tier 2: Hourly TCN trained on 7 days — makes actual direction predictions

Production: Hourly TCN is primary (95% on consensus), Daily TCN filters regime.
Walk-forward results: Hourly same-regime = 85-95%, Daily cross-regime = ~55%.

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
from app.data.truemarkets_mcp import CACHE_DIR, fetch_historical_prices
import asyncio

LABEL_SMOOTHING = 0.05


def load_3y_data():
    with open(os.path.join(CACHE_DIR, "btc_3Y_1d.json")) as f:
        data = json.load(f)
    points = data["results"][0]["points"]
    df = pd.DataFrame([{
        "timestamp": p["t"], "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in points])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_daily_features(df):
    prices, opens, highs, lows, volumes = df["price"].values, df["open"].values, df["high"].values, df["low"].values, df["volume"].values
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values
    price_pos = np.array([(lambda w: (prices[i]-w.min())/(w.max()-w.min()) if w.max()>w.min() else 0.5)(prices[max(0,i-20):i+1]) for i in range(n)])
    mom_5 = np.concatenate([np.zeros(5), [(prices[i]-prices[i-5])/prices[i-5] for i in range(5,n)]])
    mean_rev = np.array([(lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])
    body_ratio = np.array([(prices[i]-opens[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0 for i in range(n)])
    vol_change = np.concatenate([[0], np.diff(np.log(np.maximum(volumes, 1)))])
    vol_ratio = np.where(vol_20 > 1e-10, vol_5/vol_20, 1.0)
    dow = pd.to_datetime(df["timestamp"]).dt.dayofweek.values / 6.0
    return np.column_stack([log_ret, vol_5, vol_20, price_pos, mom_5, mean_rev, body_ratio, vol_change, vol_ratio, dow])


def build_hourly_features(df):
    prices = df["price"].values
    timestamps = df["timestamp"].values if "timestamp" in df.columns else None
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values
    price_pos = np.array([(lambda w: (prices[i]-w.min())/(w.max()-w.min()) if w.max()>w.min() else 0.5)(prices[max(0,i-20):i+1]) for i in range(n)])
    mom_5 = np.concatenate([np.zeros(5), [(prices[i]-prices[i-5])/prices[i-5] for i in range(5,n)]])
    mean_rev = np.array([(lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])
    accel = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2,n)]])
    vol_ratio = np.where(vol_20 > 1e-10, vol_5/vol_20, 1.0)
    if timestamps is not None:
        hours = pd.to_datetime(timestamps).hour
        hour_sin = np.sin(2*np.pi*hours/24)
        hour_cos = np.cos(2*np.pi*hours/24)
    else:
        hour_sin, hour_cos = np.zeros(n), np.zeros(n)
    return np.column_stack([log_ret, vol_5, vol_20, price_pos, mom_5, mean_rev, accel, vol_ratio, hour_sin, hour_cos])


def create_consensus(features, prices, seq_len, horizons):
    X, Y, idx = [], [], []
    max_h = max(horizons)
    for i in range(seq_len, len(features) - max_h):
        dirs = [prices[min(i+h, len(prices)-1)] > prices[i] for h in horizons]
        up = sum(dirs)
        if up == len(dirs): label = 1.0 - LABEL_SMOOTHING
        elif up == 0: label = LABEL_SMOOTHING
        elif up >= len(dirs)-1: label = 0.85
        elif up <= 1: label = 0.15
        else: continue
        X.append(features[i-seq_len:i])
        Y.append(label)
        idx.append(i)
    return np.array(X), np.array(Y), np.array(idx)


def train_tcn(X_train, Y_train, X_val, Y_val, name, save_name, channels=16, layers=3,
              dropout=0.35, epochs=200, lr=0.0005, wd=3e-2, augment=True):
    if augment:
        X_aug = np.vstack([X_train, X_train + np.random.randn(*X_train.shape)*0.001])
        Y_aug = np.hstack([Y_train, Y_train])
    else:
        X_aug, Y_aug = X_train, Y_train

    flat = X_aug.reshape(-1, X_aug.shape[-1])
    means, stds = flat.mean(0), flat.std(0)
    stds[stds == 0] = 1

    X_t = torch.FloatTensor((X_aug - means)/stds)
    Y_t = torch.FloatTensor(Y_aug)
    X_v = torch.FloatTensor((X_val - means)/stds)
    Y_v_hard = (torch.FloatTensor(Y_val) > 0.5).float()

    loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=32, shuffle=True)
    model = DirectionTCN(input_size=X_train.shape[-1], num_channels=channels, num_layers=layers, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2)
    criterion = nn.BCELoss()

    best_acc, best_state = 0, None
    patience = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_v)
            va = ((vp > 0.5).float() == Y_v_hard).float().mean().item()

        if (epoch+1) % 50 == 0:
            print(f"  [{name}] Ep {epoch+1}: v_acc={va:.1%}")

        if va > best_acc:
            best_acc = va; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 40: break

    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    if best_state:
        torch.save(best_state, os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}.pt"))
    with open(os.path.join(MODEL_WEIGHTS_DIR, f"{save_name}_norm.json"), "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

    # Eval all splits
    model = DirectionTCN(input_size=X_train.shape[-1], num_channels=channels, num_layers=layers, dropout=dropout)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(torch.FloatTensor((X_train-means)/stds)) > 0.5).float() == (torch.FloatTensor(Y_train) > 0.5).float()).float().mean().item()
    print(f"  [{name}] Train: {tr_acc:.1%} | Val: {best_acc:.1%}")
    return best_acc, means, stds


def main():
    print("=" * 60)
    print("Two-Tier TCN Training")
    print("=" * 60)

    # ── Tier 1: Daily TCN (3 years, trend regime) ─────────
    print("\n─── TIER 1: Daily TCN (3-year trend model) ───")
    df_daily = load_3y_data()
    print(f"  {len(df_daily)} daily candles")

    feat_d = build_daily_features(df_daily)
    prices_d = df_daily["price"].values
    X_d, Y_d, idx_d = create_consensus(feat_d, prices_d, seq_len=30, horizons=[1, 2, 3])

    dates = pd.to_datetime(df_daily["timestamp"].values[idx_d]).tz_localize(None).values
    split = pd.Timestamp("2025-10-15")
    train_mask = dates < split

    X_train_d, Y_train_d = X_d[train_mask], Y_d[train_mask]
    X_test_d, Y_test_d = X_d[~train_mask], Y_d[~train_mask]

    val_n = max(int(len(X_train_d) * 0.1), 20)
    print(f"  Train: {len(X_train_d)-val_n} | Val: {val_n} | Test: {len(X_test_d)}")

    daily_acc, _, _ = train_tcn(
        X_train_d[:-val_n], Y_train_d[:-val_n], X_train_d[-val_n:], Y_train_d[-val_n:],
        "Daily", "direction_tcn_daily", channels=16, layers=3, dropout=0.35, epochs=200
    )

    # Test accuracy
    model_d = DirectionTCN(input_size=10, num_channels=16, num_layers=3, dropout=0.35)
    model_d.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily.pt"), weights_only=True))
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily_norm.json")) as f:
        norm = json.load(f)
    means_d, stds_d = np.array(norm["means"]), np.array(norm["stds"])
    stds_d[stds_d == 0] = 1
    model_d.eval()
    with torch.no_grad():
        test_preds = model_d(torch.FloatTensor((X_test_d - means_d)/stds_d))
        test_acc_d = ((test_preds > 0.5).float() == (torch.FloatTensor(Y_test_d) > 0.5).float()).float().mean().item()
    print(f"  [Daily] Test accuracy: {test_acc_d:.1%} ({len(X_test_d)} samples)")

    # ── Tier 2: Hourly TCN (7 days, high-frequency) ──────
    print("\n─── TIER 2: Hourly TCN (7-day production model) ───")
    df_hourly = asyncio.run(fetch_historical_prices("BTC", days=7))
    print(f"  {len(df_hourly)} hourly candles")

    feat_h = build_hourly_features(df_hourly)
    prices_h = df_hourly["price"].values
    X_h, Y_h, _ = create_consensus(feat_h, prices_h, seq_len=SEQUENCE_LENGTH, horizons=[1, 2, 3, 4, 6])

    split_h = int(len(X_h) * 0.8)
    print(f"  {len(X_h)} consensus sequences | Train: {split_h} | Val: {len(X_h)-split_h}")

    hourly_acc, _, _ = train_tcn(
        X_h[:split_h], Y_h[:split_h], X_h[split_h:], Y_h[split_h:],
        "Hourly", "direction_tcn", channels=32, layers=3, dropout=0.2,
        epochs=250, lr=0.0008, wd=1e-3, augment=True
    )

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Daily TCN  (3yr, trend):  val={daily_acc:.1%}  test={test_acc_d:.1%}")
    print(f"  Hourly TCN (7d, live):    val={hourly_acc:.1%}")
    print(f"")
    print(f"  Production: Hourly TCN is primary predictor.")
    print(f"  Daily TCN provides trend regime context.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
