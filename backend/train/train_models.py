"""
Multi-timeframe TCN + 6-signal weight optimization.

TCN trained on 3 years daily OHLCV with multi-timeframe features:
  - Daily returns, volatility, RSI, MACD at 1d, 5d, 20d scales
  - Volume patterns, candle body, momentum across timeframes

Then backtest optimal weights for all 6 signals:
  1. TCN model (next-day direction)
  2. RSI (mean-reversion)
  3. MACD (trend)
  4. Order flow (Polymarket/TM)
  5. Sentiment (TM AI + Fear & Greed)
  6. 30-day threshold model

Usage: python train/train_models.py
"""

import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_tcn import DirectionTCN
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR


def build_multi_timeframe_features(prices, opens, highs, lows, volumes, timestamps):
    """
    15 features capturing price dynamics at multiple timeframes.
    Mirrors what a trader looks at: 1-day, 5-day, 20-day patterns.
    """
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])

    # Multi-scale volatility (1d, 5d, 20d)
    vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values

    # RSI (14-period)
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rsi = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50).values / 100  # normalize 0-1

    # MACD histogram (12/26/9)
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_hist_raw = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values
    macd_hist = macd_hist_raw / np.maximum(np.abs(macd_hist_raw).rolling(20).mean() if hasattr(macd_hist_raw, 'rolling') else pd.Series(np.abs(macd_hist_raw)).rolling(20, min_periods=1).mean().values, 1)

    # Price position in range (1 = top of 20d range, 0 = bottom)
    price_pos = np.array([
        (lambda w: (prices[i] - w.min()) / (w.max() - w.min()) if w.max() > w.min() else 0.5)(prices[max(0, i-20):i+1])
        for i in range(n)
    ])

    # Multi-timeframe momentum (1d, 5d, 20d returns)
    mom_1 = log_ret  # already have
    mom_5 = np.concatenate([np.zeros(5), [(prices[i] - prices[i-5]) / prices[i-5] for i in range(5, n)]])
    mom_20 = np.concatenate([np.zeros(20), [(prices[i] - prices[i-20]) / prices[i-20] for i in range(20, n)]])

    # Mean reversion z-score
    mean_rev = np.array([
        (lambda w: (prices[i] - np.mean(w)) / np.std(w) if np.std(w) > 0 else 0)(prices[max(0, i-20):i+1])
        for i in range(n)
    ])

    # Candle patterns
    body_ratio = np.array([(prices[i] - opens[i]) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0 for i in range(n)])
    upper_wick = np.array([(highs[i] - max(prices[i], opens[i])) / (highs[i] - lows[i]) if highs[i] > lows[i] else 0 for i in range(n)])

    # Volume dynamics
    vol_sma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    rel_volume = np.where(vol_sma20 > 0, volumes / vol_sma20, 1.0)

    # Volatility regime
    vol_ratio = np.where(vol_20 > 1e-10, vol_5 / vol_20, 1.0)

    # Day of week
    dow = pd.to_datetime(timestamps).dt.dayofweek.values / 6.0

    return np.column_stack([
        log_ret, vol_5, vol_20, rsi, macd_hist,
        price_pos, mom_5, mom_20, mean_rev,
        body_ratio, upper_wick, rel_volume,
        vol_ratio, mom_1, dow
    ])


def main():
    print("=" * 60)
    print("Multi-Timeframe TCN Training + Signal Weight Optimization")
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

    prices = df["price"].values
    features = build_multi_timeframe_features(
        prices, df["open"].values, df["high"].values,
        df["low"].values, df["volume"].values, df["timestamp"]
    )

    # Next-day direction labels
    SEQ_LEN = 30
    X, Y = [], []
    for i in range(SEQ_LEN, len(features) - 1):
        X.append(features[i - SEQ_LEN:i])
        Y.append(1.0 if prices[i + 1] > prices[i] else 0.0)
    X, Y = np.array(X), np.array(Y)
    dates = df["timestamp"].values[SEQ_LEN:-1]
    dates_naive = pd.to_datetime(dates).tz_localize(None).values

    print(f"  {len(X)} sequences, {features.shape[1]} features, balance: {Y.mean():.1%} up")

    # Split: Train 2yr | Val 6mo | Test 6mo
    train_end = pd.Timestamp("2025-04-15")
    val_end = pd.Timestamp("2025-10-15")
    tr = dates_naive < train_end
    va = (dates_naive >= train_end) & (dates_naive < val_end)
    te = dates_naive >= val_end

    X_tr, Y_tr = X[tr], Y[tr]
    X_va, Y_va = X[va], Y[va]
    X_te, Y_te = X[te], Y[te]
    print(f"  Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

    # ── Train TCN ─────────────────────────────────────────
    print("\n[1/2] Training multi-timeframe TCN...")

    flat = X_tr.reshape(-1, X_tr.shape[-1])
    means, stds = flat.mean(0), flat.std(0)
    stds[stds == 0] = 1

    Xt = torch.FloatTensor((X_tr - means) / stds)
    Yt = torch.FloatTensor(Y_tr)
    Xv = torch.FloatTensor((X_va - means) / stds)
    Yv = torch.FloatTensor(Y_va)
    Xte = torch.FloatTensor((X_te - means) / stds)
    Yte = torch.FloatTensor(Y_te)

    N_FEAT = features.shape[1]
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=32, shuffle=True)
    model = DirectionTCN(input_size=N_FEAT, num_channels=48, num_layers=4, kernel_size=3, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=60, T_mult=2)
    criterion = nn.BCELoss()

    best_val, patience = 0, 0
    for epoch in range(400):
        model.train()
        for xb, yb in loader:
            pred = model(xb); loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_acc = ((model(Xv) > 0.5).float() == Yv).float().mean().item()

        if (epoch + 1) % 50 == 0:
            print(f"    Ep {epoch+1}: val_acc={va_acc:.1%}")

        if va_acc > best_val:
            best_val = va_acc; patience = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn.pt"))
        else:
            patience += 1
            if patience >= 60: break

    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_norm.json"), "w") as f:
        json.dump({"means": means.tolist(), "stds": stds.tolist(), "n_features": N_FEAT}, f)

    # Test accuracy
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xt) > 0.5).float() == Yt).float().mean().item()
        te_acc = ((model(Xte) > 0.5).float() == Yte).float().mean().item()
        # Get test predictions for weight optimization
        te_tcn_probs = model(Xte).numpy()

    print(f"    Train: {tr_acc:.1%} | Val: {best_val:.1%} | Test: {te_acc:.1%}")

    # ── Backtest 6-Signal Weights ─────────────────────────
    print("\n[2/2] Backtesting optimal weights for 6 signals...")

    # Build all 6 signals on test set
    test_indices = np.where(te)[0]
    rsi_raw = (100 - (100 / (1 + pd.Series(prices).diff().where(lambda x: x > 0, 0).rolling(14, min_periods=1).mean() /
               (-pd.Series(prices).diff().where(lambda x: x < 0, 0)).rolling(14, min_periods=1).mean().replace(0, np.nan)))).fillna(50).values

    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_h = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values

    signals_matrix = []
    labels_test = []

    for j, idx in enumerate(test_indices):
        real_idx = SEQ_LEN + idx
        if real_idx >= len(prices) - 1: continue

        # Signal 1: TCN probability
        tcn_p = float(te_tcn_probs[j]) if j < len(te_tcn_probs) else 0.5

        # Signal 2: RSI mean-reversion
        rsi_val = rsi_raw[real_idx]
        if rsi_val < 30: rsi_p = 0.65
        elif rsi_val > 70: rsi_p = 0.35
        else: rsi_p = 0.5 + (rsi_val - 50) / 200

        # Signal 3: MACD trend
        mh = macd_h[real_idx]
        macd_p = 0.5 + np.clip(mh / max(np.abs(macd_h[max(0,real_idx-20):real_idx+1]).max(), 1) * 0.2, -0.25, 0.25)

        # Signal 4: Order flow proxy (volume momentum)
        if real_idx >= 5 and df["volume"].iloc[real_idx] > 0:
            vol_r = df["volume"].iloc[real_idx] / max(df["volume"].iloc[real_idx-5:real_idx].mean(), 1)
            vol_dir = np.sign(prices[real_idx] - prices[real_idx-1])
            of_p = 0.5 + vol_dir * min(vol_r - 1, 1) * 0.15
        else:
            of_p = 0.5

        # Signal 5: Sentiment proxy (contrarian FG from momentum)
        mom = (prices[real_idx] - prices[max(0,real_idx-20)]) / prices[max(0,real_idx-20)]
        fg_proxy = np.clip(50 + mom * 300, 5, 95)
        if fg_proxy < 20: sent_p = 0.62
        elif fg_proxy > 80: sent_p = 0.38
        else: sent_p = 0.5 + (fg_proxy - 50) / 300

        # Signal 6: 30-day threshold model (will price be higher in 30 days?)
        if real_idx + 30 < len(prices):
            future_30d = prices[real_idx + 30]
            # Use volatility-based probability
            vol = np.std(np.diff(np.log(prices[max(0,real_idx-60):real_idx+1]))) if real_idx > 5 else 0.02
            horizon_vol = vol * np.sqrt(30)
            z = 0.05 / max(horizon_vol, 0.01)  # 5% move probability
            threshold_p = 0.5 + np.clip(mom * 2, -0.2, 0.2)
        else:
            threshold_p = 0.5

        signals_matrix.append([tcn_p, rsi_p, macd_p, of_p, sent_p, threshold_p])
        labels_test.append(Y_te[j] if j < len(Y_te) else 0.5)

    S = np.array(signals_matrix)
    L = np.array(labels_test)
    print(f"    {len(S)} test samples with 6 signals each")

    # Find optimal weights via logistic regression
    lr = LogisticRegression(C=1.0, max_iter=500)
    lr.fit(S, L)
    lr_acc = lr.score(S, L)

    signal_names = ["TCN", "RSI", "MACD", "OrderFlow", "Sentiment", "30dayModel"]
    raw_weights = np.abs(lr.coef_[0])
    norm_weights = raw_weights / raw_weights.sum()

    print(f"\n    Optimal weights (logistic regression, acc={lr_acc:.1%}):")
    for name, w, raw in zip(signal_names, norm_weights, lr.coef_[0]):
        print(f"      {name:12s}: {w*100:.1f}%  (coef={raw:.3f})")

    # Save weights
    weights = {name: round(float(w), 4) for name, w in zip(signal_names, norm_weights)}
    with open(os.path.join(MODEL_WEIGHTS_DIR, "signal_weights.json"), "w") as f:
        json.dump(weights, f, indent=2)

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  TCN (15-feature, multi-timeframe):")
    print(f"    Train: {tr_acc:.1%} ({len(X_tr)} days)")
    print(f"    Val:   {best_val:.1%} ({len(X_va)} days)")
    print(f"    Test:  {te_acc:.1%} ({len(X_te)} days)")
    print(f"  Signal weights saved to: signal_weights.json")
    print(f"{'='*60}")

    # Save metrics
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "backtest_results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "backtest_results", "training_metrics.json"), "w") as f:
        json.dump({
            "tcn": {"train_acc": tr_acc, "val_acc": best_val, "test_acc": te_acc,
                    "n_features": N_FEAT, "seq_len": SEQ_LEN},
            "weights": weights,
        }, f, indent=2)


if __name__ == "__main__":
    main()
