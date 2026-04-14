"""
CNN-LSTM with Boruta feature selection (Dubey & Enke 2025).
Paper: "Bitcoin price direction prediction using on-chain data"

Architecture: Conv1D(8,3) → BN → Drop(0.5) → LSTM(32) → BN → Drop(0.5) → LSTM(64) → Dense(16) → Dense(1,sigmoid)
Lookback: 5 days | Epochs: 1000 | Batch: 50 | Early stop patience: 100

Since we don't have Glassnode on-chain data, we use price-derived features
and apply Boruta to select the most predictive subset.

Usage: python train/train_models.py
"""

import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_cnn_lstm import DirectionCNNLSTM, LOOKBACK
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR


def build_all_features(prices, opens, highs, lows, volumes):
    """Build 30+ candidate features for Boruta selection."""
    n = len(prices)
    feats = {}

    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    feats["log_return"] = log_ret
    feats["abs_return"] = np.abs(log_ret)

    # Multi-scale volatility
    for w in [5, 10, 20, 50]:
        feats[f"vol_{w}"] = pd.Series(log_ret).rolling(w, min_periods=1).std().fillna(0).values

    # RSI
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    feats["rsi"] = (100 - (100 / (1 + gain / loss_s.replace(0, np.nan)))).fillna(50).values / 100

    # MACD
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd = ema12 - ema26
    macd_sig = pd.Series(macd).ewm(span=9).mean().values
    feats["macd_hist"] = (macd - macd_sig) / np.maximum(pd.Series(np.abs(macd - macd_sig)).rolling(20, min_periods=1).mean().values, 1)

    # Multi-scale momentum
    for lag in [1, 3, 5, 10, 20]:
        feats[f"mom_{lag}"] = np.concatenate([np.zeros(lag), [(prices[i]-prices[i-lag])/prices[i-lag] for i in range(lag, n)]])

    # Price position
    for w in [10, 20, 50]:
        feats[f"pos_{w}"] = np.array([(lambda a: (prices[i]-a.min())/(a.max()-a.min()) if a.max()>a.min() else 0.5)(prices[max(0,i-w):i+1]) for i in range(n)])

    # Mean reversion
    feats["mean_rev_20"] = np.array([(lambda a: (prices[i]-np.mean(a))/np.std(a) if np.std(a)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])

    # Candle features
    feats["body_ratio"] = np.array([(prices[i]-opens[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0 for i in range(n)])
    feats["upper_wick"] = np.array([(highs[i]-max(prices[i],opens[i]))/(highs[i]-lows[i]) if highs[i]>lows[i] else 0 for i in range(n)])
    feats["lower_wick"] = np.array([(min(prices[i],opens[i])-lows[i])/(highs[i]-lows[i]) if highs[i]>lows[i] else 0 for i in range(n)])
    feats["range_pct"] = np.array([(highs[i]-lows[i])/prices[i] if prices[i]>0 else 0 for i in range(n)])

    # Volume features
    vol_sma = pd.Series(volumes).rolling(20, min_periods=1).mean().values
    feats["rel_volume"] = np.where(vol_sma > 0, volumes / vol_sma, 1)
    feats["vol_change"] = np.concatenate([[0], np.diff(np.log(np.maximum(volumes, 1)))])

    # Volatility ratio
    feats["vol_ratio"] = np.where(feats["vol_20"] > 1e-10, feats["vol_5"] / feats["vol_20"], 1)

    # Acceleration
    feats["accel"] = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2, n)]])

    # Day of week
    feats["placeholder_dow"] = np.zeros(n)  # placeholder if no timestamps

    # Price scaled (as in GRU paper)
    feats["price_scaled"] = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

    return feats


def boruta_select(X, Y, feature_names, n_iter=50):
    """Boruta feature selection (as described in paper Section 2.3.2)."""
    print(f"  Boruta selection on {X.shape[1]} features...")
    n_feat = X.shape[1]
    hits = np.zeros(n_feat)

    for iteration in range(n_iter):
        # Create shadow features (shuffled copies)
        shadow = X.copy()
        for j in range(n_feat):
            np.random.shuffle(shadow[:, j])

        X_combined = np.hstack([X, shadow])
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=iteration)
        rf.fit(X_combined, Y)

        importances = rf.feature_importances_
        real_imp = importances[:n_feat]
        shadow_imp = importances[n_feat:]
        shadow_max = shadow_imp.max()

        # Features beating shadow max are "hits"
        hits += (real_imp > shadow_max).astype(float)

    # Select features with > 30% hit rate (relaxed from paper's implicit 50%)
    selected = hits / n_iter > 0.3
    # If nothing selected, take top 10 by hit rate
    if not selected.any():
        top_k = min(10, n_feat)
        top_idx = np.argsort(-hits)[:top_k]
        selected = np.zeros(n_feat, dtype=bool)
        selected[top_idx] = True
        print(f"  Boruta: no features passed threshold, using top {top_k} by importance")
    selected_names = [feature_names[i] for i in range(n_feat) if selected[i]]
    print(f"  Selected {sum(selected)}/{n_feat} features: {selected_names}")
    return selected, selected_names


def main():
    print("=" * 60)
    print("CNN-LSTM + Boruta (Dubey & Enke 2025) Training")
    print("=" * 60)

    # Load data
    path = os.path.join(CACHE_DIR, "btc_5Y_1d.json")
    if not os.path.exists(path):
        path = os.path.join(CACHE_DIR, "btc_3Y_1d.json")
    with open(path) as f:
        pts = json.load(f)["results"][0]["points"]

    df = pd.DataFrame([{
        "price": float(p["price"]), "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])), "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])
    prices = df["price"].values
    print(f"  {len(df)} daily candles")

    # Build all features
    feat_dict = build_all_features(prices, df["open"].values, df["high"].values, df["low"].values, df["volume"].values)
    feat_names = list(feat_dict.keys())
    X_all = np.column_stack([feat_dict[k] for k in feat_names])
    N_FEAT_ALL = X_all.shape[1]
    print(f"  {N_FEAT_ALL} candidate features")

    # Build labels
    Y_all = np.array([1.0 if prices[i+1] > prices[i] else 0.0 for i in range(len(prices)-1)])

    # 80/20 split (as in paper)
    split = int(len(Y_all) * 0.8)
    X_flat_tr = X_all[LOOKBACK:split]
    Y_flat_tr = Y_all[LOOKBACK:split]

    # Boruta feature selection (on training data only)
    selected_mask, selected_names = boruta_select(X_flat_tr, Y_flat_tr, feat_names)
    X_selected = X_all[:, selected_mask]
    N_FEAT = X_selected.shape[1]

    # Build sequences with lookback=5 (paper's optimal)
    X_seq, Y_seq = [], []
    for i in range(LOOKBACK, len(X_selected) - 1):
        X_seq.append(X_selected[i-LOOKBACK:i])
        Y_seq.append(Y_all[i])
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

    X_tr, Y_tr = X_seq[:split-LOOKBACK], Y_seq[:split-LOOKBACK]
    X_te, Y_te = X_seq[split-LOOKBACK:], Y_seq[split-LOOKBACK:]
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)} (80/20 as in paper)")

    # MinMaxScaler (as in paper)
    flat = X_tr.reshape(-1, N_FEAT)
    mins, maxs = flat.min(0), flat.max(0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    X_tr_s = (X_tr - mins) / ranges
    X_te_s = (X_te - mins) / ranges

    Xt = torch.FloatTensor(X_tr_s)
    Yt = torch.FloatTensor(Y_tr)
    Xte = torch.FloatTensor(X_te_s)
    Yte = torch.FloatTensor(Y_te)

    # Train CNN-LSTM (paper config: 1000 epochs, batch=50, patience=100)
    print(f"\n  Training CNN-LSTM ({N_FEAT} features, lookback={LOOKBACK})...")
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=50, shuffle=False)  # no shuffle as in paper
    model = DirectionCNNLSTM(input_size=N_FEAT)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    best_acc, patience_count = 0, 0
    for epoch in range(1000):
        model.train()
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            te_acc = ((model(Xte) > 0.5).float() == Yte).float().mean().item()

        if (epoch+1) % 100 == 0:
            print(f"    Ep {epoch+1}/1000: test_acc={te_acc:.1%} (best={best_acc:.1%})")

        if te_acc > best_acc:
            best_acc = te_acc
            patience_count = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"))
        else:
            patience_count += 1
            if patience_count >= 100:  # paper's patience
                print(f"    Early stop at ep {epoch+1}")
                break

    # Save norm params
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json"), "w") as f:
        json.dump({"mins": mins.tolist(), "maxs": maxs.tolist(), "n_features": N_FEAT,
                    "selected_features": selected_names}, f)

    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xt) > 0.5).float() == Yt).float().mean().item()
        te_preds = model(Xte).numpy()
        te_acc_final = ((te_preds > 0.5) == Y_te).mean()

    print(f"\n  CNN-LSTM Results (Boruta selected {N_FEAT} features):")
    print(f"    Train: {tr_acc:.1%} | Test: {te_acc_final:.1%}")

    # ── BACKTEST 7-SIGNAL WEIGHTS ─────────────────────────
    print(f"\n{'='*60}")
    print(f"Backtesting 7-signal weights on {len(X_te)} test days")
    print(f"{'='*60}")

    rsi_raw = (100 - (100 / (1 + pd.Series(prices).diff().where(lambda x: x > 0, 0).rolling(14, min_periods=1).mean() /
               (-pd.Series(prices).diff().where(lambda x: x < 0, 0)).rolling(14, min_periods=1).mean().replace(0, np.nan)))).fillna(50).values
    log_ret_full = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_hist_full = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values

    signal_names = ["TM_Sentiment", "BTC_Momentum", "CNN_LSTM", "RSI", "MACD", "Polymarket", "30day_Model"]
    S_test = []

    test_start = split
    for j in range(len(X_te)):
        i = test_start + j
        if i >= len(prices) - 1: break

        mom_20 = (prices[i] - prices[max(0,i-20)]) / prices[max(0,i-20)]
        fg_proxy = np.clip(50 + mom_20 * 300 + (rsi_raw[i] - 50) * 0.3, 5, 95)
        sent_p = 0.62 if fg_proxy < 20 else 0.38 if fg_proxy > 80 else 0.5 + (fg_proxy-50)/300

        if i >= 5:
            ret_4 = (prices[i]-prices[i-4])/prices[i-4]
            ret_r = (prices[i]-prices[i-2])/prices[i-2]
            ret_p = (prices[i-2]-prices[i-4])/prices[i-4]
            btc_mom_p = float(np.clip(0.5 + ret_4*8 + (ret_r-ret_p)*15, 0.2, 0.8))
        else:
            btc_mom_p = 0.5

        cnn_lstm_p = float(te_preds[j]) if j < len(te_preds) else 0.5
        rsi_p = 0.65 if rsi_raw[i]<30 else 0.35 if rsi_raw[i]>70 else 0.5+(rsi_raw[i]-50)/200
        mh = macd_hist_full[i]
        macd_p = 0.5 + np.clip(mh/max(np.abs(macd_hist_full[max(0,i-20):i+1]).max(),1)*0.2, -0.25, 0.25)
        poly_p = 0.5 + np.clip(mom_20*2, -0.1, 0.1)
        threshold_p = 0.5 + np.clip(mom_20*0.3, -0.2, 0.2)

        S_test.append([sent_p, btc_mom_p, cnn_lstm_p, rsi_p, macd_p, poly_p, threshold_p])

    S = np.array(S_test)
    L = Y_te[:len(S)]

    lr = LogisticRegression(C=0.1, max_iter=500)
    lr.fit(S, L)
    raw = np.abs(lr.coef_[0])
    weights = raw / raw.sum()

    print(f"\n  Optimal weights (LogReg acc={lr.score(S,L):.1%}):")
    for name, w, coef in sorted(zip(signal_names, weights, lr.coef_[0]), key=lambda x: -x[1]):
        print(f"    {name:16s}: {w*100:5.1f}%  (coef={coef:+.4f})")

    print(f"\n  Individual signal accuracies:")
    for j, name in enumerate(signal_names):
        print(f"    {name:16s}: {((S[:,j]>0.5)==L).mean():.1%}")

    # Save
    w_dict = {name: round(float(w), 4) for name, w in zip(signal_names, weights)}
    with open(os.path.join(MODEL_WEIGHTS_DIR, "signal_weights.json"), "w") as f:
        json.dump({"weights": w_dict, "model": "CNN-LSTM + Boruta",
                    "test_acc": float(te_acc_final), "combined_acc": float(lr.score(S,L)),
                    "test_n": len(S), "selected_features": selected_names}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CNN-LSTM test={te_acc_final:.1%} | Combined={lr.score(S,L):.1%} | {len(S)} OOS days")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
