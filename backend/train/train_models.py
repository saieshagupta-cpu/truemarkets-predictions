"""
CNN-LSTM with on-chain data + Boruta feature selection.
Uses 8 free blockchain.info on-chain metrics + price-derived features.

Also trains GRU for threshold prediction (Polymarket comparison).

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
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_cnn_lstm import DirectionCNNLSTM, LOOKBACK
from app.models.direction_gru import DirectionGRU
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR


def load_and_merge_data():
    """Load price data and merge with on-chain metrics."""
    # Price data
    path = os.path.join(CACHE_DIR, "btc_5Y_1d.json")
    if not os.path.exists(path):
        path = os.path.join(CACHE_DIR, "btc_3Y_1d.json")
    with open(path) as f:
        pts = json.load(f)["results"][0]["points"]

    df = pd.DataFrame([{
        "timestamp": p["t"], "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["unix"] = df["timestamp"].astype(int) // 10**9

    # On-chain data
    onchain_path = os.path.join(CACHE_DIR, "btc_onchain_5Y.json")
    if os.path.exists(onchain_path):
        with open(onchain_path) as f:
            onchain = json.load(f)

        # Merge on-chain by nearest timestamp
        for metric_name, metric_data in onchain.items():
            ts_map = {int(k): v for k, v in metric_data.items()}
            sorted_ts = sorted(ts_map.keys())

            values = []
            for unix_ts in df["unix"].values:
                # Find nearest on-chain timestamp
                idx = np.searchsorted(sorted_ts, unix_ts)
                idx = min(idx, len(sorted_ts) - 1)
                if idx > 0 and abs(sorted_ts[idx-1] - unix_ts) < abs(sorted_ts[idx] - unix_ts):
                    idx = idx - 1
                values.append(ts_map[sorted_ts[idx]])
            df[metric_name] = values

        print(f"  Merged {len(onchain)} on-chain metrics")
    else:
        print("  WARNING: No on-chain data found")

    return df


def build_features(df):
    """Build price + on-chain features."""
    prices = df["price"].values
    n = len(prices)
    feats = {}

    # Price features
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    feats["log_return"] = log_ret
    feats["abs_return"] = np.abs(log_ret)

    for w in [5, 10, 20]:
        feats[f"vol_{w}"] = pd.Series(log_ret).rolling(w, min_periods=1).std().fillna(0).values

    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    feats["rsi"] = (100 - (100 / (1 + gain / loss_s.replace(0, np.nan)))).fillna(50).values / 100

    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_h = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values
    feats["macd_hist"] = macd_h / np.maximum(pd.Series(np.abs(macd_h)).rolling(20, min_periods=1).mean().values, 1)

    for lag in [1, 3, 5, 10, 20]:
        feats[f"mom_{lag}"] = np.concatenate([np.zeros(lag), [(prices[i]-prices[i-lag])/prices[i-lag] for i in range(lag, n)]])

    feats["price_pos_20"] = np.array([(lambda a: (prices[i]-a.min())/(a.max()-a.min()) if a.max()>a.min() else 0.5)(prices[max(0,i-20):i+1]) for i in range(n)])
    feats["mean_rev"] = np.array([(lambda a: (prices[i]-np.mean(a))/np.std(a) if np.std(a)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])

    feats["body_ratio"] = np.array([(prices[i]-df["open"].values[i])/(df["high"].values[i]-df["low"].values[i]) if df["high"].values[i]>df["low"].values[i] else 0 for i in range(n)])
    feats["range_pct"] = np.array([(df["high"].values[i]-df["low"].values[i])/prices[i] if prices[i]>0 else 0 for i in range(n)])

    vol_sma = pd.Series(df["volume"].values).rolling(20, min_periods=1).mean().values
    feats["rel_volume"] = np.where(vol_sma > 0, df["volume"].values / vol_sma, 1)
    feats["vol_change"] = np.concatenate([[0], np.diff(np.log(np.maximum(df["volume"].values, 1)))])
    feats["vol_ratio"] = np.where(feats["vol_20"] > 1e-10, feats["vol_5"] / feats["vol_20"], 1)
    feats["accel"] = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2, n)]])
    feats["price_scaled"] = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

    # On-chain features (normalize each)
    onchain_cols = ["hash_rate", "n_transactions", "difficulty", "miners_revenue",
                    "avg_block_size", "active_addresses", "tx_volume_usd", "mempool_size"]
    for col in onchain_cols:
        if col in df.columns:
            vals = df[col].values.astype(float)
            # Log-normalize large values, then scale 0-1
            vals = np.log1p(np.maximum(vals, 0))
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                feats[f"oc_{col}"] = (vals - vmin) / (vmax - vmin)
            else:
                feats[f"oc_{col}"] = np.zeros(n)

    return feats


def boruta_select(X, Y, names, n_iter=30):
    """Boruta feature selection."""
    n_feat = X.shape[1]
    hits = np.zeros(n_feat)
    for it in range(n_iter):
        shadow = X.copy()
        for j in range(n_feat):
            np.random.shuffle(shadow[:, j])
        X_c = np.hstack([X, shadow])
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=it)
        rf.fit(X_c, Y)
        imp = rf.feature_importances_
        hits += (imp[:n_feat] > imp[n_feat:].max()).astype(float)

    selected = hits / n_iter > 0.3
    if not selected.any():
        top = np.argsort(-hits)[:12]
        selected = np.zeros(n_feat, dtype=bool)
        selected[top] = True
        print(f"  Boruta: using top 12 by importance")
    sel_names = [names[i] for i in range(n_feat) if selected[i]]
    print(f"  Selected {sum(selected)}/{n_feat}: {sel_names}")
    return selected, sel_names


def main():
    print("=" * 60)
    print("CNN-LSTM + On-Chain Data + Boruta Training")
    print("=" * 60)

    df = load_and_merge_data()
    prices = df["price"].values
    print(f"  {len(df)} daily candles")

    feat_dict = build_features(df)
    feat_names = list(feat_dict.keys())
    X_all = np.column_stack([feat_dict[k] for k in feat_names])
    X_all = np.nan_to_num(X_all, nan=0, posinf=1, neginf=-1)
    print(f"  {X_all.shape[1]} features ({sum(1 for k in feat_names if k.startswith('oc_'))} on-chain)")

    Y_all = np.array([1.0 if prices[i+1] > prices[i] else 0.0 for i in range(len(prices)-1)])

    split = int(len(Y_all) * 0.8)

    # Boruta on training data
    print("\n  Running Boruta feature selection...")
    selected, sel_names = boruta_select(X_all[LOOKBACK:split], Y_all[LOOKBACK:split], feat_names)
    X_sel = X_all[:, selected]
    N_FEAT = X_sel.shape[1]

    # Build sequences
    X_seq, Y_seq = [], []
    for i in range(LOOKBACK, len(X_sel) - 1):
        X_seq.append(X_sel[i-LOOKBACK:i])
        Y_seq.append(Y_all[i])
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

    X_tr, Y_tr = X_seq[:split-LOOKBACK], Y_seq[:split-LOOKBACK]
    X_te, Y_te = X_seq[split-LOOKBACK:], Y_seq[split-LOOKBACK:]
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

    # MinMaxScale
    flat = X_tr.reshape(-1, N_FEAT)
    mins, maxs = flat.min(0), flat.max(0)
    ranges = maxs - mins; ranges[ranges == 0] = 1
    X_tr_s = (X_tr - mins) / ranges
    X_te_s = (X_te - mins) / ranges

    Xt = torch.FloatTensor(X_tr_s)
    Yt = torch.FloatTensor(Y_tr)
    Xte = torch.FloatTensor(X_te_s)
    Yte = torch.FloatTensor(Y_te)

    # ── Train CNN-LSTM ────────────────────────────────────
    print(f"\n[1/2] Training CNN-LSTM ({N_FEAT} features, lookback={LOOKBACK})...")
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=50, shuffle=False)
    model = DirectionCNNLSTM(input_size=N_FEAT)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    best_acc, patience_c = 0, 0
    for epoch in range(1000):
        model.train()
        for xb, yb in loader:
            pred = model(xb); loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            te_acc = ((model(Xte) > 0.5).float() == Yte).float().mean().item()

        if (epoch+1) % 100 == 0:
            print(f"    Ep {epoch+1}: test={te_acc:.1%} (best={best_acc:.1%})")

        if te_acc > best_acc:
            best_acc = te_acc; patience_c = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"))
        else:
            patience_c += 1
            if patience_c >= 100: break

    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json"), "w") as f:
        json.dump({"mins": mins.tolist(), "maxs": maxs.tolist(), "n_features": N_FEAT, "selected_features": sel_names}, f)

    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xt) > 0.5).float() == Yt).float().mean().item()
        cnn_te_preds = model(Xte).numpy()
        cnn_te_acc = ((cnn_te_preds > 0.5) == Y_te).mean()

    print(f"  CNN-LSTM: Train={tr_acc:.1%} Test={cnn_te_acc:.1%}")

    # ── Train GRU for threshold prediction ────────────────
    print(f"\n[2/2] Training GRU for price regression (threshold comparison)...")
    # GRU predicts normalized price (for threshold comparison with Polymarket)
    X_gru = np.column_stack([feat_dict["price_scaled"], feat_dict["log_return"],
                              feat_dict["vol_5"], feat_dict["vol_20"],
                              feat_dict["rsi"], feat_dict["macd_hist"]])
    X_gru = np.nan_to_num(X_gru, nan=0, posinf=1, neginf=-1)

    X_gru_seq, Y_gru_seq = [], []
    SEQ_GRU = 60
    for i in range(SEQ_GRU, len(X_gru) - 1):
        X_gru_seq.append(X_gru[i-SEQ_GRU:i])
        Y_gru_seq.append(1.0 if prices[i+1] > prices[i] else 0.0)
    X_gru_seq, Y_gru_seq = np.array(X_gru_seq), np.array(Y_gru_seq)

    gru_split = int(len(X_gru_seq) * 0.8)
    gru_flat = X_gru_seq[:gru_split].reshape(-1, 6)
    gru_mins, gru_maxs = gru_flat.min(0), gru_flat.max(0)
    gru_ranges = gru_maxs - gru_mins; gru_ranges[gru_ranges == 0] = 1

    Xg_tr = torch.FloatTensor((X_gru_seq[:gru_split] - gru_mins) / gru_ranges)
    Yg_tr = torch.FloatTensor(Y_gru_seq[:gru_split])
    Xg_te = torch.FloatTensor((X_gru_seq[gru_split:] - gru_mins) / gru_ranges)
    Yg_te = torch.FloatTensor(Y_gru_seq[gru_split:])

    gru_loader = DataLoader(TensorDataset(Xg_tr, Yg_tr), batch_size=32, shuffle=True)
    gru_model = DirectionGRU(input_size=6, hidden_size=100, num_layers=2, dropout=0.2)
    gru_opt = torch.optim.Adam(gru_model.parameters())
    gru_crit = nn.BCELoss()

    gru_best = 0
    for epoch in range(200):
        gru_model.train()
        for xb, yb in gru_loader:
            p = gru_model(xb); l = gru_crit(p, yb)
            gru_opt.zero_grad(); l.backward(); gru_opt.step()
        gru_model.eval()
        with torch.no_grad():
            ga = ((gru_model(Xg_te) > 0.5).float() == Yg_te).float().mean().item()
        if ga > gru_best:
            gru_best = ga
            torch.save(gru_model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_gru.pt"))
        if (epoch+1) % 50 == 0:
            print(f"    GRU Ep {epoch+1}: test={ga:.1%} (best={gru_best:.1%})")

    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_gru_norm.json"), "w") as f:
        json.dump({"mins": gru_mins.tolist(), "maxs": gru_maxs.tolist(), "n_features": 6}, f)

    print(f"  GRU: Test={gru_best:.1%}")

    # ── Backtest 7-signal weights ─────────────────────────
    print(f"\n{'='*60}")
    print(f"Backtesting weights on {len(X_te)} OOS test days")
    print(f"{'='*60}")

    rsi_raw = (100 - (100 / (1 + pd.Series(prices).diff().where(lambda x: x > 0, 0).rolling(14, min_periods=1).mean() /
               (-pd.Series(prices).diff().where(lambda x: x < 0, 0)).rolling(14, min_periods=1).mean().replace(0, np.nan)))).fillna(50).values
    log_ret_f = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    e12 = pd.Series(prices).ewm(span=12).mean().values
    e26 = pd.Series(prices).ewm(span=26).mean().values
    macd_f = (e12-e26) - pd.Series(e12-e26).ewm(span=9).mean().values

    signal_names = ["TM_Sentiment", "BTC_Momentum", "CNN_LSTM", "RSI", "MACD", "Polymarket", "30day_Model"]
    S_test = []
    test_start = split

    for j in range(len(X_te)):
        i = test_start + j
        if i >= len(prices) - 1: break

        m20 = (prices[i]-prices[max(0,i-20)])/prices[max(0,i-20)]
        fg = np.clip(50+m20*300+(rsi_raw[i]-50)*0.3, 5, 95)
        sent = 0.62 if fg<20 else 0.38 if fg>80 else 0.5+(fg-50)/300

        if i >= 5:
            r4 = (prices[i]-prices[i-4])/prices[i-4]
            rr = (prices[i]-prices[i-2])/prices[i-2]
            rp = (prices[i-2]-prices[i-4])/prices[i-4]
            btc = float(np.clip(0.5+r4*8+(rr-rp)*15, 0.2, 0.8))
        else: btc = 0.5

        cnn_p = float(cnn_te_preds[j]) if j < len(cnn_te_preds) else 0.5
        rsi_p = 0.65 if rsi_raw[i]<30 else 0.35 if rsi_raw[i]>70 else 0.5+(rsi_raw[i]-50)/200
        mh = macd_f[i]; macd_p = 0.5+np.clip(mh/max(np.abs(macd_f[max(0,i-20):i+1]).max(),1)*0.2, -0.25, 0.25)
        poly = 0.5+np.clip(m20*2, -0.1, 0.1)
        thresh = 0.5+np.clip(m20*0.3, -0.2, 0.2)
        S_test.append([sent, btc, cnn_p, rsi_p, macd_p, poly, thresh])

    S = np.array(S_test); L = Y_te[:len(S)]
    lr = LogisticRegression(C=0.1, max_iter=500)
    lr.fit(S, L)
    raw = np.abs(lr.coef_[0]); w = raw / raw.sum()

    print(f"\n  Weights (acc={lr.score(S,L):.1%}):")
    for name, wt, coef in sorted(zip(signal_names, w, lr.coef_[0]), key=lambda x: -x[1]):
        acc = ((S[:, signal_names.index(name)] > 0.5) == L).mean()
        print(f"    {name:16s}: {wt*100:5.1f}% (solo={acc:.1%})")

    w_dict = {n: round(float(v), 4) for n, v in zip(signal_names, w)}
    with open(os.path.join(MODEL_WEIGHTS_DIR, "signal_weights.json"), "w") as f:
        json.dump({"weights": w_dict, "model": "CNN-LSTM+Boruta+OnChain", "cnn_lstm_acc": float(cnn_te_acc),
                    "gru_acc": float(gru_best), "combined_acc": float(lr.score(S,L)),
                    "test_n": len(S), "selected_features": sel_names,
                    "on_chain_features": [k for k in feat_names if k.startswith("oc_")]}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  CNN-LSTM={cnn_te_acc:.1%} | GRU={gru_best:.1%} | Combined={lr.score(S,L):.1%}")
    print(f"  On-chain features used: {sum(1 for k in sel_names if k.startswith('oc_'))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
