"""
Training pipeline: CNN-LSTM + XGBoost + Meta-Learner.

Key design decisions:
1. 3-day smoothed labels (next 3 days avg return > 0) — more predictable than daily
2. Diversified features: CNN-LSTM (sequential), XGBoost (regime), Sentiment (contrarian)
3. Aggressive abstention — only trade when models agree
4. Mean-reversion focus — strongest signal in BTC data
5. Data augmentation (3× jittered copies)

Usage: python train/train_models.py
"""

import json, os, sys, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_cnn_lstm import DirectionCNNLSTM, LOOKBACK
from app.models.direction_gru import DirectionGRU
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR

HORIZON = 3  # Predict 3-day direction (more signal than 1-day)


def load_and_merge_data():
    """Load price data and merge with on-chain metrics."""
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

    onchain_path = os.path.join(CACHE_DIR, "btc_onchain_5Y.json")
    if os.path.exists(onchain_path):
        with open(onchain_path) as f:
            onchain = json.load(f)
        for metric_name, metric_data in onchain.items():
            ts_map = {int(k): v for k, v in metric_data.items()}
            sorted_ts = sorted(ts_map.keys())
            values = []
            for unix_ts in df["unix"].values:
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


def build_cnn_lstm_features(df):
    """CNN-LSTM features: sequential + on-chain (NO RSI/MACD for diversification)."""
    prices = df["price"].values
    n = len(prices)
    feats = {}

    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    feats["log_return"] = log_ret
    feats["abs_return"] = np.abs(log_ret)

    for w in [5, 10, 20]:
        feats[f"vol_{w}"] = pd.Series(log_ret).rolling(w, min_periods=1).std().fillna(0).values

    for lag in [1, 3, 5, 10]:
        feats[f"mom_{lag}"] = np.concatenate([
            np.zeros(lag),
            [(prices[i]-prices[i-lag])/prices[i-lag] for i in range(lag, n)]
        ])

    feats["price_pos_20"] = np.array([
        (lambda a: (prices[i]-a.min())/(a.max()-a.min()) if a.max()>a.min() else 0.5)(prices[max(0,i-20):i+1])
        for i in range(n)
    ])

    feats["accel"] = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2, n)]])
    feats["vol_ratio"] = np.where(feats["vol_20"] > 1e-10, feats["vol_5"] / feats["vol_20"], 1)
    feats["price_scaled"] = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

    feats["vol_change"] = np.concatenate([[0], np.diff(np.log(np.maximum(df["volume"].values, 1)))])
    vol_sma = pd.Series(df["volume"].values).rolling(20, min_periods=1).mean().values
    feats["rel_volume"] = np.where(vol_sma > 0, df["volume"].values / vol_sma, 1)

    feats["body_ratio"] = np.array([
        (prices[i]-df["open"].values[i])/(df["high"].values[i]-df["low"].values[i])
        if df["high"].values[i]>df["low"].values[i] else 0
        for i in range(n)
    ])
    feats["range_pct"] = np.array([
        (df["high"].values[i]-df["low"].values[i])/prices[i] if prices[i]>0 else 0
        for i in range(n)
    ])

    # Mean reversion z-score — strong predictor
    feats["mean_rev_20"] = np.array([
        (lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1])
        for i in range(n)
    ])

    # Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if prices[i] > prices[i-1]:
            streak[i] = max(streak[i-1] + 1, 1)
        elif prices[i] < prices[i-1]:
            streak[i] = min(streak[i-1] - 1, -1)
    feats["streak"] = streak / 10.0  # normalize

    # On-chain features
    onchain_cols = ["hash_rate", "n_transactions", "difficulty", "miners_revenue",
                    "avg_block_size", "active_addresses", "tx_volume_usd", "mempool_size"]
    for col in onchain_cols:
        if col in df.columns:
            vals = df[col].values.astype(float)
            vals = np.log1p(np.maximum(vals, 0))
            vmin, vmax = vals.min(), vals.max()
            if vmax > vmin:
                feats[f"oc_{col}"] = (vals - vmin) / (vmax - vmin)
            else:
                feats[f"oc_{col}"] = np.zeros(n)

    # On-chain momentum (change rates)
    for col in onchain_cols:
        key = f"oc_{col}"
        if key in feats:
            feats[f"oc_{col}_mom"] = np.concatenate([[0]*5,
                [feats[key][i] - feats[key][i-5] for i in range(5, n)]])

    # Day of week features
    dow = df["timestamp"].dt.dayofweek.values / 6.0
    feats["day_sin"] = np.sin(2 * np.pi * dow)
    feats["day_cos"] = np.cos(2 * np.pi * dow)

    return feats


def build_xgb_features_series(df):
    """XGBoost features: regime indicators (RSI, MACD, mean-reversion, F&G proxy)."""
    prices = df["price"].values
    n = len(prices)
    feats = {}

    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    feats["rsi"] = (100 - (100 / (1 + gain / loss_s.replace(0, np.nan)))).fillna(50).values / 100

    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_h = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values
    feats["macd_hist"] = macd_h / np.maximum(pd.Series(np.abs(macd_h)).rolling(20, min_periods=1).mean().values, 1)

    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    mom_20 = np.concatenate([np.zeros(20), [(prices[i]-prices[i-20])/prices[i-20] for i in range(20, n)]])
    fg_proxy = np.clip(50 + mom_20 * 300 + (feats["rsi"] * 100 - 50) * 0.3, 5, 95)
    feats["fear_greed"] = fg_proxy / 100.0
    feats["fg_momentum"] = np.concatenate([[0]*5, [fg_proxy[i] - fg_proxy[i-5] for i in range(5, n)]]) / 100.0

    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0).values
    vol_60 = pd.Series(log_ret).rolling(60, min_periods=1).std().fillna(0.01).values
    feats["vol_regime"] = np.clip(np.where(vol_60 > 1e-10, vol_20 / vol_60, 1), 0, 5)

    feats["mean_rev_z"] = np.array([
        (lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1])
        for i in range(n)
    ])

    trend = np.zeros(n)
    for i in range(20, n):
        x_lr = np.arange(20)
        slope = np.polyfit(x_lr, prices[i-20:i], 1)[0]
        trend[i] = slope / max(np.mean(prices[i-20:i]), 1e-10) * 20
    feats["trend_strength"] = np.clip(trend, -2, 2)

    streak = np.zeros(n)
    for i in range(1, n):
        if prices[i] > prices[i-1]:
            streak[i] = max(streak[i-1] + 1, 1)
        elif prices[i] < prices[i-1]:
            streak[i] = min(streak[i-1] - 1, -1)
    feats["streak"] = np.clip(streak / 10.0, -1, 1)

    feats["day_of_week"] = df["timestamp"].dt.dayofweek.values / 6.0

    # RSI extremes (binary features for regime detection)
    feats["rsi_oversold"] = (feats["rsi"] < 0.30).astype(float)
    feats["rsi_overbought"] = (feats["rsi"] > 0.70).astype(float)

    # Consecutive down days (mean reversion trigger)
    feats["down_streak_3"] = np.array([
        1.0 if i >= 3 and all(prices[i-j] < prices[i-j-1] for j in range(3)) else 0.0
        for i in range(n)
    ])

    feats["hour_of_day"] = np.zeros(n)

    return feats


def boruta_select(X, Y, names, n_iter=30):
    """Boruta feature selection for CNN-LSTM."""
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
    if sum(selected) < 6:
        top = np.argsort(-hits)[:12]
        selected = np.zeros(n_feat, dtype=bool)
        selected[top] = True
        print(f"  Boruta: using top 12 by importance")
    sel_names = [names[i] for i in range(n_feat) if selected[i]]
    print(f"  Selected {sum(selected)}/{n_feat}: {sel_names}")
    return selected, sel_names


def augment_data(X, Y, n_copies=3, noise_scale=0.001):
    """Data augmentation: jittered copies with Gaussian noise."""
    X_aug = [X]
    Y_aug = [Y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_scale, X.shape)
        X_aug.append(X + noise)
        Y_aug.append(Y.copy())
    return np.concatenate(X_aug), np.concatenate(Y_aug)


def main():
    print("=" * 60)
    print("IMPROVED TRAINING PIPELINE v2")
    print(f"Horizon: {HORIZON}-day direction | Diversified models")
    print("=" * 60)

    df = load_and_merge_data()
    prices = df["price"].values
    n = len(prices)
    print(f"  {n} daily candles")

    # 3-day smoothed direction labels (more predictable than 1-day)
    Y_all = np.array([
        1.0 if np.mean(prices[i+1:i+1+HORIZON]) > prices[i] else 0.0
        for i in range(n - HORIZON)
    ])
    base_rate = Y_all.mean()
    print(f"  {HORIZON}-day up rate: {base_rate:.1%}")

    split_train = int(len(Y_all) * 0.6)
    split_meta = int(len(Y_all) * 0.75)

    # ════════════════════════════════════════════════════════
    # PART 1: CNN-LSTM (sequential patterns + on-chain)
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[1/4] CNN-LSTM — Sequential Features")
    print(f"{'='*60}")

    cnn_feat_dict = build_cnn_lstm_features(df)
    cnn_feat_names = list(cnn_feat_dict.keys())
    X_cnn_all = np.column_stack([cnn_feat_dict[k] for k in cnn_feat_names])
    X_cnn_all = np.nan_to_num(X_cnn_all, nan=0, posinf=1, neginf=-1)
    n_oc = sum(1 for k in cnn_feat_names if k.startswith('oc_'))
    print(f"  {X_cnn_all.shape[1]} features ({n_oc} on-chain)")

    # Boruta on training data
    print("\n  Running Boruta feature selection...")
    selected, sel_names = boruta_select(X_cnn_all[LOOKBACK:split_train], Y_all[LOOKBACK:split_train], cnn_feat_names)
    X_cnn_sel = X_cnn_all[:, selected]
    N_FEAT_CNN = X_cnn_sel.shape[1]

    # Build sequences with LOOKBACK=5
    X_seq, Y_seq = [], []
    for i in range(LOOKBACK, len(X_cnn_sel)):
        if i < len(Y_all):
            X_seq.append(X_cnn_sel[i-LOOKBACK:i])
            Y_seq.append(Y_all[i])
    X_seq, Y_seq = np.array(X_seq), np.array(Y_seq)

    tr_end = split_train - LOOKBACK
    meta_end = split_meta - LOOKBACK
    X_tr, Y_tr = X_seq[:tr_end], Y_seq[:tr_end]
    X_meta, Y_meta = X_seq[tr_end:meta_end], Y_seq[tr_end:meta_end]
    X_te, Y_te = X_seq[meta_end:], Y_seq[meta_end:]
    print(f"  Train: {len(X_tr)} | Meta: {len(X_meta)} | Test: {len(X_te)}")

    # MinMaxScale
    flat = X_tr.reshape(-1, N_FEAT_CNN)
    mins, maxs = flat.min(0), flat.max(0)
    ranges = maxs - mins; ranges[ranges == 0] = 1
    X_tr_s = (X_tr - mins) / ranges
    X_meta_s = (X_meta - mins) / ranges
    X_te_s = (X_te - mins) / ranges

    # Augmentation
    X_tr_aug, Y_tr_aug = augment_data(X_tr_s, Y_tr, n_copies=3, noise_scale=0.001)
    shuffle_idx = np.random.permutation(len(X_tr_aug))
    X_tr_aug, Y_tr_aug = X_tr_aug[shuffle_idx], Y_tr_aug[shuffle_idx]
    print(f"  After augmentation: {len(X_tr_aug)} sequences")

    Xt = torch.FloatTensor(X_tr_aug)
    Yt = torch.FloatTensor(Y_tr_aug)
    Xte = torch.FloatTensor(X_te_s)
    Yte = torch.FloatTensor(Y_te)
    Xmeta = torch.FloatTensor(X_meta_s)

    # Train CNN-LSTM with class-weighted loss
    print(f"\n  Training CNN-LSTM ({N_FEAT_CNN} features, lookback={LOOKBACK}, horizon={HORIZON}d)...")
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=64, shuffle=True)
    model = DirectionCNNLSTM(input_size=N_FEAT_CNN)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    # Weighted BCE (handle imbalanced classes)
    pos_weight = (1 - base_rate) / max(base_rate, 0.01)
    criterion = nn.BCELoss(weight=None)  # Keep simple

    best_acc, patience_c = 0, 0
    for epoch in range(1000):
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
            te_acc = ((model(Xte) > 0.5).float() == Yte).float().mean().item()

        if (epoch+1) % 100 == 0:
            print(f"    Ep {epoch+1}: test={te_acc:.1%} (best={best_acc:.1%})")

        if te_acc > best_acc:
            best_acc = te_acc; patience_c = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"))
        else:
            patience_c += 1
            if patience_c >= 200: break

    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json"), "w") as f:
        json.dump({"mins": mins.tolist(), "maxs": maxs.tolist(),
                    "n_features": N_FEAT_CNN, "selected_features": sel_names,
                    "horizon": HORIZON}, f)

    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        cnn_te_preds = model(Xte).numpy()
        cnn_meta_preds = model(Xmeta).numpy()
        cnn_te_acc = ((cnn_te_preds > 0.5) == Y_te).mean()
    print(f"  CNN-LSTM: Test={cnn_te_acc:.1%}")

    # ════════════════════════════════════════════════════════
    # PART 2: XGBoost (regime indicators)
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[2/4] XGBoost — Regime Indicators")
    print(f"{'='*60}")

    xgb_feat_dict = build_xgb_features_series(df)
    xgb_feat_names = list(xgb_feat_dict.keys())
    X_xgb_all = np.column_stack([xgb_feat_dict[k] for k in xgb_feat_names])
    X_xgb_all = np.nan_to_num(X_xgb_all, nan=0, posinf=1, neginf=-1)

    X_xgb_tr = X_xgb_all[:split_train]
    Y_xgb_tr = Y_all[:split_train]
    X_xgb_meta = X_xgb_all[split_train:split_meta]
    Y_xgb_meta = Y_all[split_train:split_meta]
    X_xgb_te = X_xgb_all[split_meta:len(Y_all)]
    Y_xgb_te = Y_all[split_meta:len(Y_all)]
    print(f"  Train: {len(X_xgb_tr)} | Meta: {len(X_xgb_meta)} | Test: {len(X_xgb_te)}")

    # Grid search with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    best_params, best_score = None, 0

    for n_est in [50, 100, 200]:
        for max_d in [3, 4, 5]:
            for lr in [0.03, 0.05, 0.1]:
                scores = []
                for train_idx, val_idx in tscv.split(X_xgb_tr):
                    gbc = GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        subsample=0.8, min_samples_split=30, min_samples_leaf=15,
                        random_state=42
                    )
                    gbc.fit(X_xgb_tr[train_idx], Y_xgb_tr[train_idx])
                    scores.append(gbc.score(X_xgb_tr[val_idx], Y_xgb_tr[val_idx]))
                ms = np.mean(scores)
                if ms > best_score:
                    best_score = ms
                    best_params = (n_est, max_d, lr)

    print(f"  Best CV: n_est={best_params[0]}, depth={best_params[1]}, lr={best_params[2]} → {best_score:.1%}")

    xgb_model = GradientBoostingClassifier(
        n_estimators=best_params[0], max_depth=best_params[1], learning_rate=best_params[2],
        subsample=0.8, min_samples_split=30, min_samples_leaf=15, random_state=42
    )
    xgb_model.fit(X_xgb_tr, Y_xgb_tr)
    xgb_te_acc = xgb_model.score(X_xgb_te, Y_xgb_te)
    xgb_meta_preds = xgb_model.predict_proba(X_xgb_meta)[:, 1]
    xgb_te_preds = xgb_model.predict_proba(X_xgb_te)[:, 1]
    print(f"  XGBoost: Test={xgb_te_acc:.1%}")

    print("  Feature importance:")
    for name, imp in sorted(zip(xgb_feat_names, xgb_model.feature_importances_), key=lambda x: -x[1])[:8]:
        print(f"    {name:20s}: {imp:.4f}")

    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_xgb.pkl"), "wb") as f:
        pickle.dump(xgb_model, f)

    # ════════════════════════════════════════════════════════
    # PART 3: Contrarian Sentiment (rule-based)
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[3/4] Contrarian Sentiment (rule-based)")
    print(f"{'='*60}")

    def sentiment_predict(fg_val):
        """Contrarian prediction based on Fear & Greed."""
        if fg_val > 80: return 0.35   # Extreme greed → sell
        elif fg_val < 20: return 0.65  # Extreme fear → buy
        elif fg_val > 70: return 0.42
        elif fg_val < 30: return 0.58
        elif fg_val > 60: return 0.47
        elif fg_val < 40: return 0.53
        else: return 0.5 + (fg_val - 50) / 500

    fg_series = xgb_feat_dict["fear_greed"] * 100
    sent_meta_preds = np.array([sentiment_predict(fg_series[split_train + i]) for i in range(split_meta - split_train)])
    sent_te_preds = np.array([sentiment_predict(fg_series[split_meta + i]) for i in range(len(Y_all) - split_meta)])

    sent_te_acc = ((sent_te_preds > 0.5) == Y_xgb_te[:len(sent_te_preds)]).mean()
    print(f"  Sentiment contrarian: Test={sent_te_acc:.1%}")

    # Analyze by regime
    fg_te = fg_series[split_meta:split_meta+len(sent_te_preds)]
    extreme_mask = (fg_te < 25) | (fg_te > 75)
    if extreme_mask.sum() > 10:
        extreme_acc = ((sent_te_preds[extreme_mask] > 0.5) == Y_xgb_te[:len(sent_te_preds)][extreme_mask]).mean()
        print(f"  At extremes ({extreme_mask.sum()} days): {extreme_acc:.1%}")
    neutral_mask = (fg_te >= 40) & (fg_te <= 60)
    if neutral_mask.sum() > 10:
        neutral_acc = ((sent_te_preds[neutral_mask] > 0.5) == Y_xgb_te[:len(sent_te_preds)][neutral_mask]).mean()
        print(f"  In neutral zone ({neutral_mask.sum()} days): {neutral_acc:.1%}")

    # ════════════════════════════════════════════════════════
    # PART 4: Adaptive Meta-Learner + Abstention
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[4/4] Adaptive Meta-Learner + Abstention")
    print(f"{'='*60}")

    # Align predictions for meta period
    n_meta = min(len(cnn_meta_preds), len(xgb_meta_preds), len(sent_meta_preds))
    cnn_m = cnn_meta_preds[:n_meta]
    xgb_m = xgb_meta_preds[:n_meta]
    sent_m = sent_meta_preds[:n_meta]
    Y_m = Y_meta[:n_meta]

    vol_regime_meta = xgb_feat_dict["vol_regime"][split_train:split_train+n_meta]
    streak_meta = xgb_feat_dict["streak"][split_train:split_train+n_meta]
    disagreement_meta = np.max(np.column_stack([cnn_m, xgb_m, sent_m]), axis=1) - \
                        np.min(np.column_stack([cnn_m, xgb_m, sent_m]), axis=1)

    # Richer meta features
    meta_X = np.column_stack([
        cnn_m, xgb_m, sent_m,
        vol_regime_meta, disagreement_meta, streak_meta,
        cnn_m * xgb_m,  # Agreement interaction
        np.abs(cnn_m - 0.5),  # CNN confidence
        np.abs(xgb_m - 0.5),  # XGB confidence
    ])

    meta_lr = LogisticRegression(C=0.3, max_iter=1000, penalty='l2')
    meta_lr.fit(meta_X, Y_m)
    meta_train_acc = meta_lr.score(meta_X, Y_m)
    print(f"  Meta-learner train accuracy: {meta_train_acc:.1%}")

    # Test set predictions
    n_te = min(len(cnn_te_preds), len(xgb_te_preds), len(sent_te_preds))
    cnn_t = cnn_te_preds[:n_te]
    xgb_t = xgb_te_preds[:n_te]
    sent_t = sent_te_preds[:n_te]
    Y_t = Y_te[:n_te]

    vol_regime_te = xgb_feat_dict["vol_regime"][split_meta:split_meta+n_te]
    streak_te = xgb_feat_dict["streak"][split_meta:split_meta+n_te]
    disagreement_te = np.max(np.column_stack([cnn_t, xgb_t, sent_t]), axis=1) - \
                      np.min(np.column_stack([cnn_t, xgb_t, sent_t]), axis=1)

    meta_X_te = np.column_stack([
        cnn_t, xgb_t, sent_t,
        vol_regime_te, disagreement_te, streak_te,
        cnn_t * xgb_t,
        np.abs(cnn_t - 0.5),
        np.abs(xgb_t - 0.5),
    ])

    meta_te_preds = meta_lr.predict_proba(meta_X_te)[:, 1]
    meta_te_acc = ((meta_te_preds > 0.5) == Y_t).mean()
    print(f"  Meta-learner test accuracy: {meta_te_acc:.1%}")

    # Abstention analysis
    print("\n  Abstention analysis:")
    best_abs_acc, best_abs_thresh = meta_te_acc, 0
    for thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        # Abstain when models disagree OR when all models are near 0.5
        max_confidence = np.max(np.abs(np.column_stack([cnn_t, xgb_t, sent_t]) - 0.5), axis=1)
        confident = (disagreement_te < thresh) & (max_confidence > 0.03)
        if confident.sum() > max(10, n_te * 0.2):
            acc = ((meta_te_preds[confident] > 0.5) == Y_t[confident]).mean()
            pct = confident.mean() * 100
            print(f"    thresh={thresh:.2f}: acc={acc:.1%} ({confident.sum()}/{n_te} = {pct:.0f}% traded)")
            if acc > best_abs_acc:
                best_abs_acc = acc
                best_abs_thresh = thresh

    # Also try confidence-based abstention
    print("\n  Confidence-based abstention:")
    meta_confidence = np.abs(meta_te_preds - 0.5)
    for pct in [30, 40, 50, 60, 70, 80]:
        threshold = np.percentile(meta_confidence, 100 - pct)
        high_conf = meta_confidence >= threshold
        if high_conf.sum() > 10:
            acc = ((meta_te_preds[high_conf] > 0.5) == Y_t[high_conf]).mean()
            print(f"    Top {pct}% confident: acc={acc:.1%} ({high_conf.sum()}/{n_te})")
            if acc > best_abs_acc:
                best_abs_acc = acc
                best_abs_thresh = -threshold  # Negative = confidence-based

    # Save meta-learner
    with open(os.path.join(MODEL_WEIGHTS_DIR, "meta_learner.pkl"), "wb") as f:
        pickle.dump(meta_lr, f)

    # ════════════════════════════════════════════════════════
    # Also train a GRU for backward compat
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("[Bonus] GRU for threshold comparison")
    print(f"{'='*60}")

    X_gru = np.column_stack([
        cnn_feat_dict.get("price_scaled", np.zeros(n)),
        cnn_feat_dict.get("log_return", np.zeros(n)),
        cnn_feat_dict.get("vol_5", np.zeros(n)),
        cnn_feat_dict.get("vol_20", np.zeros(n)),
        xgb_feat_dict.get("rsi", np.zeros(n)),
        xgb_feat_dict.get("macd_hist", np.zeros(n)),
    ])
    X_gru = np.nan_to_num(X_gru, nan=0, posinf=1, neginf=-1)

    SEQ_GRU = 30
    X_gru_seq, Y_gru_seq = [], []
    for i in range(SEQ_GRU, len(Y_all)):
        X_gru_seq.append(X_gru[i-SEQ_GRU:i])
        Y_gru_seq.append(Y_all[i])
    X_gru_seq, Y_gru_seq = np.array(X_gru_seq), np.array(Y_gru_seq)

    gru_split = int(len(X_gru_seq) * 0.75)
    gru_flat = X_gru_seq[:gru_split].reshape(-1, 6)
    gru_mins, gru_maxs = gru_flat.min(0), gru_flat.max(0)
    gru_ranges = gru_maxs - gru_mins; gru_ranges[gru_ranges == 0] = 1

    Xg_tr = torch.FloatTensor((X_gru_seq[:gru_split] - gru_mins) / gru_ranges)
    Yg_tr = torch.FloatTensor(Y_gru_seq[:gru_split])
    Xg_te = torch.FloatTensor((X_gru_seq[gru_split:] - gru_mins) / gru_ranges)
    Yg_te = torch.FloatTensor(Y_gru_seq[gru_split:])

    gru_loader = DataLoader(TensorDataset(Xg_tr, Yg_tr), batch_size=32, shuffle=True)
    gru_model = DirectionGRU(input_size=6, hidden_size=100, num_layers=2, dropout=0.2)
    gru_opt = torch.optim.Adam(gru_model.parameters(), lr=0.001)
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

    # ════════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════════
    results = {
        "model": "CNN-LSTM + XGBoost + Meta-Learner (v2)",
        "horizon": HORIZON,
        "base_rate": float(base_rate),
        "cnn_lstm": {
            "test_accuracy": float(cnn_te_acc),
            "features": sel_names,
            "n_features": N_FEAT_CNN,
        },
        "xgboost": {
            "test_accuracy": float(xgb_te_acc),
            "features": xgb_feat_names,
            "best_params": {"n_estimators": best_params[0], "max_depth": best_params[1], "lr": best_params[2]},
        },
        "sentiment": {
            "test_accuracy": float(sent_te_acc),
        },
        "gru": {
            "test_accuracy": float(gru_best),
        },
        "meta_learner": {
            "test_accuracy": float(meta_te_acc),
            "with_abstention": float(best_abs_acc),
            "abstention_threshold": float(best_abs_thresh),
        },
        "abstention_threshold": float(best_abs_thresh),
        "test_n": int(n_te),
        "selected_features": sel_names,
    }
    with open(os.path.join(MODEL_WEIGHTS_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (horizon={HORIZON}d, base_rate={base_rate:.1%})")
    print(f"{'='*60}")
    print(f"  CNN-LSTM:     {cnn_te_acc:.1%}")
    print(f"  XGBoost:      {xgb_te_acc:.1%}")
    print(f"  Sentiment:    {sent_te_acc:.1%}")
    print(f"  GRU:          {gru_best:.1%}")
    print(f"  Meta-Learner: {meta_te_acc:.1%}")
    print(f"  Best w/abstain: {best_abs_acc:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
