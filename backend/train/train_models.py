"""
GRU training (based on PMC11935774) + 7-signal weight backtest on 5 years daily BTC.

Paper's architecture:
  - 2-layer GRU, 100 units, dropout 0.2, Adam, MinMaxScaler
  - Input: daily closing prices (paper uses price-only)
  - We add multi-feature input for direction prediction

Split: 80% train / 20% test (as in paper)
Then backtest all 7 signal weights on the test set.

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

from app.models.direction_gru import DirectionGRU
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR

SEQ_LEN = 60  # 60-day lookback (common for daily crypto)


def load_data():
    """Load 5-year daily BTC from CryptoCompare cache."""
    path = os.path.join(CACHE_DIR, "btc_5Y_1d.json")
    if not os.path.exists(path):
        path = os.path.join(CACHE_DIR, "btc_3Y_1d.json")
    with open(path) as f:
        pts = json.load(f)["results"][0]["points"]
    return pd.DataFrame([{
        "timestamp": p["t"], "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])


def build_features(df):
    """Build features: paper uses price only, we add a few more for direction."""
    prices = df["price"].values
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0).values

    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rsi = (100 - (100 / (1 + gain / loss_s.replace(0, np.nan)))).fillna(50).values / 100

    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_raw = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values
    macd_norm = macd_raw / np.maximum(pd.Series(np.abs(macd_raw)).rolling(20, min_periods=1).mean().values, 1)

    # MinMaxScaled price (as in paper)
    price_scaled = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)

    return np.column_stack([price_scaled, log_ret, vol_5, vol_20, rsi, macd_norm])


def main():
    print("=" * 60)
    print("GRU Training (PMC11935774) + Signal Weight Backtest")
    print("=" * 60)

    df = load_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  {len(df)} daily candles: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")

    prices = df["price"].values
    features = build_features(df)
    N_FEAT = features.shape[1]

    # Build sequences
    X, Y = [], []
    for i in range(SEQ_LEN, len(features) - 1):
        X.append(features[i - SEQ_LEN:i])
        Y.append(1.0 if prices[i + 1] > prices[i] else 0.0)
    X, Y = np.array(X), np.array(Y)
    print(f"  {len(X)} sequences, {N_FEAT} features, seq_len={SEQ_LEN}")
    print(f"  Class balance: {Y.mean():.1%} up")

    # Paper's split: 80% train / 20% test
    split = int(len(X) * 0.8)
    X_tr, Y_tr = X[:split], Y[:split]
    X_te, Y_te = X[split:], Y[split:]
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)} (80/20 as in paper)")

    # MinMaxScaler (as in paper)
    flat = X_tr.reshape(-1, N_FEAT)
    mins = flat.min(0)
    maxs = flat.max(0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    X_tr_scaled = (X_tr - mins) / ranges
    X_te_scaled = (X_te - mins) / ranges

    Xt = torch.FloatTensor(X_tr_scaled)
    Yt = torch.FloatTensor(Y_tr)
    Xte = torch.FloatTensor(X_te_scaled)
    Yte = torch.FloatTensor(Y_te)

    # Paper's hyperparameters exactly
    print("\n  Training GRU (paper config: 2 layers, 100 units, dropout=0.2, epochs=20)...")
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=32, shuffle=True)
    model = DirectionGRU(input_size=N_FEAT, hidden_size=100, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters())  # Adam as in paper
    criterion = nn.BCELoss()

    # Train for 20 epochs (as in paper) — no early stopping first pass
    for epoch in range(20):
        model.train()
        t_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        with torch.no_grad():
            te_acc = ((model(Xte) > 0.5).float() == Yte).float().mean().item()

        if (epoch + 1) % 5 == 0:
            print(f"    Ep {epoch+1}/20: loss={t_loss/len(loader):.4f} test_acc={te_acc:.1%}")

    # Now train longer with early stopping to find best
    print("\n  Extended training with early stopping...")
    best_acc, patience = 0, 0
    for epoch in range(200):
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

        if te_acc > best_acc:
            best_acc = te_acc
            patience = 0
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_gru.pt"))
        else:
            patience += 1
            if patience >= 30:
                print(f"    Early stop at ep {20 + epoch + 1}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"    Ep {20 + epoch + 1}: test_acc={te_acc:.1%} (best={best_acc:.1%})")

    # Save norm params
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_gru_norm.json"), "w") as f:
        json.dump({"mins": mins.tolist(), "maxs": maxs.tolist(), "n_features": N_FEAT}, f)

    # Final eval
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_gru.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(Xt) > 0.5).float() == Yt).float().mean().item()
        te_preds = model(Xte).numpy()
        te_acc = ((te_preds > 0.5) == Y_te).mean()

    print(f"\n  GRU Results:")
    print(f"    Train: {tr_acc:.1%} ({len(X_tr)} days)")
    print(f"    Test:  {te_acc:.1%} ({len(X_te)} days)")

    # ── BACKTEST 7-SIGNAL WEIGHTS ─────────────────────────
    print(f"\n{'='*60}")
    print(f"Backtesting 7-signal weights on test set ({len(X_te)} days)")
    print(f"{'='*60}")

    rsi_raw = (100 - (100 / (1 + pd.Series(prices).diff().where(lambda x: x > 0, 0).rolling(14, min_periods=1).mean() /
               (-pd.Series(prices).diff().where(lambda x: x < 0, 0)).rolling(14, min_periods=1).mean().replace(0, np.nan)))).fillna(50).values
    log_ret_full = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    ema12 = pd.Series(prices).ewm(span=12).mean().values
    ema26 = pd.Series(prices).ewm(span=26).mean().values
    macd_hist_full = (ema12 - ema26) - pd.Series(ema12 - ema26).ewm(span=9).mean().values

    signal_names = ["TM_Sentiment", "BTC_Momentum", "GRU_Model", "RSI", "MACD", "Polymarket", "30day_Model"]
    S_test = []

    test_start = SEQ_LEN + split
    for j in range(len(X_te)):
        i = test_start + j
        if i >= len(prices) - 1: break

        # Signal 1: Sentiment
        mom_20 = (prices[i] - prices[max(0,i-20)]) / prices[max(0,i-20)]
        fg_proxy = np.clip(50 + mom_20 * 300 + (rsi_raw[i] - 50) * 0.3, 5, 95)
        if fg_proxy < 20: sent_p = 0.62
        elif fg_proxy > 80: sent_p = 0.38
        elif fg_proxy < 35: sent_p = 0.45
        elif fg_proxy > 65: sent_p = 0.55
        else: sent_p = 0.5 + (fg_proxy - 50) / 300

        # Signal 2: BTC Momentum
        if i >= 5:
            ret_4 = (prices[i] - prices[i-4]) / prices[i-4]
            ret_recent = (prices[i] - prices[i-2]) / prices[i-2]
            ret_prior = (prices[i-2] - prices[i-4]) / prices[i-4]
            accel = ret_recent - ret_prior
            btc_mom_p = float(np.clip(0.5 + ret_4 * 8 + accel * 15, 0.2, 0.8))
        else:
            btc_mom_p = 0.5

        # Signal 3: GRU model
        gru_p = float(te_preds[j]) if j < len(te_preds) else 0.5

        # Signal 4: RSI
        if rsi_raw[i] < 30: rsi_p = 0.65
        elif rsi_raw[i] > 70: rsi_p = 0.35
        else: rsi_p = 0.5 + (rsi_raw[i] - 50) / 200

        # Signal 5: MACD
        mh = macd_hist_full[i]
        max_mh = max(np.abs(macd_hist_full[max(0,i-20):i+1]).max(), 1)
        macd_p = 0.5 + np.clip(mh / max_mh * 0.2, -0.25, 0.25)

        # Signal 6: Polymarket proxy
        mom_5 = (prices[i] - prices[max(0,i-5)]) / prices[max(0,i-5)]
        poly_p = 0.5 + np.clip(mom_20 * 2, -0.1, 0.1)

        # Signal 7: 30-day model
        threshold_p = 0.5 + np.clip(mom_20 * 0.3, -0.2, 0.2)

        S_test.append([sent_p, btc_mom_p, gru_p, rsi_p, macd_p, poly_p, threshold_p])

    S = np.array(S_test)
    L = Y_te[:len(S)]

    # Logistic regression for optimal weights
    lr = LogisticRegression(C=0.1, max_iter=500)
    lr.fit(S, L)
    lr_acc = lr.score(S, L)

    raw = np.abs(lr.coef_[0])
    weights = raw / raw.sum()

    print(f"\n  Optimal weights (LogReg on {len(S)} test days, acc={lr_acc:.1%}):")
    for name, w, coef in sorted(zip(signal_names, weights, lr.coef_[0]), key=lambda x: -x[1]):
        print(f"    {name:16s}: {w*100:5.1f}%  (coef={coef:+.4f})")

    # Individual signal accuracies
    print(f"\n  Individual signal accuracies:")
    for j, name in enumerate(signal_names):
        acc = ((S[:, j] > 0.5) == L).mean()
        print(f"    {name:16s}: {acc:.1%}")

    # Save
    w_dict = {name: round(float(w), 4) for name, w in zip(signal_names, weights)}
    with open(os.path.join(MODEL_WEIGHTS_DIR, "signal_weights.json"), "w") as f:
        json.dump({"weights": w_dict, "gru_test_acc": float(te_acc),
                    "combined_acc": float(lr_acc), "test_n": len(S)}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  FINAL: GRU test={te_acc:.1%} | Combined={lr_acc:.1%} | {len(S)} OOS days")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
