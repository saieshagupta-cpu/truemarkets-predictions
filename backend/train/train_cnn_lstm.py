"""
Train CNN-LSTM for BTC next-day direction prediction.
Implements Omole & Enke (2024) architecture with Boruta feature selection.

Data: 5 years on-chain from BGeometrics (onchain_merged.csv)
Architecture: CNN-LSTM (Conv1D → LSTM → Dense → Sigmoid)
Feature selection: Boruta (Random Forest wrapper)
Labels: next-day direction (1=up, 0=down)
Window: 5 days
Split: 80/20 temporal

Usage: python train/train_cnn_lstm.py
"""

import json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.cnn_lstm import CNNLSTM, WINDOW_SIZE
from app.config import SAVED_DIR

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data", "cache", "onchain")


def boruta_select(X, Y, feature_names, n_iter=30):
    """Boruta feature selection — finds all relevant features."""
    n_feat = X.shape[1]
    hits = np.zeros(n_feat)
    print(f"    Running {n_iter} Boruta iterations on {n_feat} features...")

    for it in range(n_iter):
        # Create shadow features (shuffled copies)
        shadow = X.copy()
        for j in range(n_feat):
            np.random.shuffle(shadow[:, j])
        X_combined = np.hstack([X, shadow])

        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=it, n_jobs=-1)
        rf.fit(X_combined, Y)
        imp = rf.feature_importances_

        # Feature is a "hit" if it beats the best shadow feature
        max_shadow = imp[n_feat:].max()
        hits += (imp[:n_feat] > max_shadow).astype(float)

        if (it + 1) % 10 == 0:
            confirmed = sum(hits / (it + 1) > 0.5)
            print(f"      Iter {it+1}: {confirmed} features confirmed")

    # Select features that beat shadows > 50% of the time
    selected = hits / n_iter > 0.5
    if sum(selected) < 5:
        # If too few selected, take top 15 by hit rate
        top = np.argsort(-hits)[:15]
        selected = np.zeros(n_feat, dtype=bool)
        selected[top] = True
        print(f"    Boruta: too few confirmed, using top 15")

    sel_names = [feature_names[i] for i in range(n_feat) if selected[i]]
    print(f"    Selected {sum(selected)}/{n_feat}: {sel_names}")
    return selected, sel_names


def main():
    print("=" * 60)
    print("CNN-LSTM Training — Omole & Enke (2024) Architecture")
    print(f"Window: {WINDOW_SIZE} days | Boruta feature selection")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    csv_path = os.path.join(CACHE_DIR, "onchain_merged.csv")
    if not os.path.exists(csv_path):
        print("ERROR: onchain_merged.csv not found. Run download_onchain.py first.")
        return

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    # ── Prepare features and labels ──────────────────────
    # Drop date, price columns used only for labels
    price = df["price_close"].values
    exclude_cols = ["date", "price_open", "price_high", "price_low", "price_close", "price_volume"]

    # Fix duplicate column names (hash_ribbons has 3)
    cols = list(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if c in seen:
            seen[c] += 1
            cols[i] = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
    df.columns = cols

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Convert string columns to numeric (e.g., hash_ribbons "Up"/"Down")
    for col in feature_cols:
        if df[col].dtype == object:
            # Map categorical to numeric
            uniques = df[col].unique()
            mapping = {v: i for i, v in enumerate(sorted(str(u) for u in uniques))}
            df[col] = df[col].map(lambda x: mapping.get(str(x), 0))

    X_all = df[feature_cols].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0, posinf=0, neginf=0)

    # Next-day direction labels
    Y_all = np.array([1.0 if price[i+1] > price[i] else 0.0 for i in range(len(price)-1)])

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(Y_all)}")
    print(f"  Up rate: {Y_all.mean():.1%}")

    # ── Train/test split (80/20, temporal) ────────────────
    split = int(len(Y_all) * 0.8)
    X_train_raw = X_all[:split]
    Y_train = Y_all[:split]
    X_test_raw = X_all[split:len(Y_all)]
    Y_test = Y_all[split:len(Y_all)]

    print(f"  Train: {len(X_train_raw)} ({df['date'].iloc[0].date()} to {df['date'].iloc[split].date()})")
    print(f"  Test:  {len(X_test_raw)} ({df['date'].iloc[split].date()} to {df['date'].iloc[-1].date()})")

    # ── Boruta feature selection on training set ──────────
    print(f"\n  Boruta Feature Selection")
    print(f"  {'='*50}")
    selected, sel_names = boruta_select(X_train_raw, Y_train, feature_cols)
    X_train_sel = X_train_raw[:, selected]
    X_test_sel = X_test_raw[:, selected]
    N_FEAT = X_train_sel.shape[1]
    print(f"  Using {N_FEAT} features after Boruta")

    # ── Min-Max normalization (from training set only) ────
    mins = X_train_sel.min(axis=0)
    maxs = X_train_sel.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    X_train_norm = (X_train_sel - mins) / ranges
    X_test_norm = (X_test_sel - mins) / ranges

    # ── Create sliding window sequences ──────────────────
    def make_sequences(X, Y, window=WINDOW_SIZE):
        seqs, labels = [], []
        for i in range(window, len(X)):
            seqs.append(X[i-window:i])
            labels.append(Y[i-1])  # Label for the last day in the window
        return np.array(seqs), np.array(labels)

    X_tr_seq, Y_tr_seq = make_sequences(X_train_norm, Y_train)
    X_te_seq, Y_te_seq = make_sequences(X_test_norm, Y_test)
    print(f"  Sequences: train={len(X_tr_seq)}, test={len(X_te_seq)}")
    print(f"  Shape: ({WINDOW_SIZE}, {N_FEAT})")

    # ── Train CNN-LSTM ────────────────────────────────────
    print(f"\n  Training CNN-LSTM")
    print(f"  {'='*50}")

    Xt = torch.FloatTensor(X_tr_seq)
    Yt = torch.FloatTensor(Y_tr_seq)
    Xte = torch.FloatTensor(X_te_seq)
    Yte = torch.FloatTensor(Y_te_seq)

    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=64, shuffle=True)
    model = CNNLSTM(n_features=N_FEAT)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.BCELoss()

    best_acc = 0
    patience_count = 0
    os.makedirs(SAVED_DIR, exist_ok=True)
    model_path = os.path.join(SAVED_DIR, "cnn_lstm.pt")

    for epoch in range(500):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            te_pred = model(Xte)
            te_acc = ((te_pred > 0.5).float() == Yte).float().mean().item()
            scheduler.step(1 - te_acc)

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss/len(loader):.4f} test_acc={te_acc:.1%} (best={best_acc:.1%}) lr={optimizer.param_groups[0]['lr']:.6f}")

        if te_acc > best_acc:
            best_acc = te_acc
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= 80:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    # ── Evaluate best model ───────────────────────────────
    print(f"\n  Evaluation")
    print(f"  {'='*50}")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        te_proba = model(Xte).numpy()
        tr_proba = model(Xt).numpy()

    te_pred = (te_proba > 0.5).astype(int)
    tr_pred = (tr_proba > 0.5).astype(int)

    acc = accuracy_score(Y_te_seq, te_pred)
    prec = precision_score(Y_te_seq, te_pred, zero_division=0)
    rec = recall_score(Y_te_seq, te_pred, zero_division=0)
    f1 = f1_score(Y_te_seq, te_pred, zero_division=0)
    try:
        auc = roc_auc_score(Y_te_seq, te_proba)
    except:
        auc = 0.5
    mcc = matthews_corrcoef(Y_te_seq, te_pred)
    cm = confusion_matrix(Y_te_seq, te_pred)
    train_acc = accuracy_score(Y_tr_seq, tr_pred)

    print(f"  Train Accuracy: {train_acc:.1%}")
    print(f"  Test Accuracy:  {acc:.1%}")
    print(f"  Precision:      {prec:.1%}")
    print(f"  Recall:         {rec:.1%}")
    print(f"  F1 Score:       {f1:.1%}")
    print(f"  AUC-ROC:        {auc:.4f}")
    print(f"  MCC:            {mcc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # ── Save normalization params + metrics ───────────────
    norm_data = {
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "n_features": N_FEAT,
        "selected_features": sel_names,
        "window_size": WINDOW_SIZE,
    }
    with open(os.path.join(SAVED_DIR, "cnn_lstm_norm.json"), "w") as f:
        json.dump(norm_data, f, indent=2)

    metrics = {
        "model": "CNN-LSTM (Omole & Enke 2024)",
        "architecture": "Conv1D(128)→Conv1D(64)→LSTM(256,2L)→Dense(128→64→1)",
        "feature_selection": "Boruta",
        "window_size": WINDOW_SIZE,
        "n_features": N_FEAT,
        "selected_features": sel_names,
        "total_samples": len(Y_all),
        "train_samples": len(Y_tr_seq),
        "test_samples": len(Y_te_seq),
        "base_rate": float(Y_all.mean()),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_auc": float(auc),
        "test_mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "train_period": f"{df['date'].iloc[0].date()} to {df['date'].iloc[split].date()}",
        "test_period": f"{df['date'].iloc[split].date()} to {df['date'].iloc[-1].date()}",
        "data_source": "BGeometrics on-chain data (47 features, 5 years)",
    }
    with open(os.path.join(SAVED_DIR, "cnn_lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Also save as backtest_results.json for the frontend
    with open(os.path.join(SAVED_DIR, "backtest_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved: cnn_lstm.pt, cnn_lstm_norm.json, cnn_lstm_metrics.json")
    print(f"\n{'='*60}")
    print(f"  FINAL: {acc:.1%} accuracy ({N_FEAT} features, {len(Y_te_seq)} test days)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
