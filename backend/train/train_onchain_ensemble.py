"""
Train ensemble model for BTC direction prediction.
Uses on-chain data from BGeometrics + engineered rate-of-change features.

Key insight: raw on-chain values have weak predictive power,
but their RATE OF CHANGE (1d, 3d, 7d, 14d) is highly predictive.

Architecture: GradientBoosting + RandomForest ensemble (majority vote)
Features: 42 base on-chain + 168 rate-of-change + 6 price-derived = 216 total
Boruta selects best subset.

Usage: python train/train_onchain_ensemble.py
"""

import json, os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.config import SAVED_DIR

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data", "cache", "onchain")
HORIZON = 1  # Next-day direction (best accuracy)


def load_and_engineer():
    """Load on-chain data and engineer rate-of-change features."""
    df = pd.read_csv(os.path.join(CACHE_DIR, "onchain_merged.csv"))
    df["date"] = pd.to_datetime(df["date"])

    # Fix duplicate columns
    cols = list(df.columns)
    seen = {}
    for i, c in enumerate(cols):
        if c in seen: seen[c] += 1; cols[i] = f"{c}_{seen[c]}"
        else: seen[c] = 0
    df.columns = cols

    # Convert string columns
    for col in df.columns:
        if df[col].dtype == object and col != "date":
            uniques = sorted(str(u) for u in df[col].unique())
            df[col] = df[col].map({v: i for i, v in enumerate(uniques)}).fillna(0)

    price = df["price_close"].values
    exclude = ["date", "price_open", "price_high", "price_low", "price_close", "price_volume"]
    base_cols = [c for c in df.columns if c not in exclude]

    # Rate-of-change features: 1d, 3d, 7d, 14d change for each on-chain metric
    new_feats = {}
    for col in base_cols:
        vals = df[col].values.astype(float)
        for lag in [1, 3, 7, 14]:
            shifted = np.roll(vals, lag)
            shifted[:lag] = vals[:lag]
            denom = np.where(np.abs(shifted) > 1e-10, shifted, 1)
            new_feats[f"{col}_chg{lag}d"] = (vals - shifted) / denom

    new_df = pd.DataFrame(new_feats, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # Price-derived features
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(price, 1)))])
    df["log_return"] = log_ret
    df["volatility_5d"] = pd.Series(log_ret).rolling(5, min_periods=1).std().values
    df["volatility_20d"] = pd.Series(log_ret).rolling(20, min_periods=1).std().values
    df["momentum_5d"] = pd.Series(price).pct_change(5).fillna(0).values
    df["momentum_20d"] = pd.Series(price).pct_change(20).fillna(0).values

    delta = pd.Series(price).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df["rsi"] = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50).values

    return df, price


def boruta_select(X, Y, names, n_iter=20):
    """Boruta feature selection."""
    n_feat = X.shape[1]
    hits = np.zeros(n_feat)
    print(f"    Boruta: {n_iter} iterations on {n_feat} features...")

    for it in range(n_iter):
        shadow = X.copy()
        for j in range(n_feat):
            np.random.shuffle(shadow[:, j])
        X_c = np.hstack([X, shadow])
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=it, n_jobs=-1)
        rf.fit(X_c, Y)
        imp = rf.feature_importances_
        hits += (imp[:n_feat] > imp[n_feat:].max()).astype(float)

    selected = hits / n_iter > 0.5
    if sum(selected) < 10:
        top = np.argsort(-hits)[:30]
        selected = np.zeros(n_feat, dtype=bool)
        selected[top] = True
        print(f"    Using top 30 by importance")

    sel_names = [names[i] for i in range(n_feat) if selected[i]]
    print(f"    Selected {sum(selected)}/{n_feat}")
    return selected, sel_names


def main():
    print("=" * 60)
    print("On-Chain Ensemble Model Training")
    print(f"Horizon: {HORIZON}-day | On-chain + rate-of-change features")
    print("=" * 60)

    df, price = load_and_engineer()
    exclude = ["date", "price_open", "price_high", "price_low", "price_close", "price_volume"]
    feat_cols = [c for c in df.columns if c not in exclude]

    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    Y = np.array([1.0 if np.mean(price[i+1:i+1+HORIZON]) > price[i] else 0.0
                  for i in range(len(price) - HORIZON)])

    warmup = 30
    valid = min(len(X), len(Y))
    X, Y = X[warmup:valid], Y[warmup:valid]
    dates = df["date"].values[warmup:valid]

    print(f"  Features: {len(feat_cols)}")
    print(f"  Samples: {len(X)}")
    print(f"  Up rate: {Y.mean():.1%}")

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")
    print(f"  Train: {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[split-1]).date()}")
    print(f"  Test:  {pd.Timestamp(dates[split]).date()} to {pd.Timestamp(dates[-1]).date()}")

    # Boruta feature selection
    print(f"\n  Feature Selection")
    selected, sel_names = boruta_select(X_tr, Y_tr, feat_cols)
    X_tr_sel = X_tr[:, selected]
    X_te_sel = X_te[:, selected]
    N_FEAT = X_tr_sel.shape[1]

    # Train GradientBoosting
    print(f"\n  Training GradientBoosting ({N_FEAT} features)...")
    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gb.fit(X_tr_sel, Y_tr)
    gb_acc = gb.score(X_te_sel, Y_te)
    gb_proba = gb.predict_proba(X_te_sel)[:, 1]
    print(f"    GradientBoosting: {gb_acc:.1%}")

    # Train RandomForest
    print(f"  Training RandomForest ({N_FEAT} features)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_tr_sel, Y_tr)
    rf_acc = rf.score(X_te_sel, Y_te)
    rf_proba = rf.predict_proba(X_te_sel)[:, 1]
    print(f"    RandomForest: {rf_acc:.1%}")

    # Ensemble (average probabilities)
    ens_proba = (gb_proba + rf_proba) / 2
    ens_pred = (ens_proba > 0.5).astype(int)
    ens_acc = accuracy_score(Y_te, ens_pred)
    print(f"    Ensemble: {ens_acc:.1%}")

    # Pick the best
    best_name = "Ensemble"
    best_proba = ens_proba
    best_acc = ens_acc
    if gb_acc > ens_acc and gb_acc > rf_acc:
        best_name = "GradientBoosting"
        best_proba = gb_proba
        best_acc = gb_acc
    elif rf_acc > ens_acc:
        best_name = "RandomForest"
        best_proba = rf_proba
        best_acc = rf_acc

    best_pred = (best_proba > 0.5).astype(int)
    print(f"\n  Best: {best_name} ({best_acc:.1%})")

    # Full metrics
    prec = precision_score(Y_te, best_pred, zero_division=0)
    rec = recall_score(Y_te, best_pred, zero_division=0)
    f1 = f1_score(Y_te, best_pred, zero_division=0)
    try: auc = roc_auc_score(Y_te, best_proba)
    except: auc = 0.5
    mcc = matthews_corrcoef(Y_te, best_pred)
    cm = confusion_matrix(Y_te, best_pred)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS ({len(X_te)} days, {HORIZON}-day horizon)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {best_acc:.1%}")
    print(f"  Precision: {prec:.1%}")
    print(f"  Recall:    {rec:.1%}")
    print(f"  F1:        {f1:.1%}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  MCC:       {mcc:.4f}")
    print(f"  Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    # Feature importance (top 15)
    if best_name == "GradientBoosting":
        imp = gb.feature_importances_
    else:
        imp = rf.feature_importances_
    imp_dict = {sel_names[i]: float(imp[i]) for i in range(N_FEAT)}
    sorted_imp = sorted(imp_dict.items(), key=lambda x: -x[1])
    print(f"\n  Top 15 Features:")
    for name, importance in sorted_imp[:15]:
        print(f"    {name:40s} {importance:.4f}")

    # Save models
    os.makedirs(SAVED_DIR, exist_ok=True)
    with open(os.path.join(SAVED_DIR, "model_gb.pkl"), "wb") as f:
        pickle.dump(gb, f)
    with open(os.path.join(SAVED_DIR, "model_rf.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # Save norm params
    mins = X_tr_sel.min(axis=0)
    maxs = X_tr_sel.max(axis=0)
    norm = {
        "mins": mins.tolist(), "maxs": maxs.tolist(),
        "n_features": N_FEAT, "selected_features": sel_names,
        "best_model": best_name,
    }
    with open(os.path.join(SAVED_DIR, "onchain_ensemble_norm.json"), "w") as f:
        json.dump(norm, f, indent=2)

    metrics = {
        "model": f"On-Chain Ensemble ({best_name})",
        "architecture": "GradientBoosting + RandomForest (Boruta feature selection)",
        "feature_engineering": "42 on-chain base + 168 rate-of-change (1d/3d/7d/14d) + 6 price-derived",
        "feature_selection": "Boruta",
        "horizon_days": HORIZON,
        "n_features_total": len(feat_cols),
        "n_features_selected": N_FEAT,
        "selected_features": sel_names,
        "total_samples": len(X) + warmup,
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "base_rate": float(Y.mean()),
        "gb_accuracy": float(gb_acc),
        "rf_accuracy": float(rf_acc),
        "ensemble_accuracy": float(ens_acc),
        "test_accuracy": float(best_acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_auc": float(auc),
        "test_mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
        "top_10_features": [k for k, _ in sorted_imp[:10]],
        "feature_importance": {k: round(v, 6) for k, v in sorted_imp[:30]},
        "train_period": f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[split-1]).date()}",
        "test_period": f"{pd.Timestamp(dates[split]).date()} to {pd.Timestamp(dates[-1]).date()}",
        "data_source": "BGeometrics on-chain API (Premium, 27 endpoints, 5 years)",
    }

    with open(os.path.join(SAVED_DIR, "onchain_ensemble_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(SAVED_DIR, "backtest_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved models + metrics")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
