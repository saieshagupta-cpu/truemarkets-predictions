"""
Train Gradient Boosting model for BTC 3-day direction prediction.
Uses XGBoost (same algorithm family as LightGBM, available on this system).

Data: 5 years of daily OHLCV from CryptoCompare (cached).
Features: 35 engineered features.
Labels: 1 if avg(next 3 days) > today's price, else 0.
Split: 75% train / 25% test (time-series, no shuffle).

Usage: python train/train_lightgbm.py
"""

import json, os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.feature_engineering import build_features, build_labels, FEATURE_NAMES
from app.config import SAVED_DIR

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data", "cache")
HORIZON = 3


def main():
    print("=" * 60)
    print("Gradient Boosting — BTC 3-Day Direction Prediction")
    print(f"35 features, {HORIZON}-day horizon")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    path = os.path.join(CACHE_DIR, "btc_5Y_1d.json")
    if not os.path.exists(path):
        path = os.path.join(CACHE_DIR, "btc_3Y_1d.json")
    with open(path) as f:
        pts = json.load(f)["results"][0]["points"]

    df = pd.DataFrame([{
        "timestamp": p["t"],
        "price": float(p["price"]),
        "open": float(p.get("open", p["price"])),
        "high": float(p.get("high", p["price"])),
        "low": float(p.get("low", p["price"])),
        "volume": float(p.get("volume", 0)),
    } for p in pts])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"  Loaded {len(df)} daily candles")
    print(f"  Date range: {df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}")

    # ── Build features ────────────────────────────────────
    print(f"\n  Building {len(FEATURE_NAMES)} features...")
    feats = build_features(df)
    prices = df["price"].values
    labels = build_labels(prices, horizon=HORIZON)

    warmup = 30
    valid_end = len(labels) - HORIZON
    X = feats[FEATURE_NAMES].values[warmup:valid_end]
    Y = labels[warmup:valid_end]
    dates = df["timestamp"].values[warmup:valid_end]

    print(f"  Usable samples: {len(X)}")
    print(f"  Base rate (up): {Y.mean():.1%}")

    # ── Train/test split ──────────────────────────────────
    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train: {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[split-1]).date()}")
    print(f"  Test:  {pd.Timestamp(dates[split]).date()} to {pd.Timestamp(dates[-1]).date()}")

    # ── Hyperparameter search with CV ─────────────────────
    print(f"\n  Grid search with 5-fold TimeSeriesSplit...")
    tscv = TimeSeriesSplit(n_splits=5)
    best_params, best_score = None, 0

    for n_est in [100, 200, 300]:
        for max_d in [3, 4, 5]:
            for lr in [0.03, 0.05, 0.1]:
                scores = []
                for tr_idx, val_idx in tscv.split(X_train):
                    gbc = GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d, learning_rate=lr,
                        subsample=0.8, min_samples_split=20, min_samples_leaf=10,
                        random_state=42,
                    )
                    gbc.fit(X_train[tr_idx], Y_train[tr_idx])
                    scores.append(gbc.score(X_train[val_idx], Y_train[val_idx]))
                mean_s = np.mean(scores)
                if mean_s > best_score:
                    best_score = mean_s
                    best_params = {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr}

    print(f"  Best CV: {best_score:.1%}")
    print(f"  Best params: {best_params}")

    # ── Train final model ─────────────────────────────────
    print(f"\n  Training final model...")
    model = GradientBoostingClassifier(
        **best_params,
        subsample=0.8, min_samples_split=20, min_samples_leaf=10,
        random_state=42,
    )
    model.fit(X_train, Y_train)

    # ── Evaluate ──────────────────────────────────────────
    test_pred_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(Y_test, test_pred)
    precision = precision_score(Y_test, test_pred, zero_division=0)
    recall = recall_score(Y_test, test_pred, zero_division=0)
    try:
        auc = roc_auc_score(Y_test, test_pred_proba)
    except Exception:
        auc = 0.5
    cm = confusion_matrix(Y_test, test_pred)

    print(f"\n{'='*60}")
    print(f"  TEST RESULTS ({len(X_test)} samples, {HORIZON}-day horizon)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  AUC-ROC:   {auc:.3f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
    print(f"    FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")

    # ── Feature importance ────────────────────────────────
    imp = model.feature_importances_
    imp_dict = {FEATURE_NAMES[i]: float(imp[i]) for i in range(len(FEATURE_NAMES))}
    sorted_imp = sorted(imp_dict.items(), key=lambda x: -x[1])

    print(f"\n  Top 15 Features (gain importance):")
    for name, importance in sorted_imp[:15]:
        bar = "#" * int(importance / sorted_imp[0][1] * 30)
        print(f"    {name:30s} {importance:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────
    os.makedirs(SAVED_DIR, exist_ok=True)

    with open(os.path.join(SAVED_DIR, "lightgbm_btc.pkl"), "wb") as f:
        pickle.dump(model, f)

    results = {
        "model": "GradientBoosting (sklearn)",
        "horizon_days": HORIZON,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "base_rate": float(Y.mean()),
        "cv_mean_accuracy": float(best_score),
        "best_params": best_params,
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
        "top_10_features": [k for k, _ in sorted_imp[:10]],
        "train_period": f"{pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[split-1]).date()}",
        "test_period": f"{pd.Timestamp(dates[split]).date()} to {pd.Timestamp(dates[-1]).date()}",
        "data_source": "CryptoCompare 5-year daily OHLCV (cached btc_5Y_1d.json)",
    }

    with open(os.path.join(SAVED_DIR, "backtest_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Saved: lightgbm_btc.pkl, backtest_results.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
