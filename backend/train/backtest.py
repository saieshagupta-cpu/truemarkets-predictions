"""
Backtest: evaluate trained TCN on test set, generate charts.
Imports features from train_models.py — single source of truth.

Usage: python train/backtest.py
"""

import json, os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_tcn import DirectionTCN
from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH
from app.data.truemarkets_mcp import CACHE_DIR
from train.train_models import build_features, create_consensus_sequences

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest_results")
BG = "#0F0F1A"
TEAL = "#00D4AA"
RED = "#FF6B6B"
YELLOW = "#FFE66D"
MUTED = "#8B8BA3"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("TCN Backtest")
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

    features = build_features(
        df["price"].values, df["timestamp"],
        has_ohlcv=True, opens=df["open"].values,
        highs=df["high"].values, lows=df["low"].values,
        volumes=df["volume"].values,
    )
    prices = df["price"].values
    X, Y, indices = create_consensus_sequences(features, prices, seq_len=30, horizons=[1, 2, 3])

    # Load trained model
    model = DirectionTCN(input_size=10, num_channels=16, num_layers=3, dropout=0.35)
    model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily.pt"), weights_only=True))
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_daily_norm.json")) as f:
        norm = json.load(f)
    means, stds = np.array(norm["means"]), np.array(norm["stds"])
    stds[stds == 0] = 1
    model.eval()

    # Split
    dates = pd.to_datetime(df["timestamp"].values[indices]).tz_localize(None).values
    test_mask = dates >= pd.Timestamp("2025-10-15")
    X_test, Y_test = X[test_mask], Y[test_mask]

    print(f"  Test set: {len(X_test)} consensus sequences (Oct 2025 – Apr 2026)")

    # Predict
    with torch.no_grad():
        preds = model(torch.FloatTensor((X_test - means) / stds)).numpy()

    actual_up = Y_test > 0.5
    pred_up = preds > 0.5
    accuracy = (pred_up == actual_up).mean()
    print(f"  Test accuracy: {accuracy:.1%}")

    # ── Chart: accuracy bar ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    cats = ["Random\nCoin Flip", "TCN Model\n(Daily, OOS)"]
    vals = [50, accuracy * 100]
    colors = ["#3D3D50", TEAL]
    bars = ax.bar(cats, vals, color=colors, width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2,
                f"{v:.0f}%", ha="center", fontsize=28, fontweight="bold", color="#FFFFFF")
    ax.set_ylim(0, 115); ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=14)
    ax.axhline(50, color="#3D3D50", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_accuracy.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved yc_accuracy.png")

    # ── Chart: cumulative returns ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # Actual returns for test period
    test_prices = prices[indices[test_mask]]
    test_returns = np.diff(test_prices) / test_prices[:-1]
    n_ret = min(len(test_returns), len(preds) - 1)
    sig_returns = np.where(preds[:n_ret] > 0.5, test_returns[:n_ret], -test_returns[:n_ret])
    tcn_cum = np.concatenate([[1], np.cumprod(1 + sig_returns)])
    bh_cum = np.concatenate([[1], np.cumprod(1 + test_returns[:n_ret])])

    ax.plot(range(len(tcn_cum)), tcn_cum, color=TEAL, linewidth=2.5, label="TCN Strategy")
    ax.plot(range(len(bh_cum)), bh_cum, color=MUTED, linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax.fill_between(range(len(tcn_cum)), 1, tcn_cum, where=np.array(tcn_cum) >= 1, alpha=0.08, color=TEAL)
    ax.axhline(1.0, color="#3D3D50", linewidth=0.8)
    ax.legend(fontsize=12, loc="upper left", frameon=False, labelcolor=[TEAL, MUTED])
    ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.08, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    final_ret = (tcn_cum[-1] - 1) * 100
    ax.set_title(f"TCN cumulative return: {final_ret:+.1f}%", fontsize=14, color="#FFFFFF", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_returns.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved yc_returns.png")

    # ── Save metrics ──────────────────────────────────────
    metrics = {
        "test_accuracy": float(accuracy),
        "test_n": len(X_test),
        "tcn_cumulative_return": float(tcn_cum[-1] - 1),
        "buyhold_return": float(bh_cum[-1] - 1),
    }
    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved backtest_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
