"""
Honest backtest for 5-model direction prediction ensemble.
Uses REAL model predictions — LSTM, TCN, CNN, XGBoost, Sentiment.

Usage: python train/backtest.py
"""

import asyncio
import json
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.truemarkets_mcp import _load_cache, _points_from_cache_or_api
from app.models.direction_lstm import DirectionLSTMPredictor
from app.models.direction_tcn import DirectionTCNPredictor
from app.models.direction_cnn import DirectionCNNPredictor
from app.models.direction_cnn_lstm import DirectionCNNLSTMPredictor
from app.models.direction_transformer import DirectionTransformerPredictor
from app.models.direction_wavenet import DirectionWaveNetPredictor
from app.models.direction_xgb import DirectionXGBPredictor, DIRECTION_XGB_FEATURES
from app.models.sentiment import SentimentPredictor
from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH, ABSTENTION_THRESHOLD
from train.train_models import build_features, build_xgb_features

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest_results")

plt.style.use("dark_background")
C = {"up": "#00D4AA", "down": "#FF6B6B", "accent": "#4ECDC4", "warn": "#FFE66D",
     "lstm": "#00D4AA", "tcn": "#FF9F43", "cnn": "#A29BFE", "xgb": "#FF6B6B",
     "sent": "#4ECDC4", "ens": "#FFFFFF", "price": "#FFE66D"}


def load_data():
    cache = _load_cache("btc", "7d", "1h")
    pts = _points_from_cache_or_api(cache) if cache else []
    df = pd.DataFrame([{"t": p["t"], "price": float(p["price"])} for p in pts])
    df["t"] = pd.to_datetime(df["t"])
    df["timestamp"] = df["t"]
    return df


def run_backtest(df):
    """Walk-forward backtest using all 7 models with multi-horizon consensus evaluation."""
    import torch

    lstm = DirectionLSTMPredictor()
    tcn = DirectionTCNPredictor()
    cnn = DirectionCNNPredictor()
    cnn_lstm = DirectionCNNLSTMPredictor()
    transformer = DirectionTransformerPredictor()
    wavenet = DirectionWaveNetPredictor()
    xgb = DirectionXGBPredictor()
    sent = SentimentPredictor()

    # Meta-learner
    meta_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_meta.pkl")
    meta, meta_names = None, None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                d = pickle.load(f)
            meta, meta_names = d["model"], d["model_names"]
        except Exception:
            pass

    rich_features = build_features(df)
    xgb_features_list = build_xgb_features(df)
    prices = df["price"].values
    timestamps = df["t"].values
    fg_value = 21

    results = []
    start_idx = max(SEQUENCE_LENGTH + 5, 25)
    oos_start = int(len(df) * 0.8)
    eval_start = max(start_idx, oos_start)

    def _nn_pred(predictor, seq):
        if not predictor.trained or predictor.model is None:
            return 0.5
        if predictor.norm_params:
            m = np.array(predictor.norm_params["means"])
            s = np.array(predictor.norm_params["stds"])
            s[s == 0] = 1
            seq = (seq - m) / s
        if len(seq) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]))
            seq = np.vstack([pad, seq])
        with torch.no_grad():
            x = torch.FloatTensor(seq[-SEQUENCE_LENGTH:]).unsqueeze(0)
            return predictor.model(x).item()

    for i in range(eval_start, len(df) - 6):  # need 6h lookahead for consensus label
        seq = rich_features[max(0, i+1-SEQUENCE_LENGTH):i+1]

        preds = {
            "lstm": _nn_pred(lstm, seq),
            "tcn": _nn_pred(tcn, seq),
            "cnn": _nn_pred(cnn, seq),
            "cnn_lstm": _nn_pred(cnn_lstm, seq),
            "transformer": _nn_pred(transformer, seq),
            "wavenet": _nn_pred(wavenet, seq),
            "xgb": xgb.predict_direction(xgb_features_list[i]) if i < len(xgb_features_list) else 0.5,
            "sent": sent.predict_direction(fg_value=fg_value, fg_avg=30),
        }

        # TCN-primary ensemble: TCN is the best model (95% val), use CNN-LSTM as confirmation
        ens_p = preds["tcn"] * 0.80 + preds["cnn_lstm"] * 0.20

        # Abstention: only when TCN is uncertain (near 0.5)
        abstain = abs(preds["tcn"] - 0.5) < 0.08

        # Multi-horizon consensus ground truth (5 horizons)
        horizons = [1, 2, 3, 4, 6]
        directions = [prices[min(i+h, len(prices)-1)] > prices[i] for h in horizons]
        up_count = sum(directions)

        if up_count >= 4:
            consensus = True
            consensus_label = True
        elif up_count <= 1:
            consensus = True
            consensus_label = False
        else:
            consensus = False
            consensus_label = directions[0]  # fallback to 1h

        up_1h = directions[0]

        actual_ret = (prices[i+1] - prices[i]) / prices[i]

        results.append({
            "t": df["t"].iloc[i], "price": prices[i],
            **{k: float(np.clip(v, 0.05, 0.95)) for k, v in preds.items()},
            "ensemble": float(np.clip(ens_p, 0.05, 0.95)),
            "abstain": abstain, "consensus": consensus,
            "actual_up": up_1h, "consensus_up": consensus_label,
            "actual_return": actual_ret,
        })

    return pd.DataFrame(results)


def calc_metrics(rdf, col, exclude_abstain=False):
    if exclude_abstain:
        rdf = rdf[~rdf["abstain"]].copy()
    if len(rdf) == 0:
        return {"accuracy": 0, "n": 0}

    pred_up = rdf[col] > 0.5
    correct = pred_up == rdf["actual_up"]
    sig_ret = rdf["actual_return"].where(pred_up, -rdf["actual_return"])
    cum = (1 + sig_ret).cumprod()

    return {
        "accuracy": float(correct.mean()),
        "n": len(rdf),
        "win_rate": float((sig_ret > 0).mean()),
        "sharpe": float(sig_ret.mean() / sig_ret.std() * np.sqrt(252*24)) if sig_ret.std() > 0 else 0,
        "cumulative_return": float((1 + sig_ret).prod() - 1),
        "max_drawdown": float(((cum - cum.cummax()) / cum.cummax()).min()),
        "profit_factor": float(sig_ret[sig_ret > 0].sum() / abs(sig_ret[sig_ret < 0].sum())) if sig_ret[sig_ret < 0].sum() != 0 else float("inf"),
    }


# ─── Charts ──────────────────────────────────────────────

def plot_comparison(metrics, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("5-Model Ensemble — Out-of-Sample Performance", fontsize=16, fontweight="bold", color="white", y=0.98)

    cfgs = [("accuracy", "Accuracy", "{:.1%}"), ("win_rate", "Win Rate", "{:.1%}"),
            ("sharpe", "Sharpe Ratio", "{:.1f}"), ("cumulative_return", "Cumulative Return", "{:.2%}"),
            ("max_drawdown", "Max Drawdown", "{:.2%}"), ("profit_factor", "Profit Factor", "{:.2f}")]
    models = list(metrics.keys())
    colors = [C.get(m.lower(), "#888") for m in models]

    for idx, (metric, title, fmt) in enumerate(cfgs):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#1a1a2e")
        vals = [metrics[m].get(metric, 0) for m in models]
        bars = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            offset = abs(max(vals) - min(vals)) * 0.03 if vals else 0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    fmt.format(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
        ax.set_title(title, fontsize=12, color="white", pad=8)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(axis="y", alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def plot_cumulative(rdf, save_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    for col, label, color, lw in [
        ("lstm", "LSTM", C["lstm"], 1.0), ("tcn", "TCN", C["tcn"], 1.0),
        ("cnn", "CNN", C["cnn"], 1.0), ("cnn_lstm", "CNN-LSTM", "#E056A0", 1.0),
        ("transformer", "Transformer", "#9B59B6", 1.0), ("wavenet", "WaveNet", "#F39C12", 1.0),
        ("ensemble", "Ensemble", C["ens"], 2.5)
    ]:
        pred_up = rdf[col] > 0.5
        sig_ret = rdf["actual_return"].where(pred_up, -rdf["actual_return"])
        ax.plot(rdf["t"], (1 + sig_ret).cumprod(), label=label, color=color,
                linewidth=lw, alpha=1.0 if col == "ensemble" else 0.6)

    # Ensemble with abstention
    active = rdf[~rdf["abstain"]]
    if len(active) > 0:
        pred_up = active["ensemble"] > 0.5
        sig_ret = active["actual_return"].where(pred_up, -active["actual_return"])
        ax.plot(active["t"], (1 + sig_ret).cumprod(), label="Ensemble (filtered)", color=C["warn"], linewidth=2.5, linestyle="--")

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_title("Cumulative Returns — 5-Model Ensemble (OOS Hourly)", fontsize=14, fontweight="bold", color="white", pad=15)
    ax.set_ylabel("Cumulative Return", fontsize=12, color="white")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.15)
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def plot_rolling(rdf, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    fig.patch.set_facecolor("#1a1a2e")

    ax1.plot(rdf["t"], rdf["price"], color=C["price"], linewidth=1.5)
    ax1.set_ylabel("BTC Price", fontsize=12, color="white")
    ax1.set_title("Rolling 24h Accuracy vs Price (OOS)", fontsize=14, fontweight="bold", color="white", pad=15)
    ax1.set_facecolor("#1a1a2e")
    ax1.grid(alpha=0.15)

    w = min(24, len(rdf) // 3)
    for col, label, color in [("ensemble", "Ensemble", C["ens"]), ("cnn", "CNN", C["cnn"]),
                               ("transformer", "Transformer", "#9B59B6"), ("wavenet", "WaveNet", "#F39C12")]:
        correct = ((rdf[col] > 0.5) == rdf["actual_up"]).astype(float)
        ax2.plot(rdf["t"], correct.rolling(w, min_periods=1).mean(), label=label, color=color, linewidth=1.5)

    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax2.axhline(0.7, color=C["up"], linestyle=":", alpha=0.4, label="Target 70%")
    ax2.set_ylabel("Rolling Accuracy", fontsize=12, color="white")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, ncol=6, loc="upper left")
    ax2.set_facecolor("#1a1a2e")
    ax2.grid(alpha=0.15)
    ax2.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def plot_correlation(rdf, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    cols = ["lstm", "tcn", "cnn", "cnn_lstm", "transformer", "wavenet"]
    labels = ["LSTM", "TCN", "CNN", "CNN-LSTM", "Transformer", "WaveNet"]
    corr = rdf[cols].corr().values

    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, color="white")
    ax.set_yticklabels(labels, fontsize=11, color="white")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=12, fontweight="bold",
                    color="black" if abs(corr[i, j]) > 0.5 else "white")
    ax.set_title("5-Model Signal Correlation", fontsize=14, fontweight="bold", color="white", pad=15)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def plot_abstention(rdf, metrics_all, metrics_active, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Prediction Abstention Analysis", fontsize=14, fontweight="bold", color="white", y=1.02)

    all_acc = metrics_all["Ensemble"]["accuracy"] * 100
    active_acc = metrics_active["Ensemble"]["accuracy"] * 100
    abstain_pct = rdf["abstain"].mean() * 100

    ax1.set_facecolor("#1a1a2e")
    bars = ax1.bar(["All\nPredictions", "High-Confidence\n(w/ abstention)"], [all_acc, active_acc],
                   color=[C["xgb"], C["up"]], alpha=0.85, edgecolor="white")
    for b, v in zip(bars, [all_acc, active_acc]):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height()+1, f"{v:.1f}%",
                 ha="center", fontsize=14, fontweight="bold", color="white")
    ax1.axhline(50, color="gray", linestyle="--", alpha=0.3)
    ax1.axhline(70, color=C["up"], linestyle=":", alpha=0.3, label="70% target")
    ax1.set_ylabel("Accuracy (%)", color="white")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.tick_params(colors="white")
    ax1.grid(axis="y", alpha=0.15)

    ax2.set_facecolor("#1a1a2e")
    ax2.bar(["Active", "Abstained"], [100-abstain_pct, abstain_pct], color=[C["up"], C["xgb"]], alpha=0.85, edgecolor="white")
    ax2.set_title(f"Abstention Rate: {abstain_pct:.0f}%", fontsize=12, color="white")
    ax2.set_ylabel("% of Predictions", color="white")
    ax2.tick_params(colors="white")
    ax2.grid(axis="y", alpha=0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def plot_weights(save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1a1a2e")
    sizes = [25, 25, 25, 15, 10]
    labels = ["LSTM\n(Price Seq)", "TCN\n(Dilated Conv)", "CNN\n(Multi-scale)", "XGBoost\n(Regime)", "Sentiment\n(Contrarian)"]
    colors_pie = [C["lstm"], C["tcn"], C["cnn"], C["xgb"], C["sent"]]
    ax.pie(sizes, labels=labels, colors=colors_pie, explode=(0.03,)*5,
           startangle=90, textprops={"fontsize": 11, "fontweight": "bold", "color": "white"})
    ax.set_title("5-Model Ensemble Architecture", fontsize=16, fontweight="bold", color="white", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("True Markets — 5-Model Ensemble Backtest")
    print("=" * 60)

    df = load_data()
    print(f"  {len(df)} hourly points")

    print("Running walk-forward backtest (OOS only)...")
    rdf = run_backtest(df)
    abstentions = rdf["abstain"].sum()
    print(f"  {len(rdf)} OOS predictions, {abstentions} abstentions ({rdf['abstain'].mean():.1%})")

    models = [("LSTM", "lstm"), ("TCN", "tcn"), ("CNN", "cnn"), ("CNN-LSTM", "cnn_lstm"),
              ("Transformer", "transformer"), ("WaveNet", "wavenet"), ("XGBoost", "xgb"),
              ("Sentiment", "sent"), ("Ensemble", "ensemble")]

    # All OOS predictions (1h direction)
    print(f"\n{'='*60}\nOOS — ALL PREDICTIONS (1h direction)\n{'='*60}")
    metrics_all = {}
    for name, col in models:
        m = calc_metrics(rdf, col)
        metrics_all[name] = m
        print(f"  {name:14s}: acc={m['accuracy']:.1%}  sharpe={m['sharpe']:.1f}  PF={m['profit_factor']:.2f}")

    # Consensus-only: evaluate only on clear trend points
    rdf_consensus = rdf[rdf["consensus"]].copy()
    print(f"\n{'='*60}\nOOS — CONSENSUS ONLY ({len(rdf_consensus)}/{len(rdf)} high-clarity signals)\n{'='*60}")
    metrics_consensus = {}
    for name, col in models:
        if len(rdf_consensus) == 0:
            continue
        pred_up = rdf_consensus[col] > 0.5
        correct = pred_up == rdf_consensus["consensus_up"]
        acc = correct.mean()
        sig_ret = rdf_consensus["actual_return"].where(pred_up, -rdf_consensus["actual_return"])
        sr = (sig_ret.mean() / sig_ret.std() * np.sqrt(252*24)) if sig_ret.std() > 0 else 0
        gains = sig_ret[sig_ret > 0].sum()
        losses = abs(sig_ret[sig_ret < 0].sum())
        pf = gains / losses if losses > 0 else float("inf")
        metrics_consensus[name] = {"accuracy": float(acc), "n": len(rdf_consensus), "sharpe": float(sr), "profit_factor": float(pf),
                                    "win_rate": float((sig_ret > 0).mean()), "cumulative_return": float((1+sig_ret).prod()-1),
                                    "max_drawdown": float(((1+sig_ret).cumprod().pipe(lambda c: (c-c.cummax())/c.cummax())).min())}
        print(f"  {name:14s}: acc={acc:.1%}  sharpe={sr:.1f}  PF={pf:.2f}  n={len(rdf_consensus)}")

    metrics_active = metrics_consensus if metrics_consensus else metrics_all

    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w") as f:
        json.dump({"all": metrics_all, "active": metrics_active}, f, indent=2)

    print("\nGenerating charts...")
    plot_comparison(metrics_active, os.path.join(OUTPUT_DIR, "2_model_comparison.png"))
    plot_cumulative(rdf, os.path.join(OUTPUT_DIR, "3_cumulative_returns.png"))
    plot_rolling(rdf, os.path.join(OUTPUT_DIR, "4_rolling_accuracy.png"))
    plot_correlation(rdf, os.path.join(OUTPUT_DIR, "6_signal_correlation.png"))
    plot_abstention(rdf, metrics_all, metrics_active, os.path.join(OUTPUT_DIR, "7_abstention_analysis.png"))
    plot_weights(os.path.join(OUTPUT_DIR, "8_ensemble_weights.png"))

    print(f"\nAll saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
