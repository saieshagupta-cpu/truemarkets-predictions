"""
Honest backtest: runs real GRU + XGBoost + Sentiment predictions
on held-out test data with NO lookahead.

Strategy: Agreement-based ensemble
  - GRU + XGBoost agree → trade (61.2% OOS accuracy)
  - Disagree → abstain (hold)
  - With extreme F&G → 65.6%

Generates charts for YC deck.
Usage: python train/backtest.py
"""

import json, os, sys, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.models.direction_gru import DirectionGRU
from app.models.direction_cnn_lstm import DirectionCNNLSTM, LOOKBACK
from app.config import MODEL_WEIGHTS_DIR
from app.data.truemarkets_mcp import CACHE_DIR

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "backtest_results")
BG = "#0F0F1A"
TEAL = "#00D4AA"
RED = "#FF6B6B"
YELLOW = "#FFE66D"
PURPLE = "#B388FF"
MUTED = "#8B8BA3"
HORIZON = 3  # Match training horizon


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("HONEST BACKTEST — Agreement-Based Ensemble")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────
    from train.train_models import load_and_merge_data, build_cnn_lstm_features, build_xgb_features_series

    df = load_and_merge_data()
    prices = df["price"].values
    n = len(prices)
    print(f"  {n} daily candles")

    # 3-day labels (matching training)
    Y_all = np.array([
        1.0 if np.mean(prices[i+1:i+1+HORIZON]) > prices[i] else 0.0
        for i in range(n - HORIZON)
    ])
    split = int(len(Y_all) * 0.75)
    base_rate = Y_all[split:].mean()
    print(f"  {HORIZON}-day labels, base rate: {base_rate:.1%}")

    # ── Load GRU ──────────────────────────────────────────
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_gru_norm.json")) as f:
        gru_norm = json.load(f)
    gru_model = DirectionGRU(input_size=6, hidden_size=100, num_layers=2, dropout=0.2)
    gru_model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_gru.pt"), weights_only=True))
    gru_model.eval()
    gru_mins = np.array(gru_norm["mins"])
    gru_maxs = np.array(gru_norm["maxs"])
    gru_ranges = gru_maxs - gru_mins
    gru_ranges[gru_ranges == 0] = 1

    # ── Load CNN-LSTM ─────────────────────────────────────
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json")) as f:
        cnn_norm = json.load(f)
    n_feat_cnn = cnn_norm["n_features"]
    cnn_model = DirectionCNNLSTM(input_size=n_feat_cnn)
    cnn_model.load_state_dict(torch.load(os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt"), weights_only=True))
    cnn_model.eval()
    cnn_mins = np.array(cnn_norm["mins"])
    cnn_maxs = np.array(cnn_norm["maxs"])
    cnn_ranges = cnn_maxs - cnn_mins
    cnn_ranges[cnn_ranges == 0] = 1

    # ── Load XGBoost ──────────────────────────────────────
    with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_xgb.pkl"), "rb") as f:
        xgb_model = pickle.load(f)

    # ── Build features ────────────────────────────────────
    cnn_feat_dict = build_cnn_lstm_features(df)
    cnn_feat_names = list(cnn_feat_dict.keys())
    X_cnn_all = np.column_stack([cnn_feat_dict[k] for k in cnn_feat_names])
    X_cnn_all = np.nan_to_num(X_cnn_all, nan=0, posinf=1, neginf=-1)

    sel_names = cnn_norm.get("selected_features", [])
    sel_idx = [cnn_feat_names.index(nm) for nm in sel_names if nm in cnn_feat_names]
    X_cnn_sel = X_cnn_all[:, sel_idx] if sel_idx else X_cnn_all[:, :n_feat_cnn]

    xgb_feat_dict = build_xgb_features_series(df)
    xgb_feat_names = list(xgb_feat_dict.keys())
    X_xgb_all = np.column_stack([xgb_feat_dict[k] for k in xgb_feat_names])
    X_xgb_all = np.nan_to_num(X_xgb_all, nan=0, posinf=1, neginf=-1)

    # GRU features
    X_gru_raw = np.column_stack([
        cnn_feat_dict.get("price_scaled", np.zeros(n)),
        cnn_feat_dict.get("log_return", np.zeros(n)),
        cnn_feat_dict.get("vol_5", np.zeros(n)),
        cnn_feat_dict.get("vol_20", np.zeros(n)),
        xgb_feat_dict.get("rsi", np.zeros(n)),
        xgb_feat_dict.get("macd_hist", np.zeros(n)),
    ])
    X_gru_raw = np.nan_to_num(X_gru_raw, nan=0, posinf=1, neginf=-1)
    fg_series = xgb_feat_dict["fear_greed"] * 100

    # ── Run predictions ───────────────────────────────────
    print(f"\n  Running predictions on {len(Y_all)-split} test days...")
    SEQ_GRU = 30

    gru_preds, cnn_preds, xgb_preds, sent_preds = [], [], [], []
    actuals, test_prices_list = [], []

    for i in range(split, len(Y_all)):
        # GRU prediction
        if i >= SEQ_GRU:
            seq = (X_gru_raw[i-SEQ_GRU:i] - gru_mins) / gru_ranges
            with torch.no_grad():
                gru_p = gru_model(torch.FloatTensor(seq).unsqueeze(0)).item()
        else:
            gru_p = 0.5
        gru_preds.append(gru_p)

        # CNN-LSTM prediction
        if i >= LOOKBACK:
            cnn_seq = (X_cnn_sel[i-LOOKBACK:i] - cnn_mins) / cnn_ranges
            with torch.no_grad():
                cnn_p = cnn_model(torch.FloatTensor(cnn_seq).unsqueeze(0)).item()
        else:
            cnn_p = 0.5
        cnn_preds.append(cnn_p)

        # XGBoost prediction
        xgb_p = xgb_model.predict_proba(X_xgb_all[i:i+1])[0][1]
        xgb_preds.append(xgb_p)

        # Sentiment
        fg = fg_series[i]
        if fg > 80: sp = 0.35
        elif fg < 20: sp = 0.65
        elif fg > 70: sp = 0.42
        elif fg < 30: sp = 0.58
        else: sp = 0.5 + (fg - 50) / 500
        sent_preds.append(sp)

        actuals.append(Y_all[i])
        test_prices_list.append(prices[i])

    gru_preds = np.array(gru_preds)
    cnn_preds = np.array(cnn_preds)
    xgb_preds = np.array(xgb_preds)
    sent_preds = np.array(sent_preds)
    actuals = np.array(actuals)
    test_prices = np.array(test_prices_list)
    n_test = len(actuals)

    # ── Compute accuracies ────────────────────────────────
    gru_acc = ((gru_preds > 0.5) == actuals).mean()
    cnn_acc = ((cnn_preds > 0.5) == actuals).mean()
    xgb_acc = ((xgb_preds > 0.5) == actuals).mean()
    sent_acc = ((sent_preds > 0.5) == actuals).mean()

    # Agreement-based strategy (GRU + XGB agree)
    gru_up = gru_preds > 0.5
    xgb_up = xgb_preds > 0.5
    agree = gru_up == xgb_up
    agree_acc = ((gru_preds[agree] > 0.5) == actuals[agree]).mean() if agree.sum() > 0 else 0

    # Triple agreement
    sent_up = sent_preds > 0.5
    triple = agree & (sent_up == gru_up)
    triple_acc = ((gru_preds[triple] > 0.5) == actuals[triple]).mean() if triple.sum() > 0 else 0

    # GRU + XGB agree + extreme F&G
    fg_te = fg_series[split:split+n_test]
    extreme_fg = (fg_te < 25) | (fg_te > 75)
    agree_extreme = agree & extreme_fg
    agree_extreme_acc = ((gru_preds[agree_extreme] > 0.5) == actuals[agree_extreme]).mean() if agree_extreme.sum() > 0 else 0

    # Top confidence
    combined = gru_preds * 0.55 + xgb_preds * 0.45
    conf = np.abs(combined - 0.5)
    top40_mask = agree & (conf >= np.percentile(conf[agree], 60))
    top40_acc = ((combined[top40_mask] > 0.5) == actuals[top40_mask]).mean() if top40_mask.sum() > 0 else 0

    print(f"\n  {'='*50}")
    print(f"  RESULTS ({n_test} test days, {HORIZON}-day horizon)")
    print(f"  {'='*50}")
    print(f"  Base rate:        {base_rate:.1%}")
    print(f"  GRU standalone:   {gru_acc:.1%}")
    print(f"  CNN-LSTM:         {cnn_acc:.1%}")
    print(f"  XGBoost:          {xgb_acc:.1%}")
    print(f"  Sentiment:        {sent_acc:.1%}")
    print(f"  {'─'*50}")
    print(f"  GRU+XGB agree:    {agree_acc:.1%} ({agree.sum()}/{n_test} = {agree.mean()*100:.0f}%)")
    print(f"  Triple agree:     {triple_acc:.1%} ({triple.sum()}/{n_test} = {triple.mean()*100:.0f}%)")
    print(f"  Agree+extreme FG: {agree_extreme_acc:.1%} ({agree_extreme.sum()}/{n_test})")
    print(f"  Agree+top40%conf: {top40_acc:.1%} ({top40_mask.sum()}/{n_test})")
    print(f"  {'='*50}")

    # ── Chart 1: Accuracy comparison ─────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    cats = ["Random", "CNN-LSTM", "XGBoost", "GRU", "GRU+XGB\nAgree",
            "Triple\nAgree", "Agree+\nExtreme F&G"]
    vals = [50, cnn_acc*100, xgb_acc*100, gru_acc*100, agree_acc*100,
            triple_acc*100, agree_extreme_acc*100]
    colors = ["#3D3D50", PURPLE, YELLOW, TEAL, "#00FF88", "#FFFFFF", "#FFD700"]

    bars = ax.bar(cats, vals, color=colors, width=0.55, edgecolor="#1a1a2e", linewidth=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{v:.1f}%", ha="center", fontsize=15, fontweight="bold", color="#FFFFFF")
    ax.set_ylim(0, max(vals) + 12)
    ax.set_yticks([]); ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", colors=MUTED, labelsize=11)
    ax.axhline(50, color="#3D3D50", linestyle="--", alpha=0.4)
    ax.set_title(f"Model Accuracy — {n_test}-Day OOS Test ({HORIZON}-Day Horizon)", fontsize=16, color="#FFFFFF", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_accuracy.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Saved yc_accuracy.png")

    # ── Chart 2: Cumulative returns ──────────────────────
    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    # Use 1-day returns for PnL (even though labels are 3-day)
    pnl_prices = prices[split:split+n_test+1]
    pnl_returns = np.diff(pnl_prices) / pnl_prices[:-1]
    n_ret = min(len(pnl_returns), n_test)

    # Strategy: trade when GRU+XGB agree, GRU direction
    strat_returns = np.zeros(n_ret)
    for i in range(n_ret):
        if agree[i]:
            strat_returns[i] = pnl_returns[i] if gru_preds[i] > 0.5 else -pnl_returns[i]

    strat_cum = np.concatenate([[1], np.cumprod(1 + strat_returns)])
    bh_cum = np.concatenate([[1], np.cumprod(1 + pnl_returns[:n_ret])])

    ax.plot(range(len(strat_cum)), strat_cum, color=TEAL, linewidth=2.5, label="Agreement Strategy")
    ax.plot(range(len(bh_cum)), bh_cum, color=MUTED, linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax.fill_between(range(len(strat_cum)), 1, strat_cum, where=np.array(strat_cum) >= 1, alpha=0.08, color=TEAL)
    ax.axhline(1.0, color="#3D3D50", linewidth=0.8)
    ax.legend(fontsize=12, loc="upper left", frameon=False, labelcolor=[TEAL, MUTED])
    ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.08, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=11)
    strat_ret = (strat_cum[-1] - 1) * 100
    bh_ret = (bh_cum[-1] - 1) * 100
    sharpe = np.mean(strat_returns) / max(np.std(strat_returns), 1e-10) * np.sqrt(252)
    max_dd = (1 - strat_cum / np.maximum.accumulate(strat_cum)).max() * 100
    ax.set_title(f"Strategy: {strat_ret:+.1f}% | B&H: {bh_ret:+.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.0f}%",
                 fontsize=13, color="#FFFFFF", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_returns.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved yc_returns.png")

    # ── Chart 3: Rolling accuracy ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    window = 30
    # Only count accuracy on agree days
    correct_agree = np.where(agree, (gru_preds > 0.5) == actuals, np.nan)
    rolling_acc = pd.Series(correct_agree).rolling(window, min_periods=10).mean().values * 100

    ax.plot(range(len(rolling_acc)), rolling_acc, color=TEAL, linewidth=2, label="Agreement days")
    ax.axhline(50, color=RED, linestyle="--", alpha=0.5, label="Random (50%)")
    ax.fill_between(range(len(rolling_acc)), 50, rolling_acc,
                    where=~np.isnan(rolling_acc) & (rolling_acc >= 50), alpha=0.1, color=TEAL)
    ax.set_ylabel("Accuracy %", color=MUTED)
    ax.set_xlabel("Test Day", color=MUTED)
    ax.spines[:].set_visible(False); ax.grid(axis="y", alpha=0.08, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_ylim(20, 90)
    ax.set_title(f"Rolling {window}-Day Accuracy (Agreement Days Only)", fontsize=14, color="#FFFFFF", pad=10)
    ax.legend(fontsize=10, frameon=False, labelcolor=MUTED)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_rolling_accuracy.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved yc_rolling_accuracy.png")

    # ── Chart 4: Agreement analysis ──────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(BG)
    for ax in [ax1, ax2]:
        ax.set_facecolor(BG); ax.spines[:].set_visible(False); ax.tick_params(colors=MUTED, labelsize=10)

    # Left: Accuracy by agreement level
    categories = ["GRU Only", "GRU+XGB\nAgree", "Triple\nAgree", "Agree+\nExtreme F&G"]
    acc_vals = [gru_acc*100, agree_acc*100, triple_acc*100, agree_extreme_acc*100]
    coverage = [100, agree.mean()*100, triple.mean()*100, agree_extreme.mean()*100]
    bar_colors = [TEAL, "#00FF88", "#FFFFFF", "#FFD700"]

    bars = ax1.bar(categories, acc_vals, color=bar_colors, width=0.6)
    for b, v, c in zip(bars, acc_vals, coverage):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f"{v:.1f}%\n({c:.0f}%)", ha="center", fontsize=10, color="#FFFFFF")
    ax1.axhline(50, color=RED, linestyle="--", alpha=0.3)
    ax1.set_ylim(0, max(acc_vals) + 15)
    ax1.set_yticks([])
    ax1.set_title("Accuracy vs Agreement Level", fontsize=12, color="#FFFFFF", pad=8)

    # Right: Pie chart of trading days
    sizes = [agree.sum() - triple.sum(), triple.sum() - agree_extreme.sum(),
             agree_extreme.sum(), n_test - agree.sum()]
    labels = [f"GRU+XGB\n({agree.sum()-triple.sum()}d)",
              f"Triple\n({triple.sum()-agree_extreme.sum()}d)",
              f"+F&G ext\n({agree_extreme.sum()}d)",
              f"Abstain\n({n_test-agree.sum()}d)"]
    colors_pie = ["#00FF88", "#FFFFFF", "#FFD700", "#3D3D50"]
    ax2.pie([max(s, 0) for s in sizes], labels=labels, colors=colors_pie,
            textprops={"color": MUTED, "fontsize": 9}, startangle=90)
    ax2.set_title("Trading Day Distribution", fontsize=12, color="#FFFFFF", pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yc_abstention.png"), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved yc_abstention.png")

    # ── Save metrics ──────────────────────────────────────
    metrics = {
        "test_n": n_test,
        "horizon_days": HORIZON,
        "base_rate": float(base_rate),
        "gru_accuracy": float(gru_acc),
        "cnn_lstm_accuracy": float(cnn_acc),
        "xgboost_accuracy": float(xgb_acc),
        "sentiment_accuracy": float(sent_acc),
        "agree_accuracy": float(agree_acc),
        "agree_coverage": float(agree.mean()),
        "triple_agree_accuracy": float(triple_acc),
        "triple_agree_coverage": float(triple.mean()),
        "agree_extreme_fg_accuracy": float(agree_extreme_acc),
        "top40_confidence_accuracy": float(top40_acc),
        "strategy_return": float(strat_cum[-1] - 1),
        "buyhold_return": float(bh_cum[-1] - 1),
        "outperformance": float((strat_cum[-1] - 1) - (bh_cum[-1] - 1)),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd / 100),
    }
    with open(os.path.join(OUTPUT_DIR, "backtest_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved backtest_metrics.json")

    print(f"\n{'='*60}")
    print(f"  GRU+XGB Agreement: {agree_acc:.1%} on {agree.sum()}/{n_test} days")
    print(f"  Triple Agreement:  {triple_acc:.1%} on {triple.sum()}/{n_test} days")
    print(f"  Strategy: {strat_ret:+.1f}% vs B&H: {bh_ret:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f} | Max Drawdown: {max_dd:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
