"""
6-signal engine for BTC prediction.
Each signal returns a standardized dict with direction, strength, reason.

Signals:
  1. Polymarket — prediction market probabilities
  2. Order Flow — Binance BTCUSDT buy/sell pressure
  3. Our Model — 3-day direction model (35 features, gradient boosting)
  4. Technical  — RSI, MACD, Bollinger from TrueMarkets data
  5. Sentiment  — TrueMarkets MCP AI summary
  6. Fear & Greed — alternative.me index
"""

import json
import os
import numpy as np
from app.config import SIGNAL_WEIGHTS_PATH


def _load_weights() -> dict:
    """Load backtested signal weights."""
    defaults = {
        "polymarket": 0.20,
        "order_flow": 0.15,
        "lightgbm": 0.20,
        "technical": 0.20,
        "sentiment": 0.10,
        "fear_greed": 0.15,
    }
    if os.path.exists(SIGNAL_WEIGHTS_PATH):
        try:
            with open(SIGNAL_WEIGHTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return defaults


WEIGHTS = _load_weights()


def compute_polymarket_signal(thresholds: list[dict], current_price: float) -> dict:
    """
    Signal 1: Polymarket prediction market probabilities.
    Compares probability-weighted expected price to current price.
    """
    if not thresholds or current_price <= 0:
        return _signal("Polymarket", "neutral", 0.5, "No Polymarket data available",
                       WEIGHTS.get("polymarket", 0.20), {})

    # Compute implied expected price from threshold probabilities
    up_markets = [t for t in thresholds if t["direction"] == "up"]
    down_markets = [t for t in thresholds if t["direction"] == "down"]

    # Probability that price reaches each threshold
    up_weighted = sum(t["yes_price"] * t["threshold"] for t in up_markets) / max(sum(t["yes_price"] for t in up_markets), 0.01)
    down_weighted = sum(t["yes_price"] * t["threshold"] for t in down_markets) / max(sum(t["yes_price"] for t in down_markets), 0.01)

    # Find nearest significant thresholds
    nearest_up = [t for t in up_markets if t["threshold"] > current_price and t["yes_price"] > 0.05]
    nearest_down = [t for t in down_markets if t["threshold"] < current_price and t["yes_price"] > 0.05]

    up_prob = nearest_up[0]["yes_price"] if nearest_up else 0
    down_prob = nearest_down[-1]["yes_price"] if nearest_down else 0

    if up_prob > down_prob + 0.05:
        direction = "bullish"
        strength = min(0.5 + up_prob * 0.5, 0.95)
        reason = f"Polymarket: {up_prob*100:.0f}% chance of reaching ${nearest_up[0]['threshold']:,}"
    elif down_prob > up_prob + 0.05:
        direction = "bearish"
        strength = max(0.5 - down_prob * 0.5, 0.05)
        reason = f"Polymarket: {down_prob*100:.0f}% chance of dropping to ${nearest_down[-1]['threshold']:,}"
    else:
        direction = "neutral"
        strength = 0.5
        reason = f"Polymarket: balanced — no clear directional signal"

    return _signal("Polymarket", direction, strength, reason,
                   WEIGHTS.get("polymarket", 0.20),
                   {"up_prob": up_prob, "down_prob": down_prob,
                    "nearest_up": nearest_up[0]["threshold"] if nearest_up else None,
                    "nearest_down": nearest_down[-1]["threshold"] if nearest_down else None})


def compute_order_flow_signal(flow: dict) -> dict:
    """
    Signal 2: Binance BTCUSDT order flow.
    Based on buy/sell volume ratio + order book imbalance.
    """
    buy_ratio = flow.get("buy_sell_ratio", 0.5)
    imbalance = flow.get("imbalance", 0)
    pressure = flow.get("pressure", "neutral")

    # ±5% neutral zone: 45-55% buy = neutral
    if buy_ratio >= 0.55:
        direction = "bullish"
        reason = f"Order flow: buyers lead ({buy_ratio*100:.0f}% buy volume), book imbalance {imbalance:+.2f}"
    elif buy_ratio <= 0.45:
        direction = "bearish"
        reason = f"Order flow: sellers lead ({(1-buy_ratio)*100:.0f}% sell volume), book imbalance {imbalance:+.2f}"
    else:
        direction = "neutral"
        reason = f"Order flow: balanced ({buy_ratio*100:.0f}% buy / {(1-buy_ratio)*100:.0f}% sell)"

    strength = buy_ratio  # 0.5 = neutral, >0.5 = bullish, <0.5 = bearish

    return _signal("Order Flow", direction, strength, reason,
                   WEIGHTS.get("order_flow", 0.15),
                   {"buy_volume": flow.get("buy_volume", 0),
                    "sell_volume": flow.get("sell_volume", 0),
                    "buy_sell_ratio": buy_ratio,
                    "imbalance": imbalance,
                    "pressure": pressure,
                    "source": "Binance BTCUSDT"})


def compute_model_signal(prob_up: float, accuracy: float) -> dict:
    """
    Signal 3: CNN-LSTM direction model (Omole & Enke 2024 architecture).
    Uses on-chain data from BGeometrics.
    """
    if prob_up > 0.55:
        direction = "bullish"
        reason = f"Our model: {prob_up*100:.0f}% probability BTC up tomorrow (on-chain signals bullish)"
    elif prob_up < 0.45:
        direction = "bearish"
        reason = f"Our model: {(1-prob_up)*100:.0f}% probability BTC down tomorrow (on-chain signals bearish)"
    else:
        direction = "neutral"
        reason = f"Our model: near 50/50 — on-chain signals mixed"

    return _signal("Our Model", direction, prob_up, reason,
                   WEIGHTS.get("lightgbm", 0.20),
                   {"prob_up": prob_up, "model_accuracy": accuracy})


def compute_technical_signal(rsi: float, macd_hist: float, bollinger_pos: float) -> dict:
    """
    Signal 4: Technical indicators (RSI + MACD + Bollinger).
    """
    signals = []

    # RSI
    if rsi < 30:
        signals.append(("bullish", 0.3, f"RSI oversold ({rsi:.0f})"))
    elif rsi > 70:
        signals.append(("bearish", 0.3, f"RSI overbought ({rsi:.0f})"))
    else:
        signals.append(("neutral", 0.0, f"RSI neutral ({rsi:.0f})"))

    # MACD
    if macd_hist > 0.01:
        signals.append(("bullish", 0.2, "MACD bullish"))
    elif macd_hist < -0.01:
        signals.append(("bearish", 0.2, "MACD bearish"))
    else:
        signals.append(("neutral", 0.0, "MACD flat"))

    # Bollinger
    if bollinger_pos < 0.2:
        signals.append(("bullish", 0.2, f"Price near lower Bollinger ({bollinger_pos:.0%})"))
    elif bollinger_pos > 0.8:
        signals.append(("bearish", 0.2, f"Price near upper Bollinger ({bollinger_pos:.0%})"))
    else:
        signals.append(("neutral", 0.0, f"Bollinger mid-range ({bollinger_pos:.0%})"))

    bullish_score = sum(s[1] for s in signals if s[0] == "bullish")
    bearish_score = sum(s[1] for s in signals if s[0] == "bearish")
    net = bullish_score - bearish_score

    if net > 0.1:
        direction = "bullish"
    elif net < -0.1:
        direction = "bearish"
    else:
        direction = "neutral"

    strength = 0.5 + net
    reasons = [s[2] for s in signals if s[0] != "neutral"]
    if not reasons:
        reasons = [s[2] for s in signals]
    reason = "Technical: " + ", ".join(reasons)

    return _signal("Technical", direction, strength, reason,
                   WEIGHTS.get("technical", 0.20),
                   {"rsi": rsi, "rsi_label": "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
                    "macd_histogram": macd_hist,
                    "bollinger_position": bollinger_pos,
                    "bollinger_label": "Lower band" if bollinger_pos < 0.2 else "Upper band" if bollinger_pos > 0.8 else "Mid-range"})


def compute_sentiment_signal(summary: dict) -> dict:
    """
    Signal 5: TrueMarkets MCP sentiment.
    """
    if not summary:
        return _signal("Sentiment", "neutral", 0.5, "No sentiment data",
                       WEIGHTS.get("sentiment", 0.10), {})

    sentiment_text = summary.get("sentiment", "neutral").lower()
    summary_text = summary.get("summary", "")

    if "bullish" in sentiment_text or "positive" in sentiment_text:
        direction = "bullish"
        strength = 0.62
        reason = f"BTC Sentiment: bullish — {summary_text[:80]}..." if len(summary_text) > 80 else f"BTC Sentiment: bullish — {summary_text}"
    elif "bearish" in sentiment_text or "negative" in sentiment_text:
        direction = "bearish"
        strength = 0.38
        reason = f"BTC Sentiment: bearish — {summary_text[:80]}..." if len(summary_text) > 80 else f"BTC Sentiment: bearish — {summary_text}"
    else:
        direction = "neutral"
        strength = 0.5
        reason = f"BTC Sentiment: neutral"

    return _signal("Sentiment", direction, strength, reason,
                   WEIGHTS.get("sentiment", 0.10),
                   {"sentiment": sentiment_text, "summary": summary_text[:200]})


def compute_fear_greed_signal(fg_data: dict) -> dict:
    """
    Signal 6: Fear & Greed Index (contrarian at extremes).
    """
    current = fg_data.get("current", {})
    value = current.get("value", 50)
    classification = current.get("classification", "Neutral")

    # Contrarian: extreme fear = buy, extreme greed = sell
    if value < 20:
        direction = "bullish"
        strength = 0.70
        reason = f"Fear & Greed: {value} ({classification}) — extreme fear, contrarian buy"
    elif value < 35:
        direction = "bullish"
        strength = 0.58
        reason = f"Fear & Greed: {value} ({classification}) — fear zone, leaning bullish"
    elif value > 80:
        direction = "bearish"
        strength = 0.30
        reason = f"Fear & Greed: {value} ({classification}) — extreme greed, contrarian sell"
    elif value > 65:
        direction = "bearish"
        strength = 0.42
        reason = f"Fear & Greed: {value} ({classification}) — greed zone, leaning bearish"
    else:
        direction = "neutral"
        strength = 0.5
        reason = f"Fear & Greed: {value} ({classification}) — neutral zone"

    return _signal("Fear & Greed", direction, strength, reason,
                   WEIGHTS.get("fear_greed", 0.15),
                   {"value": value, "classification": classification})


def aggregate_signals(signals: list[dict]) -> dict:
    """
    Weighted aggregation of all 6 signals.
    Returns recommended side (buy/sell) with confidence and per-side breakdown.
    """
    total_weight = sum(s["weight"] for s in signals)
    if total_weight == 0:
        total_weight = 1

    weighted_strength = sum(s["strength"] * s["weight"] for s in signals) / total_weight

    buy_signals = [s for s in signals if s["direction"] == "bullish"]
    sell_signals = [s for s in signals if s["direction"] == "bearish"]
    neutral_signals = [s for s in signals if s["direction"] == "neutral"]

    if weighted_strength > 0.52:
        recommended = "buy"
    elif weighted_strength < 0.48:
        recommended = "sell"
    else:
        recommended = "buy" if len(buy_signals) > len(sell_signals) else "sell"

    confidence = abs(weighted_strength - 0.5) * 2

    return {
        "recommended_side": recommended,
        "weighted_strength": round(weighted_strength, 4),
        "confidence": round(confidence, 4),
        "buy_signals": sorted(buy_signals, key=lambda s: -s["weight"]),
        "sell_signals": sorted(sell_signals, key=lambda s: -s["weight"]),
        "neutral_signals": sorted(neutral_signals, key=lambda s: -s["weight"]),
        "buy_count": len(buy_signals),
        "sell_count": len(sell_signals),
        "neutral_count": len(neutral_signals),
        "total_signals": len(signals),
        "weights": {s["name"]: s["weight"] for s in signals},
    }


def _signal(name: str, direction: str, strength: float, reason: str, weight: float, raw_data: dict) -> dict:
    return {
        "name": name,
        "direction": direction,
        "strength": round(float(np.clip(strength, 0.01, 0.99)), 4),
        "reason": reason,
        "weight": round(weight, 4),
        "raw_data": raw_data,
    }
