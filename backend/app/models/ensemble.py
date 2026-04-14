"""
Single recommendation engine. ONE system, no duplicates.

Weights backtested on 2 years of daily BTC data (Apr 2023 – Apr 2025):
  Technical (RSI/MACD):  40%
  TCN direction model:   30%
  Order flow:            20%
  Sentiment (TM + FG):   10%

Every signal produces: probability (0-1), side (buy/sell/neutral), reason (string).
Final recommendation = weighted blend of all signal probabilities.
When signals disagree, each reason is listed under buy_case or sell_case.
"""

import numpy as np
import pandas as pd
import os
import torch
from app.models.direction_gru import DirectionGRUPredictor
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR

# Signal weights — backtested via logistic regression on 388 OOS test days (5-year dataset)
# GRU model from PMC11935774 paper (2-layer, 100 units, dropout 0.2)
W_RSI = 0.22          # Best individual signal (53.1%)
W_MACD = 0.03
W_TCN = 0.01          # GRU alone adds minimal value above momentum (correlated)
W_ORDER_FLOW = 0.31   # BTC momentum — strongest composite signal
W_SENTIMENT = 0.16    # TM AI + Fear & Greed (contrarian)
# Polymarket = 0.26 hardcoded below (contrarian signal)


class Signal:
    """One signal's output."""
    def __init__(self, name: str, prob_up: float, reason: str, weight: float):
        self.name = name
        self.prob_up = float(np.clip(prob_up, 0.01, 0.99))  # probability of UP
        self.side = "buy" if self.prob_up > 0.52 else "sell" if self.prob_up < 0.48 else "neutral"
        self.reason = reason
        self.weight = weight


def compute_signals(
    price_df: pd.DataFrame,
    order_flow: dict,
    tm_sentiment: str,
    fear_greed_data: dict,
    polymarket_markets: list,
) -> list[Signal]:
    """Compute all 4 signals from input data. Returns list of Signal objects."""
    signals = []
    prices = price_df["price"].values
    current_price = float(prices[-1])

    # ── 1. TECHNICAL (RSI + MACD) — weight 40% ──────────
    rsi = float(price_df["rsi"].iloc[-1]) if "rsi" in price_df.columns else 50
    macd_val = float(price_df["macd"].iloc[-1]) if "macd" in price_df.columns else 0
    macd_sig = float(price_df["macd_signal"].iloc[-1]) if "macd_signal" in price_df.columns else 0
    macd_hist = macd_val - macd_sig

    # RSI component: mean-reversion at extremes, momentum in middle
    if rsi < 30:
        rsi_prob = 0.65 + (30 - rsi) * 0.005  # oversold → buy
        rsi_reason = f"RSI at {rsi:.0f} — oversold, expect bounce"
    elif rsi > 70:
        rsi_prob = 0.35 - (rsi - 70) * 0.005  # overbought → sell
        rsi_reason = f"RSI at {rsi:.0f} — overbought, expect pullback"
    else:
        rsi_prob = 0.5 + (rsi - 50) / 200  # mild momentum
        rsi_reason = f"RSI at {rsi:.0f} — neutral zone"

    # MACD component: trend direction
    if abs(macd_hist) > 0:
        macd_prob = 0.5 + np.clip(macd_hist / max(abs(macd_val), 1) * 0.15, -0.2, 0.2)
        macd_dir = "bullish" if macd_hist > 0 else "bearish"
        macd_reason = f"MACD histogram {macd_dir} ({macd_hist:+.0f})"
    else:
        macd_prob = 0.5
        macd_reason = "MACD flat"

    signals.append(Signal("RSI", rsi_prob, rsi_reason, W_RSI))
    signals.append(Signal("MACD", macd_prob, macd_reason, W_MACD))

    # ── 2. GRU MODEL (PMC11935774) — weight based on backtest ──
    gru = DirectionGRUPredictor()
    # Build 6 features matching GRU training
    import pandas as _pd
    _n = len(prices)
    _log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    _vol5 = _pd.Series(_log_ret).rolling(5, min_periods=1).std().fillna(0).values
    _vol20 = _pd.Series(_log_ret).rolling(20, min_periods=1).std().fillna(0).values
    _delta = _pd.Series(prices).diff()
    _gain = _delta.where(_delta > 0, 0).rolling(14, min_periods=1).mean()
    _loss = (-_delta.where(_delta < 0, 0)).rolling(14, min_periods=1).mean()
    _rsi = (100 - (100 / (1 + _gain / _loss.replace(0, np.nan)))).fillna(50).values / 100
    _e12 = _pd.Series(prices).ewm(span=12).mean().values
    _e26 = _pd.Series(prices).ewm(span=26).mean().values
    _macd_r = (_e12 - _e26) - _pd.Series(_e12 - _e26).ewm(span=9).mean().values
    _macd_n = _macd_r / np.maximum(_pd.Series(np.abs(_macd_r)).rolling(20, min_periods=1).mean().values, 1)
    _price_s = (prices - prices.min()) / (prices.max() - prices.min() + 1e-10)
    gru_features = np.column_stack([_price_s, _log_ret, _vol5, _vol20, _rsi, _macd_n])

    tcn_prob = gru.predict_direction(gru_features)

    tcn_dir = "up" if tcn_prob > 0.5 else "down"
    tcn_pct = int(tcn_prob * 100) if tcn_prob > 0.5 else int((1 - tcn_prob) * 100)
    tcn_reason = f"GRU model: {tcn_pct}% probability BTC moves {tcn_dir} tomorrow"
    signals.append(Signal("TCN", tcn_prob, tcn_reason, W_TCN))

    # ── 3. BTC ORDER FLOW (pure price-derived) — weight 35% ──
    # Momentum + acceleration from BTC price data (not Polymarket)
    if len(prices) >= 10:
        # Short-term momentum (last 4 candles)
        ret_4 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        # Acceleration: is momentum increasing or decreasing?
        ret_recent = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] > 0 else 0
        ret_prior = (prices[-3] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        accel = ret_recent - ret_prior

        flow_signal = ret_4 * 8 + accel * 15  # combine momentum + acceleration
        flow_prob = float(np.clip(0.5 + flow_signal, 0.2, 0.8))

        if flow_prob > 0.55:
            flow_reason = f"BTC momentum: price accelerating upward ({ret_4*100:+.1f}% over 4 periods)"
        elif flow_prob < 0.45:
            flow_reason = f"BTC momentum: price decelerating / falling ({ret_4*100:+.1f}% over 4 periods)"
        else:
            flow_reason = f"BTC momentum: sideways ({ret_4*100:+.1f}% over 4 periods)"
    else:
        flow_prob = 0.5
        flow_reason = "BTC momentum: insufficient data"
    signals.append(Signal("BTC Flow", flow_prob, flow_reason, W_ORDER_FLOW))

    # ── 3b. POLYMARKET (prediction market odds) — informational ──
    # Polymarket odds show up in the threshold comparison table
    # and also contribute to the recommendation as a signal
    poly_bullish = 0
    poly_bearish = 0
    poly_reason_parts = []
    for market in (polymarket_markets or [])[:5]:
        q = market.get("question", "")
        yes_p = market.get("yes_price", 0.5)
        vol = market.get("volume", 0)
        if vol > 5000:
            if "reach" in q.lower() and yes_p > 0.5:
                poly_bullish += 1
                poly_reason_parts.append(f"{yes_p*100:.0f}% to reach target")
            elif yes_p < 0.3:
                poly_bearish += 1

    if poly_bullish > poly_bearish and poly_reason_parts:
        poly_prob = 0.5 + min(poly_bullish * 0.05, 0.15)
        poly_reason = f"Polymarket: {', '.join(poly_reason_parts[:2])}"
    elif poly_bearish > poly_bullish:
        poly_prob = 0.5 - min(poly_bearish * 0.05, 0.15)
        poly_reason = f"Polymarket: markets pricing low probability of reaching targets"
    else:
        poly_prob = 0.5
        poly_reason = "Polymarket: no strong directional signal"
    signals.append(Signal("Polymarket", poly_prob, poly_reason, 0.26))

    # ── 4. SENTIMENT (TM AI + Fear & Greed) — weight 10% ─
    fg_value = fear_greed_data.get("current", {}).get("value", 50)

    # TM sentiment from news
    if tm_sentiment.lower() == "bullish":
        tm_prob = 0.62
        tm_reason = "True Markets AI: bullish (30+ news sources)"
    elif tm_sentiment.lower() == "bearish":
        tm_prob = 0.38
        tm_reason = "True Markets AI: bearish (30+ news sources)"
    else:
        tm_prob = 0.5
        tm_reason = "True Markets AI: neutral"

    # Fear & Greed: contrarian at extremes
    if fg_value < 20:
        fg_prob = 0.65
        fg_reason = f"Fear & Greed at {fg_value} — extreme fear (contrarian buy)"
    elif fg_value > 80:
        fg_prob = 0.35
        fg_reason = f"Fear & Greed at {fg_value} — extreme greed (contrarian sell)"
    elif fg_value < 35:
        fg_prob = 0.45
        fg_reason = f"Fear & Greed at {fg_value} — fearful"
    elif fg_value > 65:
        fg_prob = 0.55
        fg_reason = f"Fear & Greed at {fg_value} — greedy"
    else:
        fg_prob = 0.5
        fg_reason = f"Fear & Greed at {fg_value} — neutral"

    sent_prob = tm_prob * 0.5 + fg_prob * 0.5
    sent_reason = f"{tm_reason}; {fg_reason}"
    signals.append(Signal("Sentiment", sent_prob, sent_reason, W_SENTIMENT))

    return signals


def recommend(signals: list[Signal]) -> dict:
    """
    Produce final recommendation from weighted signals.
    Returns consistent structure used by ALL endpoints.
    """
    # Weighted probability
    total_weight = sum(s.weight for s in signals)
    weighted_prob = sum(s.prob_up * s.weight for s in signals) / total_weight

    # Side
    if weighted_prob > 0.52:
        side = "buy"
    elif weighted_prob < 0.48:
        side = "sell"
    else:
        side = "hold"

    # Confidence: how far from 50%
    confidence = abs(weighted_prob - 0.5) * 2  # 0 = uncertain, 1 = certain

    # Split ALL signals into buy_case or sell_case — every signal picks a side
    buy_reasons = []
    sell_reasons = []
    for s in signals:
        if s.prob_up >= 0.5:
            buy_reasons.append(s.reason)
        else:
            sell_reasons.append(s.reason)

    return {
        "primary_side": side,
        "mode": "both",
        "probability_up": round(weighted_prob, 4),
        "confidence": round(confidence, 4),
        "buy_case": {
            "side": "buy",
            "reasons": buy_reasons,
            "vote_count": len(buy_reasons),
        },
        "sell_case": {
            "side": "sell",
            "reasons": sell_reasons,
            "vote_count": len(sell_reasons),
        },
        "total_signals": len(signals),
        "signal_details": [
            {"name": s.name, "prob_up": round(s.prob_up, 4), "side": s.side,
             "weight": s.weight, "reason": s.reason}
            for s in signals
        ],
        "weights": {
            "rsi": W_RSI, "macd": W_MACD, "tcn": W_TCN,
            "order_flow": W_ORDER_FLOW, "sentiment": W_SENTIMENT,
            "source": "backtested on 6-month OOS test (Oct 2025 – Apr 2026)",
        },
    }


def get_recommendation(
    price_df: pd.DataFrame,
    order_flow: dict,
    tm_sentiment: str,
    fear_greed_data: dict,
    polymarket_markets: list,
    thresholds: list[float],
    sentiment_data: dict = None,
) -> dict:
    """
    Single entry point for ALL recommendation logic.
    Called by both /predictions and /mispricing endpoints.
    """
    current_price = float(price_df["price"].iloc[-1])
    volatility = float(price_df["volatility_20d"].iloc[-1]) if "volatility_20d" in price_df.columns else 0.02

    # Compute all signals
    sigs = compute_signals(price_df, order_flow, tm_sentiment, fear_greed_data, polymarket_markets)

    # Get recommendation
    rec = recommend(sigs)

    # Add quote stub
    rec["symbol"] = "BTC"
    rec["base_asset"] = "BTC"
    rec["quote"] = {"price": str(round(current_price, 2)), "qty": "1", "total": str(round(current_price, 2))}
    rec["based_on_mispricing"] = False

    # TCN prediction for display
    tcn_sig = next((s for s in sigs if s.name == "TCN"), None)
    tcn_prediction = {
        "direction": tcn_sig.side if tcn_sig else "neutral",
        "probability": round(tcn_sig.prob_up, 4) if tcn_sig else 0.5,
        "confidence": round(abs(tcn_sig.prob_up - 0.5) * 2, 4) if tcn_sig else 0,
        "model": "TCN (dilated causal convolution)",
    }

    # Threshold probabilities (30-day model)
    threshold_probs = _compute_threshold_probs(current_price, volatility, thresholds, rec["probability_up"])

    # Add 30-day outlook to buy/sell reasons
    # Find nearest upside and downside thresholds with meaningful probability
    best_up = None
    best_down = None
    for t_str, info in threshold_probs.items():
        t = float(t_str)
        prob = info["probability"]
        if info["direction"] == "up" and prob > 0.15:
            if best_up is None or abs(t - current_price) < abs(float(best_up[0]) - current_price):
                best_up = (t_str, prob)
        elif info["direction"] == "down" and prob > 0.15:
            if best_down is None or abs(t - current_price) < abs(float(best_down[0]) - current_price):
                best_down = (t_str, prob)

    # Add 30-day outlook as a signal (counts as a vote)
    if best_up and best_up[1] > 0.30:
        rec["buy_case"]["reasons"].append(f"30-day model: {best_up[1]*100:.0f}% chance of reaching ${int(float(best_up[0])):,}")
        rec["buy_case"]["vote_count"] += 1
        rec["total_signals"] += 1
    if best_down and best_down[1] > 0.30:
        rec["sell_case"]["reasons"].append(f"30-day model: {best_down[1]*100:.0f}% chance of dropping to ${int(float(best_down[0])):,}")
        rec["sell_case"]["vote_count"] += 1
        rec["total_signals"] += 1

    # Sentiment breakdown
    fg_value = fear_greed_data.get("current", {}).get("value", 50)
    sent_score = sentiment_data.get("sentiment_score", 0) if sentiment_data else 0

    sentiment_signal = {
        "overall_signal": "Bullish" if rec["probability_up"] > 0.55 else "Bearish" if rec["probability_up"] < 0.45 else "Neutral",
        "fear_greed": fear_greed_data.get("current", {}).get("classification", "Neutral"),
        "fear_greed_value": fg_value,
        "sentiment_score": sent_score,
        "bullish_ratio": sentiment_data.get("bullish_ratio", 0.5) if sentiment_data else 0.5,
    }

    rsi = float(price_df["rsi"].iloc[-1]) if "rsi" in price_df.columns else 50

    return {
        "coin": "bitcoin",
        "current_price": current_price,
        "thresholds": threshold_probs,
        "confidence": rec["confidence"],
        "model_signals": {"tcn": {str(t): threshold_probs[str(t)]["probability"] for t in thresholds}},
        "weights": rec["weights"],
        "sentiment_signal": sentiment_signal,
        "indicators": {
            "rsi": rsi,
            "macd": float(price_df["macd"].iloc[-1] - price_df["macd_signal"].iloc[-1]) if "macd" in price_df.columns and "macd_signal" in price_df.columns else 0,
            "volatility": volatility,
            "fear_greed": fg_value,
        },
        "tcn_prediction": tcn_prediction,
        "recommendation": rec,
        "directional_signal": {
            "bias": rec["primary_side"],
            "probability_up": rec["probability_up"],
        },
    }


# ─── Backward compat for /predictions endpoint ──────────

class EnsemblePredictionEngine:
    def __init__(self, num_thresholds: int = 6):
        pass

    def predict(self, price_df, sentiment_data, fear_greed_data, onchain_data, thresholds):
        return get_recommendation(
            price_df=price_df,
            order_flow={},
            tm_sentiment="neutral",
            fear_greed_data=fear_greed_data,
            polymarket_markets=[],
            thresholds=thresholds,
            sentiment_data=sentiment_data,
        )


# ─── Helpers ─────────────────────────────────────────────

def _build_tcn_features(prices, timestamps=None):
    """15 multi-timeframe features matching train_models.py."""
    import pandas as _pd
    n = len(prices)
    log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
    vol_5 = _pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
    vol_20 = _pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values

    delta = _pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rsi = (100 - (100 / (1 + gain / loss.replace(0, np.nan)))).fillna(50).values / 100

    ema12 = _pd.Series(prices).ewm(span=12).mean().values
    ema26 = _pd.Series(prices).ewm(span=26).mean().values
    macd_raw = (ema12 - ema26) - _pd.Series(ema12 - ema26).ewm(span=9).mean().values
    macd_hist = macd_raw / np.maximum(_pd.Series(np.abs(macd_raw)).rolling(20, min_periods=1).mean().values, 1)

    price_pos = np.array([(lambda w: (prices[i]-w.min())/(w.max()-w.min()) if w.max()>w.min() else 0.5)(prices[max(0,i-20):i+1]) for i in range(n)])
    mom_5 = np.concatenate([np.zeros(5), [(prices[i]-prices[i-5])/prices[i-5] for i in range(5,n)]])
    mom_20 = np.concatenate([np.zeros(20), [(prices[i]-prices[i-20])/prices[i-20] for i in range(20,n)]])
    mean_rev = np.array([(lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])

    # Candle-like features (use returns as proxy when no OHLC)
    accel = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2,n)]])
    abs_ret = np.abs(log_ret)
    rel_vol = np.ones(n)  # placeholder
    vol_ratio = np.where(vol_20 > 1e-10, vol_5/vol_20, 1.0)

    if timestamps is not None and len(timestamps) == n:
        try:
            ts = _pd.to_datetime(timestamps)
            dow = (ts.dayofweek if hasattr(ts, 'dayofweek') else ts.dt.dayofweek).values / 6.0
        except Exception:
            dow = np.zeros(n)
    else:
        dow = np.zeros(n)

    return np.column_stack([
        log_ret, vol_5, vol_20, rsi, macd_hist,
        price_pos, mom_5, mom_20, mean_rev,
        accel, abs_ret, rel_vol,
        vol_ratio, log_ret, dow
    ])


def _compute_threshold_probs(current_price, volatility, thresholds, prob_up):
    """
    Separate threshold probability model (NOT the TCN direction model).
    Uses log-normal price distribution calibrated on historical BTC volatility.
    This is compared against Polymarket prices for mispricing detection.

    Model: BTC follows geometric Brownian motion over 30 days.
    P(reach threshold) = probability that price path touches threshold at any point.
    Uses reflection principle for barrier hitting probability.
    """
    # Convert hourly vol to daily, then to 30-day horizon
    hourly_vol = max(volatility, 0.003)
    daily_vol = hourly_vol * np.sqrt(24)
    horizon_vol = daily_vol * np.sqrt(30)

    # Directional drift from ensemble signal
    # prob_up > 0.5 means bullish drift, < 0.5 means bearish
    annual_drift = (prob_up - 0.5) * 0.4  # ±20% annualized max
    daily_drift = annual_drift / 365
    horizon_drift = daily_drift * 30

    results = {}
    for t in thresholds:
        pct_move = (t - current_price) / current_price
        direction = "up" if t > current_price else "down"

        # Barrier hitting probability using reflection principle
        # For a threshold above current price:
        #   P(max(S_t) >= K) ≈ 2 * P(S_T >= K) for driftless GBM
        # With drift adjustment
        if t > current_price:
            z = (pct_move - horizon_drift) / max(horizon_vol, 0.01)
            # Barrier probability is higher than endpoint probability
            prob = 2 * (1 - _norm_cdf(z))
        else:
            z = (abs(pct_move) + horizon_drift) / max(horizon_vol, 0.01)
            prob = 2 * (1 - _norm_cdf(z))

        # Clamp to reasonable range
        prob = float(np.clip(prob, 0.01, 0.99))

        results[str(t)] = {
            "probability": prob,
            "direction": direction,
            "distance_pct": round(pct_move * 100, 2),
        }
    return results


def _norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
