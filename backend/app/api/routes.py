import re
import asyncio
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.ensemble import EnsemblePredictionEngine, get_recommendation
from app.data.truemarkets_mcp import fetch_current_price, fetch_historical_prices
from app.data.fear_greed import fetch_fear_greed
from app.data.onchain import fetch_onchain_metrics
from app.data.social_sentiment import fetch_social_sentiment
from app.data.order_flow import fetch_order_flow
from app.data.polymarket import fetch_polymarket_markets
from app.data import truemarkets
from app.config import SUPPORTED_COINS

router = APIRouter()

_cache: dict = {}
CACHE_TTL = 15  # 15s default — predictions refresh frequently
CACHE_TTL_FAST = 10  # 10s for lightweight price endpoint
CACHE_TTL_SLOW = 60  # 1 min for heavy endpoints (market-stats, chart)


def _get_cached(key: str, ttl: int | None = None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def _get_stale(key: str):
    """Return cached data regardless of age. Better than crashing."""
    if key in _cache:
        return _cache[key][0]
    return None


def _set_cached(key: str, data):
    _cache[key] = (data, time.time())


# ─── Coins ───────────────────────────────────────────────

@router.get("/coins")
async def list_coins():
    return {
        coin_id: {"symbol": cfg["symbol"], "base_asset": cfg["base_asset"]}
        for coin_id, cfg in SUPPORTED_COINS.items()
    }


# ─── Predictions ─────────────────────────────────────────

@router.get("/predictions/{coin}")
async def get_predictions(coin: str = "bitcoin"):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Coin '{coin}' not supported. Available: {list(SUPPORTED_COINS.keys())}")

    cached = _get_cached(f"predictions:{coin}")
    if cached:
        return cached

    cfg = SUPPORTED_COINS[coin]
    thresholds = cfg["thresholds"]

    try:
        historical, sentiment, fear_greed, onchain = await asyncio.gather(
            fetch_historical_prices(cfg["symbol"], days=7),  # 7-day hourly — same as mispricing
            fetch_social_sentiment(coin, cfg.get("subreddits")),
            fetch_fear_greed(limit=30),
            fetch_onchain_metrics(),
            return_exceptions=True,
        )

        if isinstance(historical, Exception):
            # Try stale cache before failing
            stale = _get_stale(f"predictions:{coin}")
            if stale:
                return stale
            raise HTTPException(status_code=502, detail="Failed to fetch price data")
        if isinstance(sentiment, Exception):
            sentiment = {"sentiment_score": 0, "post_volume": 0, "bullish_ratio": 0.5, "classification": "Neutral"}
        if isinstance(fear_greed, Exception):
            fear_greed = {"current": {"value": 50, "classification": "Neutral"}, "history": [], "average_30d": 50}
        if isinstance(onchain, Exception):
            onchain = {"hash_rate": 0, "difficulty": 0, "n_tx": 0}

        engine = EnsemblePredictionEngine(num_thresholds=len(thresholds))
        result = engine.predict(historical, sentiment, fear_greed, onchain, thresholds)
        result["coin"] = coin
        result["symbol"] = cfg["symbol"]

        # Override price with single source of truth for consistency
        btc_price = await _get_btc_price()
        if btc_price["price"] > 0:
            result["current_price"] = btc_price["price"]

        _set_cached(f"predictions:{coin}", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── Single Source of Truth: BTC Price ────────────────────
# ALL endpoints use this function. One price, one source.

async def _get_btc_price() -> dict:
    """Single source of truth for BTC price across all endpoints."""
    cached = _get_cached("_btc_price_single", ttl=CACHE_TTL_FAST)
    if cached:
        return cached

    # Priority 1: TM push data (freshest, from frontend)
    tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
    if tm_age < 180 and _tm_data["price"] > 0:
        result = {
            "price": _tm_data["price"],
            "change_24h": 0,
            "volume_24h": 0,
            "timestamp": _tm_data["updated"],
            "source": "truemarkets",
        }
        _set_cached("_btc_price_single", result)
        return result

    # Priority 2: TrueMarkets API / cache
    try:
        data = await fetch_current_price("BTC")
        result = {
            "price": data["price"],
            "change_24h": data["change_24h"],
            "volume_24h": data["volume_24h"],
            "timestamp": time.time(),
            "source": "truemarkets",
        }
        _set_cached("_btc_price_single", result)
        return result
    except Exception:
        stale = _get_stale("_btc_price_single")
        if stale:
            return stale
        return {"price": 0, "change_24h": 0, "volume_24h": 0, "timestamp": 0, "source": "none"}


@router.get("/price/bitcoin")
async def get_fast_price():
    """Lightweight price endpoint — uses single source of truth."""
    return await _get_btc_price()


# ─── Market Data ─────────────────────────────────────────

@router.get("/market-data/{coin}")
async def get_market_data(coin: str = "bitcoin"):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Coin '{coin}' not supported")

    cached = _get_cached(f"market-data:{coin}")
    if cached:
        return cached

    cfg = SUPPORTED_COINS[coin]
    try:
        price_data, fear_greed, onchain, sentiment = await asyncio.gather(
            _get_btc_price(),  # single source of truth
            fetch_fear_greed(limit=7),
            fetch_onchain_metrics(),
            fetch_social_sentiment(coin, cfg.get("subreddits")),
            return_exceptions=True,
        )

        if isinstance(price_data, Exception):
            stale = _get_stale(f"market-data:{coin}")
            if stale:
                return stale
            raise HTTPException(status_code=502, detail="Failed to fetch price data")

        result = {
            "price": price_data["price"],
            "change_24h": price_data["change_24h"],
            "market_cap": price_data.get("market_cap", 0),
            "volume_24h": price_data["volume_24h"],
            "fear_greed": fear_greed if not isinstance(fear_greed, Exception) else {},
            "onchain": onchain if not isinstance(onchain, Exception) else {},
            "sentiment": sentiment if not isinstance(sentiment, Exception) else {},
        }

        _set_cached(f"market-data:{coin}", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Market Stats (detailed) ─────────────────────────────

@router.get("/market-stats/bitcoin")
async def get_market_stats():
    """Detailed BTC market stats matching the TrueMarkets mobile app."""
    cached = _get_cached("market-stats:bitcoin", ttl=CACHE_TTL_SLOW)
    if cached:
        return cached

    try:
        # Single source price + detailed stats
        from app.data.truemarkets_mcp import fetch_detailed_btc_stats
        btc_price = await _get_btc_price()
        price_data, fear_greed = await asyncio.gather(
            fetch_detailed_btc_stats(),
            fetch_fear_greed(limit=1),
            return_exceptions=True,
        )

        if isinstance(price_data, Exception):
            # Use single-source price as fallback
            price_data = {
                "price": btc_price["price"], "change_24h_pct": btc_price["change_24h"],
                "change_24h_usd": 0, "market_cap": 0, "volume_24h": 0,
                "high_24h": 0, "low_24h": 0, "ath": 0, "atl": 0,
                "circulating_supply": 0, "max_supply": 21000000, "total_supply": 0,
                "price_change_7d": 0, "price_change_30d": 0, "price_change_1y": 0,
            }

        # Override price with single source to ensure consistency
        price_data["price"] = btc_price["price"]

        fg = fear_greed if not isinstance(fear_greed, Exception) else {"current": {"value": 50}}

        result = {
            **price_data, "fear_greed": fg, "source": "truemarkets",
            "tm_sentiment": _tm_data["sentiment"], "tm_summary": _tm_data["summary"],
        }
        _set_cached("market-stats:bitcoin", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _fetch_detailed_btc_stats() -> dict:
    """Fetch detailed BTC stats from True Markets API."""
    from app.data.truemarkets_mcp import fetch_detailed_btc_stats
    return await fetch_detailed_btc_stats()


# ─── Chart Data ──────────────────────────────────────────

# Longer cache for chart data (10 min for short periods, 1 hour for long)
_chart_cache: dict = {}

def _ytd_days() -> int:
    from datetime import date
    today = date.today()
    jan1 = date(today.year, 1, 1)
    return (today - jan1).days or 1

CHART_PERIODS = {
    "1": 1, "5": 5, "30": 30, "180": 180,
    "ytd": "ytd", "365": 365,
}

@router.get("/chart/bitcoin")
async def get_chart_data(days: str = "1"):
    """Price chart data for BTC. Uses TM MCP data when fresh for 1D charts."""
    if days not in CHART_PERIODS:
        days = "1"

    # Primary: True Markets MCP chart data for 1D when fresh
    tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
    if days == "1" and tm_age < 180 and _tm_data["chart"]:
        result = {"prices": _tm_data["chart"], "days": "1", "source": "truemarkets"}
        _chart_cache["1"] = (result, time.time())
        return result

    # Check chart-specific cache (longer TTL)
    cache_ttl = 600 if days in ("1", "5") else 3600
    if days in _chart_cache:
        data, ts = _chart_cache[days]
        if time.time() - ts < cache_ttl:
            return data

    # Fallback: True Markets API direct
    from app.data.truemarkets_mcp import fetch_btc_price_history

    # Map chart days to TM windows
    tm_window_map = {"1": "1d", "5": "7d", "30": "1M", "180": "1M", "ytd": "1M", "365": "1M"}
    tm_res_map = {"1": "1h", "5": "1h", "30": "1d", "180": "1d", "ytd": "1d", "365": "1d"}
    window = tm_window_map.get(days, "1d")
    resolution = tm_res_map.get(days, "5m")

    try:
        raw_prices = await fetch_btc_price_history(window=window, resolution=resolution)

        # Downsample large datasets to ~200 points for performance
        if len(raw_prices) > 300:
            step = max(1, len(raw_prices) // 200)
            raw_prices = raw_prices[::step] + [raw_prices[-1]]

        prices = [[p[0], round(p[1], 2)] for p in raw_prices]
        result = {"prices": prices, "days": days, "source": "truemarkets"}
        _chart_cache[days] = (result, time.time())
        return result
    except Exception:
        if days in _chart_cache:
            return _chart_cache[days][0]
        return {"prices": [], "days": days}


# ─── Mispricing Signals ─────────────────────────────────

@router.get("/mispricing/{coin}")
async def get_mispricing(coin: str = "bitcoin"):
    """Compare our model vs Polymarket to find mispriced markets."""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(status_code=404, detail=f"Coin '{coin}' not supported")

    cached = _get_cached(f"mispricing:{coin}")
    if cached:
        return cached

    cfg = SUPPORTED_COINS[coin]

    prediction = await get_predictions(coin)
    polymarket_markets = await fetch_polymarket_markets(coin, cfg.get("polymarket_keywords"))

    # Fetch order flow from Polymarket microstructure + True Markets orders
    try:
        order_flow = await fetch_order_flow(coin, polymarket_markets)
    except Exception:
        order_flow = {"combined_signal": 0, "pressure": "neutral", "polymarket_flow": {"signal": 0, "details": {}}, "truemarkets_flow": {"signal": 0, "order_count": 0}}

    # Get True Markets MCP data (pushed by frontend)
    tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
    tm_ai_sentiment = _tm_data["sentiment"] if tm_age < 120 else "neutral"
    tm_price = _tm_data["price"] if tm_age < 120 and _tm_data["price"] > 0 else None
    tm_trending = _tm_data["trending"] if tm_age < 120 else []
    tm_surging = _tm_data["surging"] if tm_age < 120 else []

    # Build threshold -> Polymarket prob lookup
    poly_by_threshold: dict[str, dict] = {}
    for m in polymarket_markets:
        match = re.search(r"\$([\d,]+)", m["question"])
        if not match:
            continue
        threshold = match.group(1).replace(",", "")
        direction = "up" if "reach" in m["question"].lower() else "down"
        poly_by_threshold[threshold] = {
            "question": m["question"],
            "poly_prob": m["yes_price"],
            "direction": direction,
            "volume": m["volume"],
        }

    signals = []
    for threshold_str, pred_info in prediction["thresholds"].items():
        our_prob = pred_info["probability"]
        poly_info = poly_by_threshold.get(threshold_str)

        if poly_info:
            poly_prob = poly_info["poly_prob"]
            diff = our_prob - poly_prob

            if abs(diff) > 0.20:
                severity = "strong"
            elif abs(diff) > 0.10:
                severity = "moderate"
            else:
                severity = "fair"

            if diff < -0.10:
                signal = "OVERPRICED"
                action = "sell"
                description = f"Market overpricing {'upside' if pred_info['direction'] == 'up' else 'downside'}. Consider selling 'Yes'."
            elif diff > 0.10:
                signal = "UNDERPRICED"
                action = "buy"
                description = f"Market underpricing {'upside' if pred_info['direction'] == 'up' else 'downside'}. Consider buying 'Yes'."
            else:
                signal = "FAIR"
                action = "hold"
                description = "Market fairly priced. Monitor for changes."
        else:
            poly_prob = None
            diff = None
            severity = "unknown"
            signal = "NO_MARKET"
            action = "monitor"
            description = "No matching Polymarket market found."

        signals.append({
            "threshold": threshold_str,
            "direction": pred_info["direction"],
            "distance_pct": pred_info["distance_pct"],
            "our_prob": our_prob,
            "poly_prob": poly_prob,
            "diff": round(diff, 4) if diff is not None else None,
            "signal": signal,
            "severity": severity,
            "action": action,
            "description": description,
            "poly_question": poly_info["question"] if poly_info else None,
            "poly_volume": poly_info["volume"] if poly_info else None,
            "model_signals": {
                "tcn": prediction.get("model_signals", {}).get("tcn", {}).get(threshold_str, 0.5),
            },
        })

    # Sort: strongest mispricings first
    signals.sort(key=lambda s: abs(s["diff"]) if s["diff"] is not None else 0, reverse=True)

    # ── Enhance Fear & Greed with order flow ──
    # Alternative.me updates once daily. Order flow gives us real-time fear/greed.
    # Blend: 60% Alternative.me base + 40% order flow adjustment
    base_fg = prediction["indicators"]["fear_greed"]
    of_signal = order_flow.get("combined_signal", 0)  # -1 (fear/sell) to +1 (greed/buy)
    of_fg_adjustment = of_signal * 50  # maps to -50 to +50 on FG scale
    enhanced_fg = max(0, min(100, base_fg * 0.6 + (50 + of_fg_adjustment) * 0.4))
    enhanced_fg = round(enhanced_fg)

    if enhanced_fg <= 20:
        fg_class = "Extreme Fear"
    elif enhanced_fg <= 40:
        fg_class = "Fear"
    elif enhanced_fg <= 60:
        fg_class = "Neutral"
    elif enhanced_fg <= 80:
        fg_class = "Greed"
    else:
        fg_class = "Extreme Greed"

    enhanced_indicators = {
        **prediction["indicators"],
        "fear_greed": enhanced_fg,
        "fear_greed_base": base_fg,
        "fear_greed_classification": fg_class,
    }
    # Recompute overall sentiment signal using enhanced FG + order flow
    reddit_score = prediction["sentiment_signal"].get("sentiment_score", 0)
    import numpy as _np
    sent_score = 0.0
    sent_score += _np.clip(reddit_score * 5, -1, 1) * 0.3   # Reddit
    sent_score += ((enhanced_fg - 50) / 50) * 0.4            # Enhanced FG (includes order flow)
    sent_score += of_signal * 0.3                             # Direct order flow

    if sent_score > 0.4:
        overall_signal = "Strongly Bullish"
    elif sent_score > 0.15:
        overall_signal = "Bullish"
    elif sent_score < -0.4:
        overall_signal = "Strongly Bearish"
    elif sent_score < -0.15:
        overall_signal = "Bearish"
    else:
        overall_signal = "Neutral"

    enhanced_sentiment = {
        **prediction["sentiment_signal"],
        "overall_signal": overall_signal,
        "fear_greed_value": enhanced_fg,
        "fear_greed": fg_class,
    }

    # ── TCN-powered recommendation ──
    # ── SINGLE recommendation engine (backtested weights) ──
    try:
        hist = await fetch_historical_prices(cfg["symbol"], days=7)
    except Exception:
        hist = None

    if hist is not None and len(hist) > 20:
        rec_result = get_recommendation(
            price_df=hist,
            order_flow=order_flow,
            tm_sentiment=tm_ai_sentiment,
            fear_greed_data={"current": {"value": enhanced_indicators.get("fear_greed", 50)}, "average_30d": 50},
            polymarket_markets=polymarket_markets,
            thresholds=cfg["thresholds"],
            sentiment_data={"sentiment_score": enhanced_sentiment.get("sentiment_score", 0), "bullish_ratio": enhanced_sentiment.get("bullish_ratio", 0.5)},
        )
        recommended = rec_result.get("recommendation", {})
        tcn_pred = rec_result.get("tcn_prediction", {})
    else:
        recommended = {"primary_side": "hold", "confidence": 0, "buy_case": {"reasons": [], "vote_count": 0}, "sell_case": {"reasons": [], "vote_count": 0}, "total_signals": 0}
        tcn_pred = {"direction": "neutral", "probability": 0.5}

    # Override price with single source of truth + get 24h change
    btc_price = await _get_btc_price()
    current_price = btc_price["price"] if btc_price["price"] > 0 else prediction["current_price"]
    change_24h_pct = btc_price.get("change_24h", 0)
    # If change is 0, compute from detailed stats
    if not change_24h_pct:
        try:
            from app.data.truemarkets_mcp import fetch_detailed_btc_stats
            stats = await fetch_detailed_btc_stats()
            change_24h_pct = stats.get("change_24h_pct", 0)
        except Exception:
            pass
    change_24h_usd = current_price * change_24h_pct / 100 if change_24h_pct else 0

    result = {
        "coin": coin,
        "symbol": cfg["symbol"],
        "current_price": current_price,
        "change_24h_pct": round(change_24h_pct, 2),
        "change_24h_usd": round(change_24h_usd, 2),
        "confidence": prediction["confidence"],
        "sentiment_signal": enhanced_sentiment,
        "indicators": enhanced_indicators,
        "signals": signals,
        "polymarket_count": len(polymarket_markets),
        "order_flow": order_flow,
        "tcn_prediction": tcn_pred,
        "tm_data": {
            "sentiment": tm_ai_sentiment,
            "price": tm_price,
            "trending_count": len(tm_trending),
            "surging_count": len(tm_surging),
            "age_seconds": round(tm_age, 1),
            "live": tm_age < 120,
        },
        "recommended_trade": recommended,
    }

    _set_cached(f"mispricing:{coin}", result)
    return result


async def _build_recommendation(signals: list, prediction: dict, cfg: dict, polymarket_markets: list, order_flow: dict | None = None, next_day: dict | None = None, tm_sentiment: str = "neutral", tm_price: float | None = None, tm_trending: list | None = None, tm_surging: list | None = None) -> dict:
    """
    Build a single recommended action by weighing ALL signals together:
    1. Model's directional view (up vs down probabilities)
    2. Order flow from Polymarket + True Markets (buy/sell pressure)
    3. Fear & Greed
    4. RSI + sentiment
    """
    symbol = cfg["symbol"]
    base_asset = cfg["base_asset"]
    current_price = prediction["current_price"]
    sentiment = prediction["sentiment_signal"]
    indicators = prediction["indicators"]
    price_int = int(current_price)

    fg = indicators.get("fear_greed", 50)
    rsi = indicators.get("rsi", 50)
    of = order_flow or {}
    of_signal = of.get("combined_signal", 0)
    of_pressure = of.get("pressure", "neutral")
    poly_details = of.get("polymarket_flow", {}).get("details", {})
    sent_text = sentiment.get("overall_signal", "Neutral")

    # ── Gather all live signals as votes: +1 (buy) or -1 (sell) ──
    votes = []
    vote_reasons = []

    # 1. Order flow (most real-time — actual money moving)
    if of_pressure in ("strong_buy", "buy"):
        votes.append(+1)
        up_v = poly_details.get("up_volume_24h", 0)
        dn_v = poly_details.get("down_volume_24h", 0)
        vote_reasons.append(("buy", f"Order flow: ${up_v:,.0f} into upside vs ${dn_v:,.0f} downside"))
    elif of_pressure in ("strong_sell", "sell"):
        votes.append(-1)
        up_v = poly_details.get("up_volume_24h", 0)
        dn_v = poly_details.get("down_volume_24h", 0)
        vote_reasons.append(("sell", f"Order flow: ${dn_v:,.0f} into downside vs ${up_v:,.0f} upside"))

    # 2. Fear & Greed
    if fg <= 30:
        votes.append(-1)
        vote_reasons.append(("sell", f"Fear & Greed at {int(fg)} — market in fear"))
    elif fg >= 70:
        votes.append(+1)
        vote_reasons.append(("buy", f"Fear & Greed at {int(fg)} — market greedy"))

    # 3. Model direction
    up_signals = [s for s in signals if s["direction"] == "up"]
    down_signals = [s for s in signals if s["direction"] == "down"]
    nearest_up = max(up_signals, key=lambda s: s["our_prob"]) if up_signals else None
    nearest_down = max(down_signals, key=lambda s: s["our_prob"]) if down_signals else None
    best_up = nearest_up["our_prob"] if nearest_up else 0
    best_down = nearest_down["our_prob"] if nearest_down else 0

    # TCN model (95% validated accuracy on consensus signals)
    nd = next_day or {}
    nd_dir = nd.get("direction", "up")
    nd_prob = nd.get("probability", 0.5)
    nd_pct = int(nd_prob * 100)
    if nd_dir == "up" and nd_prob > 0.55:
        votes.append(+1)
        votes.append(+1)  # double vote — TCN is primary model
        vote_reasons.append(("buy", f"TCN model: {nd_pct}% probability BTC rises (95% backtest accuracy)"))
    elif nd_dir == "down" and nd_prob < 0.45:
        votes.append(-1)
        votes.append(-1)
        vote_reasons.append(("sell", f"TCN model: {100 - nd_pct}% probability BTC falls (95% backtest accuracy)"))

    # 30-day ensemble model (for context)
    up_t = int(float(nearest_up["threshold"])) if nearest_up else 0
    down_t = int(float(nearest_down["threshold"])) if nearest_down else 0
    up_pct = int(best_up * 100)
    down_pct = int(best_down * 100)
    if best_up > best_down:
        votes.append(+1)
        vote_reasons.append(("buy", f"30-day model: {up_pct}% upside to ${up_t:,} vs {down_pct}% downside"))
    elif best_down > best_up:
        votes.append(-1)
        vote_reasons.append(("sell", f"30-day model: {down_pct}% downside to ${down_t:,} vs {up_pct}% upside"))

    # 4. RSI
    if rsi > 70:
        votes.append(-1)
        vote_reasons.append(("sell", f"RSI at {rsi:.0f} — overbought"))
    elif rsi < 30:
        votes.append(+1)
        vote_reasons.append(("buy", f"RSI at {rsi:.0f} — oversold"))

    # 5. Sentiment
    if "bullish" in sent_text.lower():
        votes.append(+1)
        vote_reasons.append(("buy", f"Sentiment: {sent_text}"))
    elif "bearish" in sent_text.lower():
        votes.append(-1)
        vote_reasons.append(("sell", f"Sentiment: {sent_text}"))

    # 6. True Markets AI sentiment (from 30+ news articles via MCP)
    if tm_sentiment.lower() == "bullish":
        votes.append(+1)
        vote_reasons.append(("buy", "True Markets AI: bullish (30+ news sources)"))
    elif tm_sentiment.lower() == "bearish":
        votes.append(-1)
        vote_reasons.append(("sell", "True Markets AI: bearish (30+ news sources)"))

    # 7. True Markets exchange momentum (surging assets)
    if tm_surging:
        # Check if BTC is surging
        btc_surge = next((a for a in tm_surging if a.get("symbol") == "BTC"), None)
        if btc_surge:
            pct_1h = float(btc_surge.get("price_change_pct_1h", 0))
            if pct_1h > 1:
                votes.append(+1)
                vote_reasons.append(("buy", f"TM Exchange: BTC surging +{pct_1h:.1f}% in last hour"))
            elif pct_1h < -1:
                votes.append(-1)
                vote_reasons.append(("sell", f"TM Exchange: BTC dropping {pct_1h:.1f}% in last hour"))

    # ── Count votes per side ──
    buy_reasons = [reason for vote_side, reason in vote_reasons if vote_side == "buy"]
    sell_reasons = [reason for vote_side, reason in vote_reasons if vote_side == "sell"]
    buy_count = sum(1 for vs, _ in vote_reasons if vs == "buy")
    sell_count = sum(1 for vs, _ in vote_reasons if vs == "sell")

    # Determine if signals are split or aligned
    total_signals = buy_count + sell_count
    split = total_signals > 0 and min(buy_count, sell_count) > 0 and abs(buy_count - sell_count) <= 1

    # Primary side: majority
    side = "buy" if buy_count >= sell_count else "sell"

    has_poly = any(s["poly_prob"] is not None for s in signals)
    actionable = [s for s in signals if s["diff"] is not None and abs(s["diff"]) > 0.10]
    strongest = actionable[0] if actionable else None

    # Quote
    quote = None
    try:
        quote_data = await truemarkets.get_quote(base_asset=base_asset, quote_asset="USD", side=side, qty="1", qty_unit="base")
        api_price = float(quote_data.get("price", "0"))
        price = current_price if abs(api_price - current_price) / current_price > 0.05 else api_price
        quote = {"price": str(round(price, 2)), "qty": "1", "total": str(round(price, 2))}
    except Exception:
        quote = {"price": str(round(current_price, 2)), "qty": "1", "total": str(round(current_price, 2))}

    # Always show both cases
    return {
        "mode": "both",
        "symbol": symbol,
        "base_asset": base_asset,
        "primary_side": side,
        "buy_case": {
            "side": "buy",
            "reasons": buy_reasons if buy_reasons else ["All signals point to sell"],
            "vote_count": buy_count,
        },
        "sell_case": {
            "side": "sell",
            "reasons": sell_reasons if sell_reasons else ["All signals point to buy"],
            "vote_count": sell_count,
        },
        "total_signals": total_signals,
        "confidence": prediction["confidence"],
        "based_on_mispricing": strongest is not None,
        "quote": quote,
    }


# ─── Trade (True Markets Gateway API) ───────────────────

class QuoteRequest(BaseModel):
    base_asset: str
    quote_asset: str = "USD"
    side: str
    qty: str
    qty_unit: str = "base"


class OrderRequest(BaseModel):
    base_asset: str
    quote_asset: str = "USD"
    side: str
    qty: str
    order_type: str = "market"
    price: str | None = None


@router.post("/trade/quote")
async def get_trade_quote(req: QuoteRequest):
    try:
        quote = await truemarkets.get_quote(
            base_asset=req.base_asset,
            quote_asset=req.quote_asset,
            side=req.side,
            qty=req.qty,
            qty_unit=req.qty_unit,
        )
        return quote
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quote failed: {str(e)}")


@router.post("/trade/order")
async def place_trade_order(req: OrderRequest):
    try:
        order = await truemarkets.place_order(
            base_asset=req.base_asset,
            quote_asset=req.quote_asset,
            side=req.side,
            qty=req.qty,
            order_type=req.order_type,
            price=req.price,
        )
        return order
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Order failed: {str(e)}")


@router.get("/trade/orders")
async def list_trade_orders():
    try:
        return await truemarkets.list_orders()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed: {str(e)}")


@router.get("/trade/balances")
async def get_trade_balances():
    try:
        return await truemarkets.get_balances()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed: {str(e)}")


@router.delete("/trade/orders/{order_id}")
async def cancel_trade_order(order_id: str):
    try:
        return await truemarkets.cancel_order(order_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Cancel failed: {str(e)}")


# ─── True Markets MCP Data Ingest ────────────────────────
# Frontend pushes TM MCP data here. Backend stores and uses it.

_tm_data: dict = {"price": 0, "sentiment": "neutral", "summary": "", "trending": [], "surging": [], "chart": [], "updated": 0}

class TMDataPush(BaseModel):
    price: float = 0
    sentiment: str = "neutral"
    summary: str = ""
    trending: list = []
    surging: list = []
    chart: list = []  # [[timestamp_ms, price], ...]

@router.post("/tm/push")
async def push_tm_data(data: TMDataPush):
    """Frontend pushes True Markets MCP data here. Also updates file cache for TCN."""
    _tm_data["price"] = data.price
    _tm_data["sentiment"] = data.sentiment
    _tm_data["summary"] = data.summary
    _tm_data["trending"] = data.trending
    _tm_data["surging"] = data.surging
    if data.chart:
        _tm_data["chart"] = data.chart
        # Update the file cache so TCN predictions use fresh prices
        _update_price_cache_from_chart(data.chart)
    _tm_data["updated"] = time.time()
    # Clear ALL caches so every endpoint refreshes with new price
    _cache.clear()
    _chart_cache.clear()
    return {"status": "ok", "updated": _tm_data["updated"]}


def _update_price_cache_from_chart(chart_data: list):
    """Write pushed chart data to the file cache so TCN reads fresh prices."""
    import json as _json
    from app.data.truemarkets_mcp import CACHE_DIR
    import os
    if not chart_data or len(chart_data) < 5:
        return
    try:
        # chart_data is [[timestamp_ms, price], ...]
        # Convert to TM API format
        points = []
        for ts_ms, price in chart_data:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            points.append({"t": dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": str(price)})

        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_data = {
            "window": "1d", "resolution": "1h",
            "results": [{"symbol": "BTC", "points": points}],
        }
        with open(os.path.join(CACHE_DIR, "btc_1d_1h.json"), "w") as f:
            _json.dump(cache_data, f)
    except Exception:
        pass  # don't crash the push endpoint

@router.get("/tm/data")
async def get_tm_data():
    """Get latest True Markets data pushed by frontend."""
    age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else -1
    return {**_tm_data, "age_seconds": round(age, 1)}


# ─── Health ──────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}
