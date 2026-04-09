import re
import asyncio
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.ensemble import EnsemblePredictionEngine
from app.data.coingecko import fetch_current_price, fetch_historical_prices
from app.data.fear_greed import fetch_fear_greed
from app.data.onchain import fetch_onchain_metrics
from app.data.social_sentiment import fetch_social_sentiment
from app.data.order_flow import fetch_order_flow
from app.data.polymarket import fetch_polymarket_markets
from app.data import truemarkets
from app.config import SUPPORTED_COINS

router = APIRouter()

_cache: dict = {}
CACHE_TTL = 300


def _get_cached(key: str):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
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
            fetch_historical_prices(cfg["coingecko_id"], days=90),
            fetch_social_sentiment(coin, cfg.get("subreddits")),
            fetch_fear_greed(limit=30),
            fetch_onchain_metrics(),
            return_exceptions=True,
        )

        if isinstance(historical, Exception):
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

        _set_cached(f"predictions:{coin}", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
            fetch_current_price(cfg["coingecko_id"]),
            fetch_fear_greed(limit=7),
            fetch_onchain_metrics(),
            fetch_social_sentiment(coin, cfg.get("subreddits")),
            return_exceptions=True,
        )

        if isinstance(price_data, Exception):
            raise HTTPException(status_code=502, detail="Failed to fetch price data")

        result = {
            "price": price_data["price"],
            "change_24h": price_data["change_24h"],
            "market_cap": price_data["market_cap"],
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
    cached = _get_cached("market-stats:bitcoin")
    if cached:
        return cached

    try:
        price_data, fear_greed = await asyncio.gather(
            _fetch_detailed_btc_stats(),
            fetch_fear_greed(limit=1),
            return_exceptions=True,
        )

        if isinstance(price_data, Exception):
            raise HTTPException(status_code=502, detail="Failed to fetch market stats")

        fg = fear_greed if not isinstance(fear_greed, Exception) else {"current": {"value": 50}}

        result = {**price_data, "fear_greed": fg}
        _set_cached("market-stats:bitcoin", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _fetch_detailed_btc_stats() -> dict:
    """Fetch detailed BTC stats from CoinGecko."""
    import httpx
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin",
            params={"localization": "false", "tickers": "false", "community_data": "false", "developer_data": "false"},
        )
        resp.raise_for_status()
        data = resp.json()

    md = data.get("market_data", {})
    return {
        "price": md.get("current_price", {}).get("usd", 0),
        "change_24h_pct": md.get("price_change_percentage_24h", 0),
        "change_24h_usd": md.get("price_change_24h", 0),
        "market_cap": md.get("market_cap", {}).get("usd", 0),
        "volume_24h": md.get("total_volume", {}).get("usd", 0),
        "high_24h": md.get("high_24h", {}).get("usd", 0),
        "low_24h": md.get("low_24h", {}).get("usd", 0),
        "ath": md.get("ath", {}).get("usd", 0),
        "atl": md.get("atl", {}).get("usd", 0),
        "circulating_supply": md.get("circulating_supply", 0),
        "max_supply": md.get("max_supply", 0),
        "total_supply": md.get("total_supply", 0),
        "price_change_7d": md.get("price_change_percentage_7d", 0),
        "price_change_30d": md.get("price_change_percentage_30d", 0),
        "price_change_1y": md.get("price_change_percentage_1y", 0),
    }


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
    """Price chart data for BTC."""
    if days not in CHART_PERIODS:
        days = "1"

    # Check chart-specific cache (longer TTL)
    cache_ttl = 600 if days in ("1", "5") else 3600
    if days in _chart_cache:
        data, ts = _chart_cache[days]
        if time.time() - ts < cache_ttl:
            return data

    import httpx as _httpx
    cg_days = _ytd_days() if days == "ytd" else CHART_PERIODS[days]
    try:
        async with _httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": cg_days},
            )
            if resp.status_code == 429:
                # Rate limited — return cached if available, else empty
                if days in _chart_cache:
                    return _chart_cache[days][0]
                return {"prices": [], "days": days}
            resp.raise_for_status()
            data = resp.json()

        # Downsample large datasets to ~200 points for performance
        raw_prices = data.get("prices", [])
        if len(raw_prices) > 300:
            step = max(1, len(raw_prices) // 200)
            raw_prices = raw_prices[::step] + [raw_prices[-1]]

        prices = [[p[0], round(p[1], 2)] for p in raw_prices]
        result = {"prices": prices, "days": days}
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
                "lstm": prediction["model_signals"]["lstm"].get(threshold_str, 0.5),
                "xgboost": prediction["model_signals"]["xgboost"].get(threshold_str, 0.5),
                "sentiment": prediction["model_signals"]["sentiment"].get(threshold_str, 0.5),
            },
        })

    # Sort: strongest mispricings first
    signals.sort(key=lambda s: abs(s["diff"]) if s["diff"] is not None else 0, reverse=True)

    # ── Generate single recommended trade ──
    recommended = await _build_recommendation(
        signals, prediction, cfg, polymarket_markets, order_flow
    )

    result = {
        "coin": coin,
        "symbol": cfg["symbol"],
        "current_price": prediction["current_price"],
        "confidence": prediction["confidence"],
        "sentiment_signal": prediction["sentiment_signal"],
        "indicators": prediction["indicators"],
        "signals": signals,
        "polymarket_count": len(polymarket_markets),
        "order_flow": order_flow,
        "recommended_trade": recommended,
    }

    _set_cached(f"mispricing:{coin}", result)
    return result


async def _build_recommendation(signals: list, prediction: dict, cfg: dict, polymarket_markets: list, order_flow: dict | None = None) -> dict:
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

    # ── 1. Model direction: compare up vs down probabilities ──
    up_signals = [s for s in signals if s["direction"] == "up"]
    down_signals = [s for s in signals if s["direction"] == "down"]
    # Use the nearest threshold as the most relevant
    nearest_up = max(up_signals, key=lambda s: s["our_prob"]) if up_signals else None
    nearest_down = max(down_signals, key=lambda s: s["our_prob"]) if down_signals else None
    best_up_prob = nearest_up["our_prob"] if nearest_up else 0
    best_down_prob = nearest_down["our_prob"] if nearest_down else 0

    # Model leans toward whichever direction has higher probability
    if best_up_prob > best_down_prob:
        model_direction = "bullish"
    elif best_down_prob > best_up_prob:
        model_direction = "bearish"
    else:
        model_direction = "neutral"

    # ── 2. Polymarket analysis: look at BOTH sides ──
    has_poly = any(s["poly_prob"] is not None for s in signals)
    poly_up = [s for s in up_signals if s["poly_prob"] is not None]
    poly_down = [s for s in down_signals if s["poly_prob"] is not None]

    # Is Polymarket pricing high volatility in both directions?
    poly_up_avg = sum(s["poly_prob"] for s in poly_up) / max(len(poly_up), 1) if poly_up else 0
    poly_down_avg = sum(s["poly_prob"] for s in poly_down) / max(len(poly_down), 1) if poly_down else 0
    poly_both_high = poly_up_avg > 0.4 and poly_down_avg > 0.4

    # ── 3. Fear & Greed: fear = bearish, greed = bullish ──
    if fg <= 20:
        fg_signal = "strongly_bearish"
    elif fg <= 40:
        fg_signal = "bearish"
    elif fg >= 80:
        fg_signal = "strongly_bullish"
    elif fg >= 60:
        fg_signal = "bullish"
    else:
        fg_signal = "neutral"

    # ── 4. Score: accumulate bullish/bearish evidence ──
    # Positive = bullish, negative = bearish
    # Model is primary signal (highest weight)
    score = 0.0
    reasons = []

    # Model direction (dominant weight — this is what we built the model for)
    if best_up_prob > 0 or best_down_prob > 0:
        model_ratio = best_up_prob / max(best_up_prob + best_down_prob, 0.01)
        # model_ratio: 0=fully bearish, 0.5=neutral, 1=fully bullish
        score += (model_ratio - 0.5) * 8  # strong: ±4 range

    # Fear & Greed (secondary: fear = bearish, greed = bullish)
    if fg_signal == "strongly_bearish":
        score -= 1.5
    elif fg_signal == "bearish":
        score -= 0.75
    elif fg_signal == "strongly_bullish":
        score += 1.5
    elif fg_signal == "bullish":
        score += 0.75

    # Order flow (from Polymarket microstructure + True Markets orders)
    of = order_flow or {}
    of_signal = of.get("combined_signal", 0)
    of_pressure = of.get("pressure", "neutral")
    if of_signal != 0:
        score += of_signal * 2  # ±2 range, same tier as FG

    # Model probability context (lead with this — it's the primary signal)
    if nearest_up and nearest_down:
        up_t = int(float(nearest_up["threshold"]))
        down_t = int(float(nearest_down["threshold"]))
        up_pct = int(best_up_prob * 100)
        down_pct = int(best_down_prob * 100)
        if best_up_prob > best_down_prob:
            ratio = best_up_prob / max(best_down_prob, 0.01)
            reasons.insert(0, f"Our model predicts upward movement is {ratio:.1f}x more likely — {up_pct}% chance of reaching ${up_t:,} vs {down_pct}% chance of dropping to ${down_t:,}")
        elif best_down_prob > best_up_prob:
            ratio = best_down_prob / max(best_up_prob, 0.01)
            reasons.insert(0, f"Our model predicts downward movement is {ratio:.1f}x more likely — {down_pct}% chance of dropping to ${down_t:,} vs {up_pct}% chance of reaching ${up_t:,}")
        else:
            reasons.insert(0, f"Our model sees equal probability — {up_pct}% up to ${up_t:,}, {down_pct}% down to ${down_t:,}")

    # Preliminary side decision (model + FG so far, RSI/sentiment added below)
    side = "buy" if score >= 0 else "sell"

    # Polymarket analysis — only include bullets that support the recommendation
    if has_poly:
        if side == "buy" and poly_up_avg > poly_down_avg:
            reasons.append(f"Polymarket leans bullish — {int(poly_up_avg*100)}% upside vs {int(poly_down_avg*100)}% downside")
        elif side == "sell" and poly_down_avg > poly_up_avg:
            reasons.append(f"Polymarket leans bearish — {int(poly_down_avg*100)}% downside vs {int(poly_up_avg*100)}% upside")

        if nearest_up and nearest_down:
            if side == "buy" and poly_up:
                nearest_poly_up = min(poly_up, key=lambda s: float(s["threshold"]))
                poly_up_pct = int(nearest_poly_up["poly_prob"] * 100)
                our_up_pct = int(nearest_poly_up["our_prob"] * 100)
                t = int(float(nearest_poly_up["threshold"]))
                if poly_up_pct >= our_up_pct:
                    reasons.append(f"Polymarket also sees upside — {poly_up_pct}% chance of reaching ${t:,}")
            elif side == "sell" and poly_down:
                nearest_poly_down = min(poly_down, key=lambda s: abs(float(s["threshold"]) - current_price))
                poly_down_pct = int(nearest_poly_down["poly_prob"] * 100)
                our_down_pct = int(nearest_poly_down["our_prob"] * 100)
                t = int(float(nearest_poly_down["threshold"]))
                if poly_down_pct >= our_down_pct:
                    reasons.append(f"Polymarket also sees downside — {poly_down_pct}% chance of dropping to ${t:,}")

    # Order flow reason — only if it agrees
    poly_details = of.get("polymarket_flow", {}).get("details", {})
    if of_pressure in ("strong_buy", "buy") and side == "buy":
        vol_detail = f"${poly_details.get('up_volume_24h', 0):,.0f} flowing into upside bets vs ${poly_details.get('down_volume_24h', 0):,.0f} into downside" if poly_details.get("up_volume_24h") else ""
        reasons.append(f"Order flow shows buy pressure — {vol_detail}" if vol_detail else "Order flow shows buy-side pressure")
    elif of_pressure in ("strong_sell", "sell") and side == "sell":
        vol_detail = f"${poly_details.get('down_volume_24h', 0):,.0f} flowing into downside bets vs ${poly_details.get('up_volume_24h', 0):,.0f} into upside" if poly_details.get("down_volume_24h") else ""
        reasons.append(f"Order flow shows sell pressure — {vol_detail}" if vol_detail else "Order flow shows sell-side pressure")

    # RSI — only show if it agrees
    if rsi > 70:
        score -= 1
        if side == "sell":
            reasons.append(f"RSI at {rsi:.0f} — overbought, pullback likely")
    elif rsi < 30:
        score += 1
        if side == "buy":
            reasons.append(f"RSI at {rsi:.0f} — oversold, bounce likely")

    # Overall sentiment — only include if it agrees with the side
    sent_text = sentiment.get("overall_signal", "Neutral")
    sent_lower = sent_text.lower()
    if (side == "buy" and "bullish" in sent_lower) or (side == "sell" and "bearish" in sent_lower):
        reasons.append(f"Overall sentiment: {sent_text}")

    # ── Final decision (re-evaluate after all signals) ──
    side = "buy" if score >= 0 else "sell"

    # Find the strongest single signal for the "based_on" field
    actionable = [s for s in signals if s["diff"] is not None and abs(s["diff"]) > 0.10]
    strongest = actionable[0] if actionable else None

    # Get quote — use real market price, with True Markets API for execution
    quote = None
    try:
        quote_data = await truemarkets.get_quote(
            base_asset=base_asset,
            quote_asset="USD",
            side=side,
            qty="1",
            qty_unit="base",
        )
        api_price = float(quote_data.get("price", "0"))
        # Use current market price if API returns stale/mock data
        price = current_price if abs(api_price - current_price) / current_price > 0.05 else api_price
        quote = {
            "price": str(round(price, 2)),
            "qty": "1",
            "total": str(round(price, 2)),
        }
    except Exception:
        # Fallback to current market price
        quote = {
            "price": str(round(current_price, 2)),
            "qty": "1",
            "total": str(round(current_price, 2)),
        }

    return {
        "side": side,
        "symbol": symbol,
        "base_asset": base_asset,
        "confidence": prediction["confidence"],
        "reasons": reasons,
        "based_on_mispricing": strongest is not None,
        "strongest_signal": {
            "threshold": strongest["threshold"],
            "diff": strongest["diff"],
            "signal": strongest["signal"],
        } if strongest else None,
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


# ─── Health ──────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}
