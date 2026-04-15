"""
API Routes — Rebuilt prediction page, preserved market page endpoints.

Prediction endpoint: GET /api/prediction/bitcoin
  Computes 6 signals in parallel, aggregates with backtested weights.
  Returns buy/sell recommendation with per-signal reasoning.
"""

import os
import re
import asyncio
import time
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.data.truemarkets_mcp import fetch_current_price, fetch_historical_prices
from app.data.fear_greed import fetch_fear_greed
from app.data.polymarket import fetch_polymarket_thresholds
from app.data.order_flow import fetch_binance_order_flow
from app.data import truemarkets
from app.models.lightgbm_model import LightGBMPredictor
from app.models.signals import (
    compute_polymarket_signal, compute_order_flow_signal,
    compute_lightgbm_signal, compute_technical_signal,
    compute_sentiment_signal, compute_fear_greed_signal,
    aggregate_signals,
)
from app.models.feature_engineering import build_features, FEATURE_NAMES
from app.config import SUPPORTED_COINS, BACKTEST_RESULTS_PATH, SIGNAL_WEIGHTS_PATH

router = APIRouter()

# ─── Caching ────────────────────────────────────────────

_cache: dict = {}
CACHE_TTL = 15
CACHE_TTL_FAST = 10
CACHE_TTL_SLOW = 60


def _get_cached(key: str, ttl: int | None = None):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < (ttl or CACHE_TTL):
            return data
    return None


def _get_stale(key: str):
    if key in _cache:
        return _cache[key][0]
    return None


def _set_cached(key: str, data):
    _cache[key] = (data, time.time())


# ─── LightGBM Model (loaded once) ──────────────────────

_lgbm = LightGBMPredictor()


# ─── Coins ──────────────────────────────────────────────

@router.get("/coins")
async def list_coins():
    return {
        coin_id: {"symbol": cfg["symbol"], "base_asset": cfg["base_asset"]}
        for coin_id, cfg in SUPPORTED_COINS.items()
    }


# ─── Single Source of Truth: BTC Price ──────────────────
# ALL endpoints use this. One price, one source (TrueMarkets MCP).

_tm_data: dict = {"price": 0, "sentiment": "neutral", "summary": "", "trending": [], "surging": [], "chart": [], "updated": 0}


async def _get_btc_price() -> dict:
    cached = _get_cached("_btc_price_single", ttl=CACHE_TTL_FAST)
    if cached:
        return cached

    tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
    if tm_age < 180 and _tm_data["price"] > 0:
        change_24h = 0
        try:
            from app.data.truemarkets_mcp import fetch_detailed_btc_stats
            stats = await fetch_detailed_btc_stats()
            change_24h = stats.get("change_24h_pct", 0)
        except Exception:
            pass
        result = {"price": _tm_data["price"], "change_24h": change_24h, "volume_24h": 0, "timestamp": _tm_data["updated"], "source": "truemarkets"}
        _set_cached("_btc_price_single", result)
        return result

    try:
        data = await fetch_current_price("BTC")
        result = {"price": data["price"], "change_24h": data["change_24h"], "volume_24h": data["volume_24h"], "timestamp": time.time(), "source": "truemarkets"}
        _set_cached("_btc_price_single", result)
        return result
    except Exception:
        stale = _get_stale("_btc_price_single")
        if stale:
            return stale
        return {"price": 0, "change_24h": 0, "volume_24h": 0, "timestamp": 0, "source": "none"}


@router.get("/price/bitcoin")
async def get_fast_price():
    return await _get_btc_price()


# ═══════════════════════════════════════════════════════
# NEW PREDICTION ENDPOINT — 6 signals, weighted, with backtest
# ═══════════════════════════════════════════════════════

@router.get("/prediction/bitcoin")
async def get_prediction():
    """
    The single prediction endpoint. Computes 6 signals in parallel,
    aggregates with backtested weights, returns buy/sell recommendation.
    """
    cached = _get_cached("prediction:bitcoin")
    if cached:
        return cached

    try:
        # Fetch all data in parallel
        btc_price_task = _get_btc_price()
        poly_task = fetch_polymarket_thresholds()
        flow_task = fetch_binance_order_flow()
        hist_task = fetch_historical_prices("BTC", days=7)
        fg_task = fetch_fear_greed(limit=1)

        btc_price, poly_thresholds, order_flow, historical, fear_greed = await asyncio.gather(
            btc_price_task, poly_task, flow_task, hist_task, fg_task,
            return_exceptions=True,
        )

        # Handle failures gracefully
        if isinstance(btc_price, Exception):
            btc_price = {"price": 0, "change_24h": 0}
        if isinstance(poly_thresholds, Exception):
            poly_thresholds = []
        if isinstance(order_flow, Exception):
            order_flow = {"signal": 0, "pressure": "neutral", "buy_sell_ratio": 0.5, "imbalance": 0, "buy_volume": 0, "sell_volume": 0, "bid_depth": 0, "ask_depth": 0}
        if isinstance(fear_greed, Exception):
            fear_greed = {"current": {"value": 50, "classification": "Neutral"}}

        current_price = btc_price["price"]

        # ── Compute technical indicators from historical data ──
        rsi, macd_hist, bollinger_pos = 50.0, 0.0, 0.5
        lgbm_prob = 0.5

        if not isinstance(historical, Exception) and len(historical) > 20:
            rsi = float(historical["rsi"].iloc[-1]) if "rsi" in historical.columns else 50
            macd_val = float(historical["macd"].iloc[-1]) if "macd" in historical.columns else 0
            macd_sig = float(historical["macd_signal"].iloc[-1]) if "macd_signal" in historical.columns else 0
            macd_hist = macd_val - macd_sig

            # Bollinger position
            if "sma_20" in historical.columns:
                sma20 = float(historical["sma_20"].iloc[-1])
                vol20 = float(historical["volatility_20d"].iloc[-1]) if "volatility_20d" in historical.columns else 0.02
                bb_upper = sma20 + 2 * vol20 * sma20
                bb_lower = sma20 - 2 * vol20 * sma20
                bb_range = bb_upper - bb_lower
                bollinger_pos = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
            bollinger_pos = max(0, min(1, bollinger_pos))

            # ── LightGBM prediction ──
            try:
                feats = build_features(historical)
                last_features = {f: float(feats[f].iloc[-1]) if f in feats.columns else 0.0 for f in FEATURE_NAMES}
                lgbm_prob = _lgbm.predict(last_features)
            except Exception:
                lgbm_prob = 0.5

        # ── TrueMarkets sentiment ──
        tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
        sentiment_data = {}
        if tm_age < 300:
            sentiment_data = {"sentiment": _tm_data["sentiment"], "summary": _tm_data["summary"]}

        # ── Compute all 6 signals ──
        signals = [
            compute_polymarket_signal(poly_thresholds, current_price),
            compute_order_flow_signal(order_flow),
            compute_lightgbm_signal(lgbm_prob, _lgbm.get_accuracy()),
            compute_technical_signal(rsi, macd_hist, bollinger_pos),
            compute_sentiment_signal(sentiment_data),
            compute_fear_greed_signal(fear_greed),
        ]

        # ── Aggregate ──
        agg = aggregate_signals(signals)

        # ── Load backtest results ──
        backtest = {}
        if os.path.exists(BACKTEST_RESULTS_PATH):
            with open(BACKTEST_RESULTS_PATH) as f:
                backtest = json.load(f)

        weights = {}
        if os.path.exists(SIGNAL_WEIGHTS_PATH):
            with open(SIGNAL_WEIGHTS_PATH) as f:
                weights = json.load(f)

        # ── MACD line and signal for frontend ──
        macd_line = 0
        macd_signal_line = 0
        if not isinstance(historical, Exception) and len(historical) > 20:
            macd_line = float(historical["macd"].iloc[-1]) if "macd" in historical.columns else 0
            macd_signal_line = float(historical["macd_signal"].iloc[-1]) if "macd_signal" in historical.columns else 0

        result = {
            "current_price": current_price,
            "change_24h": btc_price.get("change_24h", 0),
            "recommended_side": agg["recommended_side"],
            "confidence": agg["confidence"],
            "weighted_strength": agg["weighted_strength"],
            "buy_signals": agg["buy_signals"],
            "sell_signals": agg["sell_signals"],
            "buy_count": agg["buy_count"],
            "sell_count": agg["sell_count"],
            "total_signals": agg["total_signals"],
            "polymarket_thresholds": poly_thresholds,
            "order_flow": order_flow,
            "technical_indicators": {
                "rsi": round(rsi, 1),
                "rsi_label": "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral",
                "macd_line": round(macd_line, 2),
                "macd_signal": round(macd_signal_line, 2),
                "macd_histogram": round(macd_hist, 2),
                "bollinger_position": round(bollinger_pos, 3),
                "bollinger_label": "Lower band" if bollinger_pos < 0.2 else "Upper band" if bollinger_pos > 0.8 else "Mid-range",
            },
            "sentiment_summary": sentiment_data.get("summary", ""),
            "fear_greed": fear_greed,
            "backtest_results": backtest,
            "weights": weights,
            "updated_at": time.time(),
        }

        _set_cached("prediction:bitcoin", result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ═══════════════════════════════════════════════════════
# KEPT ENDPOINTS (market page, chart, trade, tm/push)
# ═══════════════════════════════════════════════════════

# ─── Market Stats ───────────────────────────────────────

@router.get("/market-stats/bitcoin")
async def get_market_stats():
    cached = _get_cached("market-stats:bitcoin", ttl=CACHE_TTL_SLOW)
    if cached:
        return cached
    try:
        from app.data.truemarkets_mcp import fetch_detailed_btc_stats
        btc_price = await _get_btc_price()
        price_data, fg = await asyncio.gather(
            fetch_detailed_btc_stats(), fetch_fear_greed(limit=1), return_exceptions=True,
        )
        if isinstance(price_data, Exception):
            price_data = {"price": btc_price["price"], "change_24h_pct": btc_price["change_24h"],
                          "change_24h_usd": 0, "market_cap": 0, "volume_24h": 0,
                          "high_24h": 0, "low_24h": 0, "ath": 0, "atl": 0,
                          "circulating_supply": 0, "max_supply": 21000000,
                          "price_change_7d": 0, "price_change_30d": 0, "price_change_1y": 0}
        price_data["price"] = btc_price["price"]
        result = {**price_data, "fear_greed": fg if not isinstance(fg, Exception) else {"current": {"value": 50}},
                  "source": "truemarkets", "tm_sentiment": _tm_data["sentiment"], "tm_summary": _tm_data["summary"]}
        _set_cached("market-stats:bitcoin", result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chart ──────────────────────────────────────────────

_chart_cache: dict = {}

def _ytd_days() -> int:
    from datetime import date
    today = date.today()
    return (today - date(today.year, 1, 1)).days or 1

CHART_PERIODS = {"1": 1, "5": 5, "30": 30, "180": 180, "ytd": "ytd", "365": 365}

@router.get("/chart/bitcoin")
async def get_chart_data(days: str = "1"):
    if days not in CHART_PERIODS:
        days = "1"

    tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
    if days == "1" and tm_age < 180 and _tm_data["chart"]:
        result = {"prices": _tm_data["chart"], "days": "1", "source": "truemarkets"}
        _chart_cache["1"] = (result, time.time())
        return result

    cache_ttl = 600 if days in ("1", "5") else 3600
    if days in _chart_cache:
        data, ts = _chart_cache[days]
        if time.time() - ts < cache_ttl:
            return data

    from app.data.truemarkets_mcp import fetch_btc_price_history, CACHE_DIR, _iso_to_ms

    cg_days = _ytd_days() if days == "ytd" else CHART_PERIODS[days]
    if isinstance(cg_days, str):
        cg_days = 30

    if cg_days > 30:
        try:
            with open(os.path.join(CACHE_DIR, "btc_3Y_1d.json")) as f:
                hist = json.load(f)
            pts = hist["results"][0]["points"]
            pts = pts[-cg_days:] if cg_days < len(pts) else pts
            raw_prices = [[_iso_to_ms(p["t"]), float(p["price"])] for p in pts]
        except Exception:
            raw_prices = []
    else:
        tm_window_map = {"1": "1d", "5": "7d", "30": "1M"}
        tm_res_map = {"1": "1h", "5": "1h", "30": "1d"}
        raw_prices = await fetch_btc_price_history(window=tm_window_map.get(days, "1d"), resolution=tm_res_map.get(days, "1h"))

    try:
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


# ─── Legacy mispricing endpoint (redirect to new prediction) ──

@router.get("/mispricing/{coin}")
async def get_mispricing(coin: str = "bitcoin"):
    """Legacy endpoint — returns prediction data in old format for MarketView compatibility."""
    pred = await get_prediction()
    fg = pred.get("fear_greed", {})
    return {
        "coin": coin, "symbol": "BTC", "current_price": pred["current_price"],
        "change_24h_pct": pred.get("change_24h", 0), "change_24h_usd": 0,
        "confidence": pred["confidence"],
        "sentiment_signal": {
            "overall_signal": "Bullish" if pred["weighted_strength"] > 0.55 else "Bearish" if pred["weighted_strength"] < 0.45 else "Neutral",
            "fear_greed": fg.get("current", {}).get("classification", "Neutral"),
            "fear_greed_value": fg.get("current", {}).get("value", 50),
            "sentiment_score": 0, "bullish_ratio": 0.5,
        },
        "indicators": pred["technical_indicators"],
        "signals": [], "polymarket_count": len(pred.get("polymarket_thresholds", [])),
        "order_flow": pred["order_flow"],
        "recommended_trade": {
            "primary_side": pred["recommended_side"],
            "buy_case": {"reasons": [s["reason"] for s in pred["buy_signals"]], "vote_count": pred["buy_count"]},
            "sell_case": {"reasons": [s["reason"] for s in pred["sell_signals"]], "vote_count": pred["sell_count"]},
            "total_signals": pred["total_signals"], "confidence": pred["confidence"],
            "mode": "both", "symbol": "BTC", "base_asset": "BTC",
            "quote": {"price": str(round(pred["current_price"], 2)), "qty": "1", "total": str(round(pred["current_price"], 2))},
        },
    }


# ─── Trade (True Markets Gateway) ──────────────────────

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
        return await truemarkets.get_quote(base_asset=req.base_asset, quote_asset=req.quote_asset, side=req.side, qty=req.qty, qty_unit=req.qty_unit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Quote failed: {str(e)}")

@router.post("/trade/order")
async def place_trade_order(req: OrderRequest):
    try:
        return await truemarkets.place_order(base_asset=req.base_asset, quote_asset=req.quote_asset, side=req.side, qty=req.qty, order_type=req.order_type, price=req.price)
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


# ─── TrueMarkets MCP Data Ingest ───────────────────────

class TMDataPush(BaseModel):
    price: float = 0
    sentiment: str = "neutral"
    summary: str = ""
    trending: list = []
    surging: list = []
    chart: list = []

@router.post("/tm/push")
async def push_tm_data(data: TMDataPush):
    _tm_data["price"] = data.price
    _tm_data["sentiment"] = data.sentiment
    _tm_data["summary"] = data.summary
    _tm_data["trending"] = data.trending
    _tm_data["surging"] = data.surging
    if data.chart:
        _tm_data["chart"] = data.chart
        _update_price_cache_from_chart(data.chart)
    _tm_data["updated"] = time.time()
    _cache.clear()
    _chart_cache.clear()
    return {"status": "ok", "updated": _tm_data["updated"]}

def _update_price_cache_from_chart(chart_data: list):
    from app.data.truemarkets_mcp import CACHE_DIR
    if not chart_data or len(chart_data) < 5:
        return
    try:
        points = []
        for ts_ms, price in chart_data:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            points.append({"t": dt.strftime("%Y-%m-%dT%H:%M:%SZ"), "price": str(price)})
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "btc_1d_1h.json"), "w") as f:
            json.dump({"window": "1d", "resolution": "1h", "results": [{"symbol": "BTC", "points": points}]}, f)
    except Exception:
        pass

@router.get("/tm/data")
async def get_tm_data():
    age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else -1
    return {**_tm_data, "age_seconds": round(age, 1)}


# ─── Health ─────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0"}
