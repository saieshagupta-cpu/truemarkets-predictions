import asyncio
import json
import os
import time
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price, CACHE_DIR
from app.data.tm_api_client import fetch_price_history as rest_price_history
from app.data.tm_mcp_client import fetch_asset_summary  # sentiment stays on MCP
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None
_data_refresh_task = None

BGEOMETRICS_TOKEN = "4KlmMZzF0B"


def _write_price_cache(points: list, window: str, resolution: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = f"btc_{window}_{resolution}.json"
    payload = {
        "window": window, "resolution": resolution,
        "results": [{"symbol": "BTC", "points": points}],
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(CACHE_DIR, fname), "w") as f:
        json.dump(payload, f)


def _points_to_chart(points: list) -> list:
    return [
        [int(time.mktime(time.strptime(p["t"], "%Y-%m-%dT%H:%M:%SZ"))) * 1000, float(p["price"])]
        for p in points
    ]


async def _refresh_loop():
    """Every 30s: pull fresh BTC price from TM REST API.
    Fast path: 5m/5s ticks for latest price.
    Slow path: 1d/1h history for 24h chart, written to on-disk cache.
    """
    _iter = 0
    loop = asyncio.get_event_loop()
    while True:
        try:
            # Fast path: ~1-min-old tick from 5m/5s series
            tick = await loop.run_in_executor(None, rest_price_history, "BTC", "5m", "5s")
            tick_points = tick.get("points", [])
            if tick_points:
                _tm_data["price"] = float(tick_points[-1]["price"])
                _tm_data["updated"] = time.time()

            # Slow path: 1d/1h chart + cache (every 5th iter ~2.5 min)
            if _iter % 5 == 0:
                ph = await loop.run_in_executor(None, rest_price_history, "BTC", "1d", "1h")
                points = ph.get("points", [])
                if points:
                    _write_price_cache(points, "1d", "1h")
                    _tm_data["chart"] = _points_to_chart(points)

                # Also refresh 7d (fear_greed / longer-term technical signals)
                try:
                    ph7 = await loop.run_in_executor(None, rest_price_history, "BTC", "7d", "1h")
                    pts7 = ph7.get("points", [])
                    if pts7:
                        _write_price_cache(pts7, "7d", "1h")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"TM REST price refresh failed: {e}")
        try:
            _cache.clear()
            _chart_cache.clear()
        except Exception:
            pass
        _iter += 1
        await asyncio.sleep(30)


async def _mcp_refresh_loop():
    """Every 5 min: refresh sentiment from TM MCP asset summary (REST has no equivalent)."""
    while True:
        try:
            summary = await fetch_asset_summary("BTC")
            if summary:
                _tm_data["sentiment"] = summary.get("sentiment", _tm_data.get("sentiment", "neutral"))
                _tm_data["summary"] = summary.get("body", _tm_data.get("summary", ""))
        except Exception as e:
            logger.warning(f"MCP sentiment refresh failed: {e}")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _data_refresh_task
    logger.info("Prediction engine started — 6 signals, auto-refresh every 30 min")
    _refresh_task = asyncio.create_task(_refresh_loop())
    _data_refresh_task = asyncio.create_task(_mcp_refresh_loop())
    yield
    for task in [_refresh_task, _data_refresh_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="True Markets Prediction Engine",
    description="6-signal BTC prediction with auto-refresh every 30 minutes",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "True Markets Prediction Engine v3",
        "model": "On-Chain Ensemble (67.4% accuracy, 30 Boruta features from BGeometrics)",
        "signals": ["Polymarket (20%)", "Binance Order Flow (15%)", "Our Model (20%)", "Technical (20%)", "TM Sentiment (10%)", "Fear & Greed (15%)"],
        "refresh": "30s endpoints, 30min data refresh",
        "deploy": "Railway + Vercel",
    }
