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
from app.data.tm_mcp_client import fetch_price_history as mcp_price_history, fetch_asset_summary
from app.data.onchain_daily import daily_loop as onchain_daily_loop
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None
_data_refresh_task = None
_onchain_task = None

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
    """Every 30s: pull fresh BTC price from TrueMarkets MCP (no auth needed).
    Fast path: 1h/5m ticks for latest price.
    Slow path: 1d/1h history for 24h chart + 7d cache.
    """
    _iter = 0
    while True:
        try:
            # Fast path: 5m ticks
            tick = await mcp_price_history("BTC", "1h", "5m")
            tick_points = tick.get("results", [{}])[0].get("points", [])
            if tick_points:
                _tm_data["price"] = float(tick_points[-1]["price"])
                _tm_data["updated"] = time.time()

            # Slow path: 1d/1h chart + cache (every 5th iter ~2.5 min)
            if _iter % 5 == 0:
                ph = await mcp_price_history("BTC", "1d", "1h")
                points = ph.get("results", [{}])[0].get("points", [])
                if points:
                    _write_price_cache(points, "1d", "1h")
                    _tm_data["chart"] = _points_to_chart(points)

                # Also refresh 7d history for technical signals
                try:
                    ph7 = await mcp_price_history("BTC", "7d", "1h")
                    pts7 = ph7.get("results", [{}])[0].get("points", [])
                    if pts7:
                        _write_price_cache(pts7, "7d", "1h")
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"MCP price refresh failed: {e}")
        try:
            _cache.clear()
            _chart_cache.clear()
        except Exception:
            pass
        _iter += 1
        await asyncio.sleep(30)


import re

_TM_PLACEHOLDER_RE = re.compile(r"\{\{\s*[a-zA-Z_]+\s*:\s*([^}]+?)\s*\}\}")


def _strip_tm_placeholders(text: str) -> str:
    """Replace {{token:BTC}} -> BTC, {{price:$77,000}} -> $77,000, etc."""
    if not text:
        return text
    return _TM_PLACEHOLDER_RE.sub(r"\1", text)


async def _mcp_refresh_loop():
    """Every 5 min: refresh sentiment from TM MCP asset summary (REST has no equivalent)."""
    while True:
        try:
            summary = await fetch_asset_summary("BTC")
            if summary:
                _tm_data["sentiment"] = summary.get("sentiment", _tm_data.get("sentiment", "neutral"))
                _tm_data["summary"] = _strip_tm_placeholders(
                    summary.get("body", _tm_data.get("summary", ""))
                )
        except Exception as e:
            logger.warning(f"MCP sentiment refresh failed: {e}")
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _data_refresh_task, _onchain_task
    logger.info("Prediction engine started — 6 signals, price 30s, sentiment 5m, on-chain 24h")
    _refresh_task = asyncio.create_task(_refresh_loop())
    _data_refresh_task = asyncio.create_task(_mcp_refresh_loop())
    _onchain_task = asyncio.create_task(onchain_daily_loop())
    yield
    for task in [_refresh_task, _data_refresh_task, _onchain_task]:
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
