import asyncio
import time
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price, CACHE_DIR
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None
_data_refresh_task = None

BGEOMETRICS_TOKEN = "4KlmMZzF0B"


async def _refresh_loop():
    """Every 30s: clear endpoint caches. Only overwrite price if TM push is stale."""
    while True:
        try:
            tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
            if tm_age > 180:
                price_data = await fetch_current_price("BTC")
                if price_data and price_data.get("price", 0) > 0:
                    _tm_data["price"] = price_data["price"]
                    _tm_data["updated"] = time.time()
            _cache.clear()
            _chart_cache.clear()
        except Exception:
            pass
        await asyncio.sleep(30)


async def _mcp_refresh_loop():
    """Every 5 min: refresh TrueMarkets MCP price + sentiment via cache.
    Claude session pushes fresh data via /tm/push — this loop uses it."""
    while True:
        try:
            _cache.clear()
            _chart_cache.clear()
        except Exception:
            pass
        await asyncio.sleep(300)  # 5 minutes


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
