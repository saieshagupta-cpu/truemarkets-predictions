import asyncio
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None


async def _refresh_loop():
    """Background: refresh BTC price + clear caches every 30 seconds.
    Does NOT overwrite TM push data if it's fresh (< 180s old)."""
    while True:
        try:
            # Only update price from cache if TM push data is stale
            tm_age = time.time() - _tm_data["updated"] if _tm_data["updated"] > 0 else 999
            if tm_age > 180:  # TM push is stale, use cache
                price_data = await fetch_current_price("BTC")
                if price_data and price_data.get("price", 0) > 0:
                    _tm_data["price"] = price_data["price"]
                    _tm_data["updated"] = time.time()
            # Always clear endpoint caches so they re-fetch
            _cache.clear()
            _chart_cache.clear()
        except Exception:
            pass
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task
    logger.info("Prediction engine started — 6 signals, GradientBoosting model loaded")
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="True Markets Prediction Engine",
    description="6-signal BTC prediction: Polymarket, Binance Order Flow, GradientBoosting, Technical, Sentiment, Fear & Greed",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "True Markets Prediction Engine v3",
        "model": "GradientBoosting (35 features, 3-day horizon, 52.6% test accuracy)",
        "signals": ["Polymarket (20%)", "Binance Order Flow (15%)", "GradientBoosting (20%)", "Technical (20%)", "TM Sentiment (10%)", "Fear & Greed (15%)"],
        "refresh": "every 30 seconds",
    }
