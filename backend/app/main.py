import asyncio
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price, _fetch_price_data
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None


async def _refresh_loop():
    """Background loop: refresh BTC price from cache and clear endpoint caches every 30 seconds."""
    while True:
        try:
            price_data = await fetch_current_price("BTC")
            if price_data and price_data.get("price", 0) > 0:
                _tm_data["price"] = price_data["price"]
                _tm_data["updated"] = time.time()
                _cache.clear()
                _chart_cache.clear()
        except Exception:
            pass
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background refresh on startup."""
    global _refresh_task
    logger.info("TCN model trained on 3 years daily data (loaded from saved weights)")
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
    description="TCN next-day direction prediction + backtested signal ensemble for BTC",
    version="2.0.0",
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
        "name": "True Markets Prediction Engine",
        "model": "TCN trained on 3 years daily BTC (58.6% test accuracy)",
        "prediction": "Next-day BTC direction",
        "signals": ["RSI (24%)", "TCN (30%)", "MACD (16%)", "Order Flow (20%)", "Sentiment (10%)"],
        "refresh": "every 30 seconds",
    }
