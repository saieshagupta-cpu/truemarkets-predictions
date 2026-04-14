import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price, _save_cache, _fetch_price_data
from app.config import FRONTEND_URL

# ─── Background price refresh ────────────────────────────
# Fetches fresh BTC price every 30 seconds from TrueMarkets cache/API.
# Updates _tm_data so all endpoints serve consistent, fresh data.
# Clears caches so predictions recompute with new prices.

_refresh_task = None


async def _refresh_loop():
    """Background loop: refresh BTC price and clear caches every 30 seconds."""
    while True:
        try:
            # Fetch fresh price from TrueMarkets (cache file or API)
            price_data = await fetch_current_price("BTC")
            if price_data and price_data.get("price", 0) > 0:
                import time
                _tm_data["price"] = price_data["price"]
                _tm_data["updated"] = time.time()

                # Also refresh 1d/1h cache for TCN
                try:
                    await _fetch_price_data("BTC", "1d", "1h")
                except Exception:
                    pass

                # Clear all endpoint caches so they recompute
                _cache.clear()
                _chart_cache.clear()

        except Exception:
            pass

        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background refresh on startup, stop on shutdown."""
    global _refresh_task
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
    description="TCN direction prediction + backtested signal ensemble for BTC",
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
        "model": "TCN (Temporal Convolutional Network)",
        "signals": ["Technical (40%)", "TCN (30%)", "Order Flow (20%)", "Sentiment (10%)"],
        "refresh": "every 30 seconds",
        "docs": "/docs",
    }
