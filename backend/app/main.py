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


async def _data_refresh_loop():
    """Every 30 min: fetch fresh BTC price + sentiment from external APIs.
    This replaces the need for Claude/MCP to push data."""
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # 1. Fetch latest BTC price from BGeometrics OHLC
                try:
                    resp = await client.get(
                        "https://api.bitcoin-data.com/v1/btc-ohlc/last",
                        headers={"Authorization": f"Bearer {BGEOMETRICS_TOKEN}"},
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        price = float(data.get("close", 0))
                        if price > 0:
                            _tm_data["price"] = price
                            _tm_data["updated"] = time.time()
                            logger.info(f"[auto-refresh] BTC price: ${price:,.2f} from BGeometrics")

                            # Also update the cache file
                            import json, os
                            from datetime import datetime, timezone
                            cache_path = os.path.join(CACHE_DIR, "btc_1d_1h.json")
                            try:
                                with open(cache_path) as f:
                                    cache = json.load(f)
                                pts = cache["results"][0]["points"]
                                now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
                                # Append or update last point
                                if pts and pts[-1]["t"][:13] == now_iso[:13]:
                                    pts[-1]["price"] = str(price)
                                else:
                                    pts.append({"t": now_iso, "price": str(price)})
                                # Keep last 25 points
                                cache["results"][0]["points"] = pts[-25:]
                                with open(cache_path, "w") as f:
                                    json.dump(cache, f)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"[auto-refresh] BGeometrics price failed: {e}")

                # 2. Fetch TM sentiment via BGeometrics market summary or keep existing
                # (TM MCP sentiment stays until next push — no free API for it)

            _cache.clear()
            _chart_cache.clear()

        except Exception as e:
            logger.warning(f"[auto-refresh] Error: {e}")

        await asyncio.sleep(1800)  # 30 minutes


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _refresh_task, _data_refresh_task
    logger.info("Prediction engine started — 6 signals, auto-refresh every 30 min")
    _refresh_task = asyncio.create_task(_refresh_loop())
    _data_refresh_task = asyncio.create_task(_data_refresh_loop())
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
