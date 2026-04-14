"""
True Markets data layer — cache-only, no direct API calls.

Data sources:
  1. True Markets MCP tools → populate local cache files (prices, sentiment, market data)
  2. CryptoCompare free API → 3-year daily OHLCV for training (btc_3Y_1d.json)
  3. Alternative.me → Fear & Greed index (via fear_greed.py)
  4. Polymarket Gamma API → order flow and market probabilities (via polymarket.py)
  5. Reddit → social sentiment via VADER (via social_sentiment.py)
  6. Blockchain.info → on-chain metrics (via onchain.py)

Cache files (app/data/cache/):
  btc_3Y_1d.json   — 3 years daily OHLCV from CryptoCompare (training)
  btc_1M_1d.json   — 1 month daily from True Markets MCP
  btc_7d_1h.json   — 7 days hourly from True Markets MCP
  btc_7d_1d.json   — 7 days daily from True Markets MCP
  btc_1d_1h.json   — 24 hours from True Markets MCP (or frontend push)
"""

import json
import os
import time
import numpy as np
import pandas as pd

from app.config import TRUEMARKETS_API_BASE, TRUEMARKETS_KEY_FILE

TM_API_BASE = TRUEMARKETS_API_BASE
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ─── JWT auth (kept for future direct API use) ───────────

_jwk_cache: dict | None = None


def _load_jwk() -> dict:
    global _jwk_cache
    if _jwk_cache is None and TRUEMARKETS_KEY_FILE:
        try:
            with open(TRUEMARKETS_KEY_FILE) as f:
                _jwk_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _jwk_cache = {}
    return _jwk_cache or {}


def _make_jwt() -> str:
    try:
        import jwt as pyjwt
    except ImportError:
        return ""
    jwk_data = _load_jwk()
    if not jwk_data:
        return ""
    key_id = jwk_data["key_id"]
    private_key = jwk_data["private_key"]
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": key_id, "iat": now, "exp": now + 300},
        pyjwt.algorithms.ECAlgorithm.from_jwk(json.dumps(private_key)),
        algorithm="ES256", headers={"kid": key_id, "alg": "ES256"},
    )
    return token


def _headers() -> dict:
    h = {"Accept": "application/json"}
    token = _make_jwt()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# ─── Local cache (primary data source) ───────────────────

def _load_cache(symbol: str, window: str, resolution: str) -> dict | None:
    cache_file = os.path.join(CACHE_DIR, f"{symbol.lower()}_{window}_{resolution}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_cache(symbol: str, window: str, resolution: str, data: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol.lower()}_{window}_{resolution}.json")
    data["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(cache_file, "w") as f:
        json.dump(data, f)


def _points_from_cache_or_api(data: dict) -> list:
    results = data.get("results", [])
    if results and results[0].get("points"):
        return results[0]["points"]
    return []


async def _fetch_price_data(symbol: str, window: str, resolution: str) -> dict:
    """Fetch from local cache only. No direct API calls."""
    cached = _load_cache(symbol, window, resolution)
    if cached:
        points = _points_from_cache_or_api(cached)
        if points:
            return cached
    return {"results": []}


# ─── Price functions ──────────────────────────────────────

COIN_ID_MAP = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL",
    "hyperliquid": "HYPE", "coinbase-wrapped-btc": "CBBTC",
}


def _coin_id_to_symbol(coin_id: str) -> str:
    return COIN_ID_MAP.get(coin_id.lower(), coin_id.upper())


def _iso_to_ms(iso_str: str) -> int:
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


async def fetch_current_price(symbol: str = "BTC") -> dict:
    """Get current price from cache."""
    symbol = _coin_id_to_symbol(symbol)
    data = await _fetch_price_data(symbol, "1d", "1h")
    points = _points_from_cache_or_api(data)
    if points:
        latest = points[-1]
        first = points[0]
        price = float(latest["price"])
        first_price = float(first["price"])
        change = ((price - first_price) / first_price * 100) if first_price > 0 else 0
        return {"price": price, "change_24h": round(change, 2), "market_cap": 0, "volume_24h": 0}
    raise Exception(f"No price data for {symbol} — refresh cache via MCP tools")


async def fetch_historical_prices(coin_id: str = "bitcoin", days: int = 30) -> pd.DataFrame:
    """Fetch historical prices with technical indicators from cache."""
    symbol = _coin_id_to_symbol(coin_id)
    if days <= 1:
        window, resolution = "1d", "1h"
    elif days <= 7:
        window, resolution = "7d", "1h"
    else:
        window, resolution = "1M", "1d"

    data = await _fetch_price_data(symbol, window, resolution)
    points = _points_from_cache_or_api(data)
    if not points:
        return pd.DataFrame()

    df = pd.DataFrame([{"timestamp": p["t"], "price": float(p["price"])} for p in points])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["volume"] = 0
    df = _add_technical_indicators(df)
    return df


async def fetch_btc_price_history(window: str = "1d", resolution: str = "5m") -> list[list]:
    """Get BTC chart data from cache. Returns [[timestamp_ms, price], ...]"""
    data = await _fetch_price_data("BTC", window, resolution)
    points = _points_from_cache_or_api(data)
    if points:
        return [[_iso_to_ms(p["t"]), float(p["price"])] for p in points]
    return []


async def fetch_btc_summary() -> dict:
    """Placeholder — TM sentiment comes via MCP push, not direct API."""
    return {}


async def fetch_market_summary() -> dict:
    """Placeholder — market data comes via MCP push, not direct API."""
    return {}


async def fetch_detailed_btc_stats() -> dict:
    """Compute detailed stats from cached price history."""
    # 24h data
    data_1d = await _fetch_price_data("BTC", "1d", "1h")
    points_1d = _points_from_cache_or_api(data_1d)

    price, change_24h_pct, high_24h, low_24h = 0, 0, 0, 0
    if points_1d:
        price = float(points_1d[-1]["price"])
        first_price = float(points_1d[0]["price"])
        if first_price > 0:
            change_24h_pct = ((price - first_price) / first_price) * 100
        all_prices = [float(p["price"]) for p in points_1d]
        high_24h = max(all_prices)
        low_24h = min(all_prices)

    # 7d change
    data_7d = await _fetch_price_data("BTC", "7d", "1d")
    points_7d = _points_from_cache_or_api(data_7d)
    change_7d = 0
    if points_7d and price > 0:
        p0 = float(points_7d[0]["price"])
        if p0 > 0:
            change_7d = ((price - p0) / p0) * 100

    # 30d change
    data_1m = await _fetch_price_data("BTC", "1M", "1d")
    points_1m = _points_from_cache_or_api(data_1m)
    change_30d = 0
    if points_1m and price > 0:
        p0 = float(points_1m[0]["price"])
        if p0 > 0:
            change_30d = ((price - p0) / p0) * 100

    btc_supply = 19_850_000
    return {
        "price": price,
        "change_24h_pct": round(change_24h_pct, 2),
        "change_24h_usd": round(price * change_24h_pct / 100, 2),
        "market_cap": round(price * btc_supply),
        "volume_24h": 0,
        "high_24h": high_24h,
        "low_24h": low_24h,
        "ath": 109000,
        "atl": 67.81,
        "circulating_supply": btc_supply,
        "max_supply": 21000000,
        "total_supply": 0,
        "price_change_7d": round(change_7d, 2),
        "price_change_30d": round(change_30d, 2),
        "price_change_1y": 0,
    }


# ─── Technical indicators ────────────────────────────────

def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 2:
        return df

    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    sma20 = df["price"].rolling(window=20).mean()
    std20 = df["price"].rolling(window=20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bollinger_position"] = (df["price"] - df["bb_lower"]) / bb_range

    df["sma_7"] = df["price"].rolling(window=7).mean()
    df["sma_30"] = df["price"].rolling(window=min(30, len(df))).mean()
    df["sma_50"] = df["price"].rolling(window=min(50, len(df))).mean()

    df["volume_change"] = df["volume"].pct_change().fillna(0)
    df["returns"] = df["price"].pct_change()
    df["volatility_20d"] = df["returns"].rolling(window=min(20, len(df))).std()
    df["volatility_5d"] = df["returns"].rolling(window=min(5, len(df))).std()
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_20d"].replace(0, np.nan)

    for lag in [1, 3, 7, 14, 30]:
        if len(df) > lag:
            df[f"return_{lag}d"] = df["price"].pct_change(lag)
        else:
            df[f"return_{lag}d"] = 0

    df["relative_volume"] = 1.0
    df["rsi_momentum"] = df["rsi"].diff(min(5, len(df) - 1))

    win = min(20, len(df))
    high_20d = df["price"].rolling(window=win).max()
    low_20d = df["price"].rolling(window=win).min()
    price_range = (high_20d - low_20d).replace(0, np.nan)
    df["price_position"] = (df["price"] - low_20d) / price_range

    df = df.ffill().bfill()
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df
