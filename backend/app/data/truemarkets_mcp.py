"""
True Markets data layer.
Replaces CoinGecko for live prices, charts, sentiment, and market data.

Data flow:
  1. Primary: Local cache files in app/data/cache/ (populated via Claude MCP tools)
  2. Fallback: Direct API calls to api.truemarkets.co (may be blocked by Cloudflare)
  3. Frontend push: Live data pushed to /tm/push endpoint by the frontend

To refresh cached data, ask Claude: "refresh truemarkets price cache"
(uses the TrueMarkets MCP tools which bypass Cloudflare).
"""

import json
import os
import time
import httpx
import numpy as np
import pandas as pd

from app.config import TRUEMARKETS_API_BASE, TRUEMARKETS_KEY_FILE

TM_API_BASE = TRUEMARKETS_API_BASE
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ─── JWT auth (ES256) ────────────────────────────────────

_jwk_cache: dict | None = None


def _load_jwk() -> dict:
    """Load the JWK from the key file (cached)."""
    global _jwk_cache
    if _jwk_cache is None and TRUEMARKETS_KEY_FILE:
        try:
            with open(TRUEMARKETS_KEY_FILE) as f:
                _jwk_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _jwk_cache = {}
    return _jwk_cache or {}


def _make_jwt() -> str:
    """Create a short-lived JWT signed with the ES256 private key."""
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
    payload = {
        "sub": key_id,
        "iat": now,
        "exp": now + 300,  # 5 min expiry
    }
    headers = {"kid": key_id, "alg": "ES256"}

    token = pyjwt.encode(
        payload,
        pyjwt.algorithms.ECAlgorithm.from_jwk(json.dumps(private_key)),
        algorithm="ES256",
        headers=headers,
    )
    return token


def _headers() -> dict:
    """Auth headers for True Markets API with JWT."""
    h = {"Accept": "application/json"}
    token = _make_jwt()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# ─── Local cache ─────────────────────────────────────────


def _load_cache(symbol: str, window: str, resolution: str) -> dict | None:
    """Load cached price data from local JSON file."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol.lower()}_{window}_{resolution}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _save_cache(symbol: str, window: str, resolution: str, data: dict):
    """Save price data to local JSON cache file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol.lower()}_{window}_{resolution}.json")
    data["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(cache_file, "w") as f:
        json.dump(data, f)


def _points_from_cache_or_api(data: dict) -> list:
    """Extract points from TM API response format."""
    results = data.get("results", [])
    if results and results[0].get("points"):
        return results[0]["points"]
    return []


async def _fetch_price_data(symbol: str, window: str, resolution: str) -> dict:
    """
    Fetch price data: try local cache first, then API.
    Returns raw TM API response format.
    """
    # 1. Check local cache
    cached = _load_cache(symbol, window, resolution)
    if cached:
        points = _points_from_cache_or_api(cached)
        if points:
            return cached

    # 2. Try direct API (may fail due to Cloudflare)
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(
                f"{TM_API_BASE}/v1/prices/history",
                params={"symbols": symbol, "window": window, "resolution": resolution},
                headers=_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                _save_cache(symbol, window, resolution, data)
                return data
    except Exception:
        pass

    # 3. Return whatever cache we have (even if stale)
    if cached:
        return cached

    return {"results": []}


# ─── CoinGecko-compatible interface ──────────────────────


async def fetch_current_price(symbol: str = "BTC") -> dict:
    """
    Get current price data. Drop-in replacement for coingecko.fetch_current_price.
    Returns: {price, change_24h, market_cap, volume_24h}
    """
    symbol = _coin_id_to_symbol(symbol)

    data = await _fetch_price_data(symbol, "1d", "1h")
    points = _points_from_cache_or_api(data)

    if points:
        latest = points[-1]
        first = points[0]
        price = float(latest["price"])
        first_price = float(first["price"])
        change_24h = ((price - first_price) / first_price * 100) if first_price > 0 else 0

        return {
            "price": price,
            "change_24h": round(change_24h, 2),
            "market_cap": 0,
            "volume_24h": 0,
        }

    raise Exception(f"No price data for {symbol} — run 'refresh truemarkets price cache' via Claude")


async def fetch_historical_prices(
    coin_id: str = "bitcoin", days: int = 30
) -> pd.DataFrame:
    """
    Fetch historical prices with technical indicators.
    Drop-in replacement for coingecko.fetch_historical_prices.
    """
    symbol = _coin_id_to_symbol(coin_id)

    # Map days to TM window parameter
    if days <= 1:
        window, resolution = "1d", "5m"
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
    df["volume"] = 0  # TM price history doesn't include volume

    df = _add_technical_indicators(df)
    return df


# ─── TM-specific endpoints ──────────────────────────────


async def fetch_btc_price() -> dict:
    """Get current BTC price."""
    return await fetch_current_price("BTC")


async def fetch_btc_price_history(window: str = "1d", resolution: str = "5m") -> list[list]:
    """Get BTC price chart data. Returns [[timestamp_ms, price], ...]"""
    data = await _fetch_price_data("BTC", window, resolution)
    points = _points_from_cache_or_api(data)
    if points:
        return [[_iso_to_ms(p["t"]), float(p["price"])] for p in points]
    return []


async def fetch_btc_summary() -> dict:
    """Get AI-generated BTC summary with sentiment from True Markets."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(
                f"{TM_API_BASE}/v1/assets/summary",
                params={"symbol": "BTC"},
                headers=_headers(),
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


async def fetch_market_summary() -> dict:
    """Get overall market summary from True Markets."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            resp = await client.get(
                f"{TM_API_BASE}/v1/market/summary",
                headers=_headers(),
            )
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


async def fetch_historical_for_indicators(window: str = "1M") -> pd.DataFrame:
    """Fetch price history and compute technical indicators for inference."""
    data = await _fetch_price_data("BTC", window, "1d")
    points = _points_from_cache_or_api(data)

    if not points:
        return pd.DataFrame()

    df = pd.DataFrame([{"timestamp": p["t"], "price": float(p["price"])} for p in points])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["volume"] = 0

    df = _add_technical_indicators(df)
    return df


async def fetch_detailed_btc_stats() -> dict:
    """
    Fetch detailed BTC stats. Replaces CoinGecko detailed stats.
    Returns the same shape dict that routes.py expects.
    """
    # 24h data
    data_1d = await _fetch_price_data("BTC", "1d", "1h")
    points_1d = _points_from_cache_or_api(data_1d)

    price = 0
    change_24h_pct = 0
    high_24h = 0
    low_24h = 0

    if points_1d:
        price = float(points_1d[-1]["price"])
        first_price = float(points_1d[0]["price"])
        if first_price > 0:
            change_24h_pct = ((price - first_price) / first_price) * 100
        prices = [float(p["price"]) for p in points_1d]
        high_24h = max(prices)
        low_24h = min(prices)

    # 7d change
    data_7d = await _fetch_price_data("BTC", "7d", "1d")
    points_7d = _points_from_cache_or_api(data_7d)
    change_7d = 0
    if points_7d:
        p0 = float(points_7d[0]["price"])
        if p0 > 0:
            change_7d = ((price - p0) / p0) * 100

    # 30d change
    data_1m = await _fetch_price_data("BTC", "1M", "1d")
    points_1m = _points_from_cache_or_api(data_1m)
    change_30d = 0
    if points_1m:
        p0 = float(points_1m[0]["price"])
        if p0 > 0:
            change_30d = ((price - p0) / p0) * 100

    # BTC circulating supply ~19.85M (slowly increasing, updated periodically)
    btc_supply = 19_850_000
    market_cap = price * btc_supply

    return {
        "price": price,
        "change_24h_pct": round(change_24h_pct, 2),
        "change_24h_usd": round(price * change_24h_pct / 100, 2),
        "market_cap": round(market_cap),
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


# ─── Helpers ─────────────────────────────────────────────


COIN_ID_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "hyperliquid": "HYPE",
    "coinbase-wrapped-btc": "CBBTC",
}


def _coin_id_to_symbol(coin_id: str) -> str:
    """Convert CoinGecko-style coin ID to TM symbol. Pass through if already a symbol."""
    return COIN_ID_MAP.get(coin_id.lower(), coin_id.upper())


def _iso_to_ms(iso_str: str) -> int:
    """Convert ISO timestamp to milliseconds."""
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on a price DataFrame."""
    if len(df) < 2:
        return df

    # RSI
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["price"].ewm(span=12, adjust=False).mean()
    ema26 = df["price"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = df["price"].rolling(window=20).mean()
    std20 = df["price"].rolling(window=20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bollinger_position"] = (df["price"] - df["bb_lower"]) / bb_range

    # Simple Moving Averages
    df["sma_7"] = df["price"].rolling(window=7).mean()
    df["sma_30"] = df["price"].rolling(window=min(30, len(df))).mean()
    df["sma_50"] = df["price"].rolling(window=min(50, len(df))).mean()

    # Volume change
    df["volume_change"] = df["volume"].pct_change().fillna(0)

    # Returns & Volatility
    df["returns"] = df["price"].pct_change()
    df["volatility_20d"] = df["returns"].rolling(window=min(20, len(df))).std()
    df["volatility_5d"] = df["returns"].rolling(window=min(5, len(df))).std()
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_20d"].replace(0, np.nan)

    # Lagged returns
    for lag in [1, 3, 7, 14, 30]:
        if len(df) > lag:
            df[f"return_{lag}d"] = df["price"].pct_change(lag)
        else:
            df[f"return_{lag}d"] = 0

    # Relative volume
    df["relative_volume"] = 1.0  # no volume data from TM price API

    # RSI momentum
    df["rsi_momentum"] = df["rsi"].diff(min(5, len(df) - 1))

    # Price position in 20-day range
    win = min(20, len(df))
    high_20d = df["price"].rolling(window=win).max()
    low_20d = df["price"].rolling(window=win).min()
    price_range = (high_20d - low_20d).replace(0, np.nan)
    df["price_position"] = (df["price"] - low_20d) / price_range

    # Forward-fill then drop only rows where price is NaN
    df = df.ffill().bfill()
    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df
