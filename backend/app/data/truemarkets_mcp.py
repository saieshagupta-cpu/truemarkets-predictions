"""
True Markets MCP data layer.
Replaces CoinGecko for live prices, charts, sentiment, and market data.
Uses the True Markets MCP tools for all real-time data.

Note: For multi-year training data, blockchain.info is still used
(True Markets MCP only provides ~1 month of history).
"""

import httpx
import numpy as np
import pandas as pd

# True Markets MCP API base (the MCP tools call this internally,
# but we also need direct HTTP for the backend server)
TM_API_BASE = "https://api.truemarkets.co"


async def fetch_btc_price() -> dict:
    """Get current BTC price from True Markets price history (latest point)."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        # Use the price history with 1h window, 5m resolution for recent price
        resp = await client.get(
            f"{TM_API_BASE}/v1/prices/history",
            params={"symbols": "BTC", "window": "1h", "resolution": "5m"},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results and results[0].get("points"):
                points = results[0]["points"]
                latest = points[-1]
                first = points[0]
                price = float(latest["price"])
                first_price = float(first["price"])
                change_1h = ((price - first_price) / first_price * 100) if first_price > 0 else 0
                return {"price": price, "change_1h": round(change_1h, 2)}

    # Fallback
    return {"price": 0, "change_1h": 0}


async def fetch_btc_price_history(window: str = "1d", resolution: str = "5m") -> list[list]:
    """Get BTC price chart data. Returns [[timestamp_ms, price], ...]"""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(
            f"{TM_API_BASE}/v1/prices/history",
            params={"symbols": "BTC", "window": window, "resolution": resolution},
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results and results[0].get("points"):
                points = results[0]["points"]
                return [[_iso_to_ms(p["t"]), float(p["price"])] for p in points]
    return []


async def fetch_btc_summary() -> dict:
    """Get AI-generated BTC summary with sentiment from True Markets."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(f"{TM_API_BASE}/v1/assets/summary", params={"symbol": "BTC"})
        if resp.status_code == 200:
            return resp.json()
    return {}


async def fetch_market_summary() -> dict:
    """Get overall market summary from True Markets."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(f"{TM_API_BASE}/v1/market/summary")
        if resp.status_code == 200:
            return resp.json()
    return {}


async def fetch_historical_for_indicators(window: str = "1M") -> pd.DataFrame:
    """
    Fetch price history and compute technical indicators.
    Used for the ensemble model at inference time.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(
            f"{TM_API_BASE}/v1/prices/history",
            params={"symbols": "BTC", "window": window, "resolution": "1d"},
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
        results = data.get("results", [])
        if not results or not results[0].get("points"):
            return pd.DataFrame()

    points = results[0]["points"]
    df = pd.DataFrame([{"timestamp": p["t"], "price": float(p["price"])} for p in points])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # We don't have volume from the MCP, so use price-derived features only
    df["volume"] = 0  # placeholder

    # Technical indicators
    delta = df["price"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

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

    df["returns"] = df["price"].pct_change()
    df["volatility_20d"] = df["returns"].rolling(window=20).std()
    df["volatility_5d"] = df["returns"].rolling(window=5).std()
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_20d"].replace(0, np.nan)
    df["volume_change"] = 0  # no volume data from MCP

    for lag in [1, 3, 7, 14, 30]:
        if len(df) > lag:
            df[f"return_{lag}d"] = df["price"].pct_change(lag)
        else:
            df[f"return_{lag}d"] = 0

    df["relative_volume"] = 1.0  # no volume
    df["rsi_momentum"] = df["rsi"].diff(5)

    h20 = df["price"].rolling(window=20).max()
    l20 = df["price"].rolling(window=20).min()
    price_range = (h20 - l20).replace(0, np.nan)
    df["price_position"] = (df["price"] - l20) / price_range

    return df.dropna().reset_index(drop=True)


def _iso_to_ms(iso_str: str) -> int:
    """Convert ISO timestamp to milliseconds."""
    from datetime import datetime
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)
