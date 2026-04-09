import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.config import COINGECKO_BASE, LOOKBACK_DAYS


async def fetch_current_price(coin_id: str = "bitcoin") -> dict:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/simple/price",
            params={
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            },
        )
        resp.raise_for_status()
        data = resp.json()[coin_id]
        return {
            "price": data["usd"],
            "change_24h": data.get("usd_24h_change", 0),
            "market_cap": data.get("usd_market_cap", 0),
            "volume_24h": data.get("usd_24h_vol", 0),
        }


async def fetch_historical_prices(
    coin_id: str = "bitcoin", days: int = LOOKBACK_DAYS
) -> pd.DataFrame:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(
            f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

    prices = data["prices"]
    volumes = data["total_volumes"]
    market_caps = data["market_caps"]

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["volume"] = [v[1] for v in volumes[: len(df)]]
    df["market_cap"] = [m[1] for m in market_caps[: len(df)]]

    df = _add_technical_indicators(df)
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    # Simple Moving Averages
    df["sma_7"] = df["price"].rolling(window=7).mean()
    df["sma_30"] = df["price"].rolling(window=30).mean()
    df["sma_50"] = df["price"].rolling(window=50).mean()

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    # Volatility (20-day rolling std of returns)
    df["returns"] = df["price"].pct_change()
    df["volatility_20d"] = df["returns"].rolling(window=20).std()

    return df.dropna().reset_index(drop=True)
