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

    # ── New features for improved GBM ──

    # Lagged returns
    for lag in [1, 3, 7, 14, 30]:
        df[f"return_{lag}d"] = df["price"].pct_change(lag)

    # Relative volume: current vs 20-day average
    vol_avg_20 = df["volume"].rolling(window=20).mean()
    df["relative_volume"] = df["volume"] / vol_avg_20.replace(0, np.nan)

    # Volatility ratio: 5d vol / 20d vol (expanding vs contracting)
    df["volatility_5d"] = df["returns"].rolling(window=5).std()
    df["volatility_ratio"] = df["volatility_5d"] / df["volatility_20d"].replace(0, np.nan)

    # RSI momentum: how RSI is changing
    df["rsi_momentum"] = df["rsi"].diff(5)

    # Price position: where price sits in 20-day range (0=low, 1=high)
    high_20d = df["price"].rolling(window=20).max()
    low_20d = df["price"].rolling(window=20).min()
    price_range = (high_20d - low_20d).replace(0, np.nan)
    df["price_position"] = (df["price"] - low_20d) / price_range

    # Bollinger position: normalized within bands (0=lower, 1=upper)
    bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bollinger_position"] = (df["price"] - df["bb_lower"]) / bb_range

    return df.dropna().reset_index(drop=True)
