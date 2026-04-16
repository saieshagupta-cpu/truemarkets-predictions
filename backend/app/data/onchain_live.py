"""
Live on-chain data fetcher from BGeometrics API.
Fetches latest values + computes rate-of-change features for model input.
Cached for 5 minutes (data is daily).
"""

import os
import time
import json
import numpy as np
import pandas as pd
import httpx

BGEOMETRICS_TOKEN = "4KlmMZzF0B"
BASE_URL = "https://api.bitcoin-data.com/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "onchain")

_cache: dict = {}
_cache_ts: float = 0
CACHE_TTL = 300  # 5 minutes


async def fetch_live_onchain() -> dict:
    """Fetch latest on-chain values + rate-of-change features for model input."""
    global _cache, _cache_ts

    if time.time() - _cache_ts < CACHE_TTL and _cache:
        return _cache

    # Load the merged CSV and use the last 14 days for rate-of-change computation
    try:
        merged_path = os.path.join(CACHE_DIR, "onchain_merged.csv")
        if not os.path.exists(merged_path):
            return {}

        df = pd.read_csv(merged_path)
        df["date"] = pd.to_datetime(df["date"])

        # Fix dupes
        cols = list(df.columns)
        seen = {}
        for i, c in enumerate(cols):
            if c in seen: seen[c] += 1; cols[i] = f"{c}_{seen[c]}"
            else: seen[c] = 0
        df.columns = cols

        # Convert any non-numeric column to numeric (covers numpy str, pandas object, mixed).
        # Use pd.to_numeric first; if that fails, ordinal-encode the string values.
        for col in df.columns:
            if col == "date":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any():
                df[col] = coerced.fillna(0.0)
            else:
                uniques = sorted(str(u) for u in df[col].dropna().unique())
                df[col] = df[col].astype(str).map({v: i for i, v in enumerate(uniques)}).fillna(0)

        exclude = ["date", "price_open", "price_high", "price_low", "price_close", "price_volume"]
        base_cols = [c for c in df.columns if c not in exclude]

        # Get latest row as dict
        latest = df.iloc[-1]
        result = {}
        for col in base_cols:
            try:
                result[col] = float(latest[col]) if pd.notna(latest[col]) else 0.0
            except (ValueError, TypeError):
                result[col] = 0.0

        # Rate-of-change features from last 14 days
        for col in base_cols:
            try:
                vals = df[col].values.astype(float)
            except (ValueError, TypeError):
                continue
            current = vals[-1]
            for lag in [1, 3, 7, 14]:
                if len(vals) > lag:
                    prev = vals[-1 - lag]
                    denom = prev if abs(prev) > 1e-10 else 1
                    result[f"{col}_chg{lag}d"] = (current - prev) / denom
                else:
                    result[f"{col}_chg{lag}d"] = 0.0

        # Price-derived features
        prices = df["price_close"].values.astype(float)
        log_ret = np.diff(np.log(np.maximum(prices[-21:], 1)))
        result["log_return"] = float(log_ret[-1]) if len(log_ret) > 0 else 0.0
        result["volatility_5d"] = float(np.std(log_ret[-5:])) if len(log_ret) >= 5 else 0.0
        result["volatility_20d"] = float(np.std(log_ret)) if len(log_ret) >= 10 else 0.0
        result["momentum_5d"] = float((prices[-1] - prices[-6]) / prices[-6]) if len(prices) > 6 else 0.0
        result["momentum_20d"] = float((prices[-1] - prices[-21]) / prices[-21]) if len(prices) > 21 else 0.0

        # RSI
        deltas = np.diff(prices[-15:])
        gains = np.mean([d for d in deltas if d > 0]) if any(d > 0 for d in deltas) else 0
        losses = np.mean([-d for d in deltas if d < 0]) if any(d < 0 for d in deltas) else 0.001
        result["rsi"] = 100 - (100 / (1 + gains / losses))

        _cache = result
        _cache_ts = time.time()
        return result

    except Exception as e:
        print(f"[onchain_live] Error: {e}")
        return _cache if _cache else {}
