"""
Live on-chain data fetcher from BGeometrics API.
Fetches latest values for CNN-LSTM model features.
Cached for 5 minutes (data is daily).
"""

import time
import httpx

BGEOMETRICS_TOKEN = "4KlmMZzF0B"
BASE_URL = "https://api.bitcoin-data.com/v1"

# Map feature names to BGeometrics endpoints
FEATURE_ENDPOINTS = {
    "exchange_netflow": "exchange-netflow-btc",
    "nvt_ratio": "nvt-ratio",
    "puell_multiple": "puell-multiple",
    "realized_loss_usd": "realized-loss-usd",
    "hodl_age_0d_1d": None,  # From hodl-waves-supply
    "hodl_age_1d_1w": None,
    "hodl_age_1w_1m": None,
    "hodl_age_1m_3m": None,
    "hodl_age_3m_6m": None,
    "hodl_age_6m_1y": None,
    "hodl_age_1y_2y": None,
    "hodl_age_2y_3y": None,
}

_cache: dict = {}
_cache_ts: float = 0
CACHE_TTL = 300  # 5 minutes


async def fetch_live_onchain() -> dict:
    """Fetch latest on-chain values for CNN-LSTM model input."""
    global _cache, _cache_ts

    if time.time() - _cache_ts < CACHE_TTL and _cache:
        return _cache

    result = {}
    headers = {"Authorization": f"Bearer {BGEOMETRICS_TOKEN}"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Fetch individual metrics
            for feat, endpoint in FEATURE_ENDPOINTS.items():
                if endpoint is None:
                    continue
                try:
                    resp = await client.get(f"{BASE_URL}/{endpoint}/last", headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        # Extract the value (second non-date field)
                        for k, v in data.items():
                            if k not in ("d", "unixTs"):
                                result[feat] = float(v) if v is not None else 0.0
                                break
                except Exception:
                    result[feat] = 0.0

            # Fetch HODL waves (one call, all age bands)
            try:
                resp = await client.get(f"{BASE_URL}/hodl-waves-supply/last", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    hodl_map = {
                        "age_0d_1d": "hodl_age_0d_1d",
                        "age_1d_1w": "hodl_age_1d_1w",
                        "age_1w_1m": "hodl_age_1w_1m",
                        "age_1m_3m": "hodl_age_1m_3m",
                        "age_3m_6m": "hodl_age_3m_6m",
                        "age_6m_1y": "hodl_age_6m_1y",
                        "age_1y_2y": "hodl_age_1y_2y",
                        "age_2y_3y": "hodl_age_2y_3y",
                    }
                    for api_key, feat_name in hodl_map.items():
                        if api_key in data:
                            result[feat_name] = float(data[api_key])
            except Exception:
                pass

            # Fetch hash ribbons
            try:
                resp = await client.get(f"{BASE_URL}/hashribbons/last", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    result["hash_ribbons"] = float(data.get("sma30", 0))
                    result["hash_ribbons_1"] = float(data.get("sma60", 0))
                    hr_val = data.get("hashribbons", "Up")
                    result["hash_ribbons_2"] = 1.0 if hr_val == "Up" else 0.0
            except Exception:
                pass

    except Exception as e:
        print(f"[onchain_live] Error: {e}")

    _cache = result
    _cache_ts = time.time()
    return result
