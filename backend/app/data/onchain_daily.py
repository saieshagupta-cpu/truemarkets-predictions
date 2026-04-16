"""
Daily BGeometrics on-chain data refresher.

Appends today's row to onchain_merged.csv once per day. Respects the free-tier
rate limit (8 requests/hour) by staggering calls.

Only fetches endpoints the model actually uses (19 base columns + 3 HODL bands + OHLC).
Full history is never re-downloaded — just the latest value per endpoint.
"""

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

logger = logging.getLogger("truemarkets.onchain_daily")

BGEOMETRICS_TOKEN = os.getenv("BGEOMETRICS_TOKEN", "4KlmMZzF0B")
BASE_URL = "https://api.bitcoin-data.com/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "onchain")
MERGED_CSV = os.path.join(CACHE_DIR, "onchain_merged.csv")

# Minimum endpoint set to feed the 19 base columns the Boruta model selected.
# Each entry: (BGeometrics endpoint, CSV column name, JSON field that holds the value)
ENDPOINTS = [
    ("asopr/last",                  "adjusted_sopr",             "asopr"),
    ("average-dormancy/last",       "avg_dormancy",              "averageDormancy"),
    ("asol/last",                   "avg_spent_output_lifespan", "asol"),
    ("exchange-outflow-btc/last",   "exchange_outflow",          "exchangeOutflowBtc"),
    ("mvrv/last",                   "mvrv_ratio",                "mvrv"),
    ("puell-multiple/last",         "puell_multiple",            "puellMultiple"),
    ("reserve-risk/last",           "reserve_risk",              "reserveRisk"),
    ("utxos-in-profit-pct/last",    "utxos_profit_pct",          "utxosInProfitPct"),
    ("supply-profit/last",          "supply_in_profit",          "supplyProfit"),
    ("supply-loss/last",            "supply_in_loss",            "supplyLoss"),
    ("difficulty-btc/last",         "difficulty",                "difficultyBtc"),
    ("hashribbons/last",            "hash_ribbons_raw",          None),  # special: sma_30 + sma_60
    ("hodl-waves-supply/last",      "hodl_waves_raw",            None),  # special: age_3m_6m, age_5y_7y, age_10y
    ("btc-ohlc/last",               "ohlc_raw",                  None),  # special: OHLCV
]

# Premium tier: 600 req/hour → 6s between requests is the safe floor. 0.5s uses <1% of budget.
# With 14 endpoints → ~7 seconds per full refresh.
REQUEST_INTERVAL_SECONDS = 0.5
RUN_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


async def _fetch_one(endpoint: str) -> dict | None:
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Authorization": f"Bearer {BGEOMETRICS_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, headers=headers)
            if r.status_code == 200:
                return r.json()
            logger.warning(f"BGeometrics {endpoint}: HTTP {r.status_code}")
            return None
    except Exception as e:
        logger.warning(f"BGeometrics {endpoint}: {e}")
        return None


def _extract_value(data: dict, field: str) -> float:
    """Pull a value out of a BGeometrics /last response."""
    if not data:
        return 0.0
    val = data.get(field)
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _extract_date(data: dict) -> str | None:
    """Return YYYY-MM-DD, stripping any time component (e.g. OHLC returns '2026-04-16 02:00:00')."""
    if not data:
        return None
    raw = data.get("d") or data.get("date")
    if not raw:
        return None
    return str(raw).split(" ")[0].split("T")[0]


async def _fetch_all() -> dict | None:
    """Fetch all endpoints with rate-limit-safe spacing. Returns merged row dict."""
    row: dict = {}
    latest_date: str | None = None

    for i, (endpoint, col_name, json_field) in enumerate(ENDPOINTS):
        if i > 0:
            await asyncio.sleep(REQUEST_INTERVAL_SECONDS)

        data = await _fetch_one(endpoint)
        if not data:
            continue

        d = _extract_date(data)
        if d and (latest_date is None or d > latest_date):
            latest_date = d

        # Special handlers for multi-field endpoints
        if col_name == "hash_ribbons_raw":
            # Response shape: {d, sma_30, sma_60, hashribbons}
            row["hash_ribbons"] = _extract_value(data, "sma_30")       # 1st column
            row["hash_ribbons.1"] = _extract_value(data, "sma_60")     # 2nd column
        elif col_name == "hodl_waves_raw":
            # Response shape: {d, age_0d_1d, age_1d_1w, ..., age_10y}
            for age_key in ["age_3m_6m", "age_5y_7y", "age_10y"]:
                if age_key in data:
                    row[f"hodl_{age_key}"] = _extract_value(data, age_key)
        elif col_name == "ohlc_raw":
            # Response shape: {d, open, high, low, close, volume}
            for k in ["open", "high", "low", "close", "volume"]:
                row[f"price_{k}"] = _extract_value(data, k)
        else:
            row[col_name] = _extract_value(data, json_field)

    if not latest_date:
        logger.warning("BGeometrics: no successful fetches, skipping append")
        return None

    row["date"] = latest_date
    return row


def _append_if_new(row: dict) -> bool:
    """Append row to onchain_merged.csv if its date is newer than the last row.
    Returns True if appended."""
    if not os.path.exists(MERGED_CSV):
        logger.warning(f"{MERGED_CSV} does not exist — skipping append")
        return False

    df = pd.read_csv(MERGED_CSV)
    if df.empty:
        return False

    last_date = str(df["date"].iloc[-1]).split("T")[0]
    new_date = str(row.get("date", "")).split("T")[0]

    if not new_date or new_date <= last_date:
        logger.info(f"BGeometrics row date={new_date} not newer than CSV last_date={last_date}, skipping")
        return False

    # Derive pct_supply_in_profit the same way download_onchain.py does
    sp = row.get("supply_in_profit", 0.0)
    sl = row.get("supply_in_loss", 0.0)
    total = sp + sl
    row["pct_supply_in_profit"] = sp / total if total > 0 else 0.0

    # Build a row matching the CSV's schema (fill missing with forward-fill from last row)
    last_row = df.iloc[-1].to_dict()
    new_row = {}
    for col in df.columns:
        if col in row:
            new_row[col] = row[col]
        else:
            new_row[col] = last_row.get(col, 0.0)  # forward-fill unchanged columns

    df_new = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df_new.to_csv(MERGED_CSV, index=False)
    logger.info(f"Appended BGeometrics row for {new_date} → {MERGED_CSV}")
    return True


async def run_daily_refresh() -> bool:
    """Execute one full daily fetch cycle. Returns True if a new row was appended."""
    logger.info("Starting BGeometrics daily refresh...")
    row = await _fetch_all()
    if row:
        return _append_if_new(row)
    return False


async def daily_loop():
    """Long-running task — call once at startup, then once every 24 hours."""
    # On startup: only run if CSV is stale by >20 hours
    try:
        if os.path.exists(MERGED_CSV):
            df = pd.read_csv(MERGED_CSV)
            if not df.empty:
                last_date = pd.to_datetime(str(df["date"].iloc[-1])).date()
                today = datetime.now(timezone.utc).date()
                days_behind = (today - last_date).days
                if days_behind < 1:
                    logger.info(f"onchain_merged.csv is fresh (last: {last_date}), skipping initial refresh")
                    await asyncio.sleep(RUN_INTERVAL_SECONDS)
    except Exception as e:
        logger.warning(f"Startup freshness check failed: {e}")

    while True:
        try:
            await run_daily_refresh()
            # Also invalidate the in-memory onchain cache so next prediction re-reads CSV
            try:
                from app.data import onchain_live
                onchain_live._cache = {}
                onchain_live._cache_ts = 0
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Daily refresh failed: {e}")
        await asyncio.sleep(RUN_INTERVAL_SECONDS)
