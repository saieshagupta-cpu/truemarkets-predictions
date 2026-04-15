"""
Download on-chain data from BGeometrics API (Premium).
Fetches all Boruta-selected features from Omole & Enke (2024) paper.
Merges into single CSV for CNN-LSTM training.

Usage: python train/download_onchain.py
"""

import os
import sys
import time
import io
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BGEOMETRICS_TOKEN = "4KlmMZzF0B"
BASE_URL = "https://api.bitcoin-data.com/v1"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "data", "cache", "onchain")

# All endpoints to download — matching paper's Boruta-selected features
ENDPOINTS = {
    # On-chain indicators
    "asopr": "adjusted_sopr",
    "average-dormancy": "avg_dormancy",
    "asol": "avg_spent_output_lifespan",
    "cdd": "coin_days_destroyed",
    "supply-adjusted-cdd": "supply_adjusted_cdd",
    "exchange-netflow-btc": "exchange_netflow",
    "exchange-inflow-btc": "exchange_inflow",
    "exchange-outflow-btc": "exchange_outflow",
    "mvrv": "mvrv_ratio",
    "mvrv-zscore": "mvrv_zscore",
    "nupl": "nupl",
    "nvt-ratio": "nvt_ratio",
    "active-addresses": "active_addresses",
    "puell-multiple": "puell_multiple",
    "reserve-risk": "reserve_risk",
    "utxos-in-profit-pct": "utxos_profit_pct",
    "utxos-in-loss-pct": "utxos_loss_pct",
    "supply-profit": "supply_in_profit",
    "supply-loss": "supply_in_loss",
    "nrpl-btc": "net_realized_pl",
    "realized-loss-usd": "realized_loss_usd",
    "rpv": "rpv_ratio",
    "liveliness": "liveliness",
    "difficulty-btc": "difficulty",
    "hashribbons": "hash_ribbons",
    # HODL waves (one CSV, 13 age band columns)
    "hodl-waves-supply": "hodl_waves",
    # Price
    "btc-ohlc": "ohlc",
}


def download_csv(endpoint: str, name: str) -> pd.DataFrame | None:
    """Download CSV from BGeometrics and parse into DataFrame."""
    url = f"{BASE_URL}/{endpoint}/csv"
    headers = {"Authorization": f"Bearer {BGEOMETRICS_TOKEN}"}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            print(f"  SKIP {name}: HTTP {resp.status_code}")
            return None

        text = resp.text.strip()
        if not text or len(text) < 10:
            print(f"  SKIP {name}: empty response")
            return None

        df = pd.read_csv(io.StringIO(text))
        if "d" in df.columns:
            df["d"] = pd.to_datetime(df["d"]).dt.normalize()  # Strip time component
            df = df.rename(columns={"d": "date"})
        if "unixTs" in df.columns:
            df = df.drop(columns=["unixTs"])

        # Save individual CSV
        csv_path = os.path.join(CACHE_DIR, f"{name}.csv")
        df.to_csv(csv_path, index=False)

        print(f"  OK   {name:30s} {len(df):6d} rows, {len(df.columns)-1} features")
        return df

    except Exception as e:
        print(f"  ERR  {name}: {e}")
        return None


def main():
    print("=" * 60)
    print("Downloading BGeometrics On-Chain Data")
    print(f"Token: {BGEOMETRICS_TOKEN[:4]}...")
    print(f"Cache: {CACHE_DIR}")
    print("=" * 60)

    os.makedirs(CACHE_DIR, exist_ok=True)
    all_dfs = {}
    delay = 0.5  # 600 req/h = 6 sec buffer, but 0.5s is fine

    for endpoint, name in ENDPOINTS.items():
        df = download_csv(endpoint, name)
        if df is not None and "date" in df.columns:
            all_dfs[name] = df
        time.sleep(delay)

    print(f"\n  Downloaded {len(all_dfs)}/{len(ENDPOINTS)} endpoints")

    # ── Merge all on date ──────────────────────────────────
    print(f"\n  Merging datasets...")

    # Start with OHLC as base (has date + price)
    if "ohlc" not in all_dfs:
        print("  ERROR: OHLC data missing, cannot merge")
        return

    merged = all_dfs["ohlc"].copy()
    merged = merged.rename(columns={
        "open": "price_open", "high": "price_high",
        "low": "price_low", "close": "price_close",
        "volume": "price_volume",
    })

    for name, df in all_dfs.items():
        if name == "ohlc":
            continue

        # Rename columns to avoid collisions
        rename_map = {}
        for col in df.columns:
            if col == "date":
                continue
            if name == "hodl_waves":
                rename_map[col] = f"hodl_{col}"
            else:
                rename_map[col] = name

        df_renamed = df.rename(columns=rename_map)
        # LEFT join: keep only dates where OHLC exists
        merged = pd.merge(merged, df_renamed, on="date", how="left")

    # Sort by date
    merged = merged.sort_values("date").reset_index(drop=True)

    # Filter to last 5 years for training
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=5*365)
    merged_5y = merged[merged["date"] >= cutoff].copy()

    # Forward-fill missing values (MNAR strategy from paper)
    merged_5y = merged_5y.ffill()

    # Drop rows with any remaining NaN (MCAR)
    before = len(merged_5y)
    merged_5y = merged_5y.dropna(subset=["price_close"])
    after = len(merged_5y)

    # Compute derived features
    # Drawdown from ATH
    merged_5y["drawdown_from_ath"] = merged_5y["price_close"] / merged_5y["price_close"].cummax() - 1

    # Percent supply in profit (if supply columns exist)
    if "supply_in_profit" in merged_5y.columns and "supply_in_loss" in merged_5y.columns:
        total_supply = merged_5y["supply_in_profit"] + merged_5y["supply_in_loss"]
        merged_5y["pct_supply_in_profit"] = merged_5y["supply_in_profit"] / total_supply.replace(0, 1)

    print(f"  Merged: {len(merged_5y)} rows, {len(merged_5y.columns)} columns")
    print(f"  Date range: {merged_5y['date'].iloc[0].date()} to {merged_5y['date'].iloc[-1].date()}")
    print(f"  Dropped {before - after} rows with missing price")

    # Fill remaining NaNs with 0
    merged_5y = merged_5y.fillna(0)

    # Save
    out_path = os.path.join(CACHE_DIR, "onchain_merged.csv")
    merged_5y.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # List all features
    feat_cols = [c for c in merged_5y.columns if c != "date"]
    print(f"\n  Features ({len(feat_cols)}):")
    for i, c in enumerate(feat_cols):
        non_zero = (merged_5y[c] != 0).sum()
        print(f"    {i+1:2d}. {c:35s} ({non_zero}/{len(merged_5y)} non-zero)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
