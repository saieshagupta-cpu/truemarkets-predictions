"""
Run this script to fetch data from TrueMarkets API and save to cache.
It reads your API key from the local file — never prints it.

Usage:
  python fetch_tm_data.py

Or in Google Colab:
  1. Upload your truemarkets-api-key-c761341e.json file
  2. pip install PyJWT cryptography requests
  3. Run this script
"""

import json
import time
import os
import sys

# --- CONFIG ---
# Change this path to where your key file is
KEY_FILE = os.environ.get("TRUEMARKETS_KEY_FILE",
    "/Users/saieshagupta/Downloads/truemarkets-api-key-c761341e.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "backend", "app", "data", "cache")
API_BASE = "https://api.truemarkets.co"

# For Google Colab, set these:
# KEY_FILE = "/content/truemarkets-api-key-c761341e.json"
# OUTPUT_DIR = "/content/cache"

# --- LOAD KEY ---
try:
    import jwt
except ImportError:
    print("Installing PyJWT...")
    os.system(f"{sys.executable} -m pip install 'PyJWT[crypto]'")
    import jwt

with open(KEY_FILE) as f:
    jwk = json.load(f)

key_id = jwk["key_id"]
private_key = jwk["private_key"]

def make_token():
    now = int(time.time())
    return jwt.encode(
        {"sub": key_id, "iat": now, "exp": now + 300},
        jwt.algorithms.ECAlgorithm.from_jwk(json.dumps(private_key)),
        algorithm="ES256",
        headers={"kid": key_id, "alg": "ES256"},
    )

# --- TRY DIFFERENT HTTP CLIENTS ---
def fetch(url, params=None):
    """Try cloudscraper first (bypasses Cloudflare), fall back to requests."""
    headers = {"Authorization": f"Bearer {make_token()}", "Accept": "application/json"}

    # Try cloudscraper
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper()
        r = scraper.get(url, params=params, headers=headers, timeout=15)
        return r.status_code, r.text
    except ImportError:
        pass

    # Fall back to requests
    import requests
    r = requests.get(url, params=params, headers=headers, timeout=15)
    return r.status_code, r.text

# --- FETCH DATA ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

endpoints = [
    ("prices/history", {"symbols": "BTC", "window": "1d", "resolution": "1h"}, "btc_1d_1h.json"),
    ("prices/history", {"symbols": "BTC", "window": "7d", "resolution": "1h"}, "btc_7d_1h.json"),
    ("prices/history", {"symbols": "BTC", "window": "1M", "resolution": "1d"}, "btc_1M_1d.json"),
    ("market/summary", {}, "market_summary.json"),
    ("assets/summary", {"symbol": "BTC"}, "btc_summary.json"),
]

print(f"API Base: {API_BASE}")
print(f"Key ID: {key_id}")
print(f"Output: {OUTPUT_DIR}")
print()

for path, params, filename in endpoints:
    url = f"{API_BASE}/v1/{path}"
    print(f"Fetching {path}...", end=" ")

    status, text = fetch(url, params)
    print(f"Status: {status}")

    if status == 200:
        data = json.loads(text)
        outpath = os.path.join(OUTPUT_DIR, filename)
        with open(outpath, "w") as f:
            json.dump(data, f)
        print(f"  Saved to {outpath}")

        # Show preview
        if "results" in data:
            pts = data.get("results", [{}])[0].get("points", [])
            if pts:
                print(f"  {len(pts)} data points, latest: ${float(pts[-1]['price']):,.2f}")
        elif "sentiment" in data:
            print(f"  Sentiment: {data.get('sentiment', '?')}")
    elif status == 401:
        print(f"  AUTH ERROR: {text[:200]}")
        print("  Your API key may need to be registered/activated on the TrueMarkets dashboard.")
    elif status == 403:
        print(f"  CLOUDFLARE BLOCKED. Try: pip install cloudscraper")
    else:
        print(f"  Error: {text[:200]}")

print("\nDone. If files were saved, copy them to backend/app/data/cache/")
