import os
from dotenv import load_dotenv

load_dotenv()

# ─── External APIs ────────────────────────────────────────
FEAR_GREED_BASE = "https://api.alternative.me/fng"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
BINANCE_BASE = "https://api.binance.com/api/v3"

# ─── Polymarket ──────────────────────────────────────────
POLYMARKET_APRIL_SLUG = "what-price-will-bitcoin-hit-in-april-2026"

# ─── True Markets API ─────────────────────────────────────
TRUEMARKETS_API_BASE = os.getenv("TRUEMARKETS_API_BASE", "https://api.truemarkets.co")
TRUEMARKETS_KEY_FILE = os.getenv("TRUEMARKETS_KEY_FILE", "")

# ─── Supported Coins (kept for market page compatibility) ─
SUPPORTED_COINS = {
    "bitcoin": {
        "symbol": "BTC",
        "base_asset": "BTC",
        "thresholds": [45000, 50000, 55000, 80000, 90000, 100000],
        "subreddits": ["Bitcoin", "CryptoCurrency"],
        "polymarket_keywords": ["bitcoin", "btc"],
    },
    "ethereum": {
        "symbol": "ETH",
        "base_asset": "ETH",
        "thresholds": [1500, 2000, 2500, 3000, 4000, 5000],
        "subreddits": ["ethereum", "CryptoCurrency"],
        "polymarket_keywords": ["ethereum", "eth"],
    },
    "solana": {
        "symbol": "SOL",
        "base_asset": "SOL",
        "thresholds": [80, 100, 130, 160, 200, 250],
        "subreddits": ["solana", "CryptoCurrency"],
        "polymarket_keywords": ["solana", "sol"],
    },
    "hyperliquid": {
        "symbol": "HYPE",
        "base_asset": "HYPE",
        "thresholds": [10, 15, 20, 30, 40, 50],
        "subreddits": ["hyperliquid", "CryptoCurrency"],
        "polymarket_keywords": ["hyperliquid", "hype"],
    },
    "coinbase-wrapped-btc": {
        "symbol": "CBBTC",
        "base_asset": "CBBTC",
        "thresholds": [45000, 50000, 55000, 80000, 90000, 100000],
        "subreddits": ["Bitcoin", "CryptoCurrency"],
        "polymarket_keywords": ["bitcoin", "btc"],
    },
}

# ─── Model Config ─────────────────────────────────────────
SAVED_DIR = os.path.join(os.path.dirname(__file__), "models", "saved")
LIGHTGBM_MODEL_PATH = os.path.join(SAVED_DIR, "lightgbm_btc.pkl")
SIGNAL_WEIGHTS_PATH = os.path.join(SAVED_DIR, "signal_weights.json")
BACKTEST_RESULTS_PATH = os.path.join(SAVED_DIR, "backtest_results.json")

# ─── Frontend ─────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
