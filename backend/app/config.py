import os
from dotenv import load_dotenv

load_dotenv()

# ─── External APIs ────────────────────────────────────────
FEAR_GREED_BASE = "https://api.alternative.me/fng"
BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

# ─── True Markets API ─────────────────────────────────────
TRUEMARKETS_API_BASE = os.getenv("TRUEMARKETS_API_BASE", "https://api.truemarkets.co")
TRUEMARKETS_KEY_FILE = os.getenv("TRUEMARKETS_KEY_FILE", "")

# ─── Supported Coins ──────────────────────────────────────
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
MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "models", "saved")
SEQUENCE_LENGTH = 30  # Lookback window (30 periods)

# ─── Frontend ─────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
