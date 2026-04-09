import os
from dotenv import load_dotenv

load_dotenv()

COINGECKO_BASE = "https://api.coingecko.com/api/v3"
FEAR_GREED_BASE = "https://api.alternative.me/fng"
BLOCKCHAIN_INFO_BASE = "https://api.blockchain.info"
POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"

# True Markets Gateway API
TRUEMARKETS_API_BASE = os.getenv(
    "TRUEMARKETS_API_BASE",
    "https://docs.truemarkets.co/_mock/apis/gateway/v1",  # Mock server
)
TRUEMARKETS_API_KEY = os.getenv("TRUEMARKETS_API_KEY", "mock-demo-key")

SUPPORTED_COINS = {
    "bitcoin": {
        "symbol": "BTC",
        "coingecko_id": "bitcoin",
        "base_asset": "BTC",
        "thresholds": [45000, 50000, 55000, 80000, 90000, 100000],
        "subreddits": ["Bitcoin", "CryptoCurrency"],
        "polymarket_keywords": ["bitcoin", "btc"],
    },
    "ethereum": {
        "symbol": "ETH",
        "coingecko_id": "ethereum",
        "base_asset": "ETH",
        "thresholds": [1500, 2000, 2500, 3000, 4000, 5000],
        "subreddits": ["ethereum", "CryptoCurrency"],
        "polymarket_keywords": ["ethereum", "eth"],
    },
    "solana": {
        "symbol": "SOL",
        "coingecko_id": "solana",
        "base_asset": "SOL",
        "thresholds": [80, 100, 130, 160, 200, 250],
        "subreddits": ["solana", "CryptoCurrency"],
        "polymarket_keywords": ["solana", "sol"],
    },
    "hyperliquid": {
        "symbol": "HYPE",
        "coingecko_id": "hyperliquid",
        "base_asset": "HYPE",
        "thresholds": [10, 15, 20, 30, 40, 50],
        "subreddits": ["hyperliquid", "CryptoCurrency"],
        "polymarket_keywords": ["hyperliquid", "hype"],
    },
    "coinbase-wrapped-btc": {
        "symbol": "CBBTC",
        "coingecko_id": "coinbase-wrapped-btc",
        "base_asset": "CBBTC",
        "thresholds": [45000, 50000, 55000, 80000, 90000, 100000],
        "subreddits": ["Bitcoin", "CryptoCurrency"],
        "polymarket_keywords": ["bitcoin", "btc"],
    },
}

MODEL_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "models", "saved")
LSTM_WEIGHT = 0.55
XGBOOST_WEIGHT = 0.30
SENTIMENT_WEIGHT = 0.15

DATA_REFRESH_HOURS = 6
LOOKBACK_DAYS = 365
SEQUENCE_LENGTH = 30

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
