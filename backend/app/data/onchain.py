import httpx
from app.config import BLOCKCHAIN_INFO_BASE


async def fetch_onchain_metrics() -> dict:
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        stats_resp = await client.get(f"{BLOCKCHAIN_INFO_BASE}/stats")
        stats_resp.raise_for_status()
        stats = stats_resp.json()

    return {
        "hash_rate": stats.get("hash_rate", 0),
        "difficulty": stats.get("difficulty", 0),
        "n_tx": stats.get("n_tx", 0),
        "n_blocks_total": stats.get("n_blocks_total", 0),
        "minutes_between_blocks": stats.get("minutes_between_blocks", 0),
        "total_btc_sent": stats.get("total_btc_sent", 0) / 1e8,
        "market_price_usd": stats.get("market_price_usd", 0),
        "trade_volume_usd": stats.get("trade_volume_usd", 0),
    }
