import json
import httpx
from app.config import POLYMARKET_GAMMA_BASE

# Known event slugs for price prediction markets
KNOWN_SLUGS = {
    "bitcoin": "what-price-will-bitcoin-hit-before-2027",
    "ethereum": None,
    "solana": None,
    "hyperliquid": None,
    "coinbase-wrapped-btc": "what-price-will-bitcoin-hit-before-2027",
}


async def fetch_polymarket_markets(coin: str = "bitcoin", keywords: list[str] | None = None) -> list[dict]:
    """Fetch prediction markets for a given coin from Polymarket."""
    slug = KNOWN_SLUGS.get(coin)
    if slug:
        result = await _fetch_by_slug(slug)
        if result:
            return result

    # Fallback: search events by keyword
    if keywords:
        return await _search_events(keywords)
    return []


async def _fetch_by_slug(slug: str) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{POLYMARKET_GAMMA_BASE}/events", params={"slug": slug})
        if resp.status_code != 200 or not resp.json():
            return []
        return _parse_markets(resp.json()[0].get("markets", []))


async def _search_events(keywords: list[str]) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        for offset in range(0, 500, 100):
            resp = await client.get(
                f"{POLYMARKET_GAMMA_BASE}/events",
                params={"closed": "false", "limit": 100, "offset": offset},
            )
            if resp.status_code != 200:
                continue
            for event in resp.json():
                title = event.get("title", "").lower()
                if any(kw in title for kw in keywords) and "price" in title:
                    markets = event.get("markets", [])
                    if markets:
                        return _parse_markets(markets)
    return []


def _parse_markets(markets: list[dict]) -> list[dict]:
    result = []
    for m in markets:
        op = m.get("outcomePrices", [])
        if isinstance(op, str):
            try:
                op = json.loads(op)
            except (json.JSONDecodeError, TypeError):
                op = []
        prices = [float(p) for p in op] if op else []

        oc = m.get("outcomes", [])
        if isinstance(oc, str):
            try:
                oc = json.loads(oc)
            except (json.JSONDecodeError, TypeError):
                oc = []

        result.append({
            "question": m.get("question", ""),
            "outcomes": oc,
            "prices": prices,
            "yes_price": prices[0] if prices else 0,
            "volume": float(m.get("volume", 0)),
            "liquidity": float(m.get("liquidity", 0)),
            "end_date": m.get("endDate", ""),
        })
    result.sort(key=lambda x: x["question"])
    return result
