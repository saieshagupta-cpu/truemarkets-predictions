"""
Polymarket scraper for BTC April 2026 price thresholds.
Source: Gamma API (public, no auth).
Endpoint: GET gamma-api.polymarket.com/events?slug=what-price-will-bitcoin-hit-in-april-2026
"""

import re
import json
import httpx
from app.config import POLYMARKET_GAMMA_BASE, POLYMARKET_APRIL_SLUG


async def fetch_polymarket_thresholds() -> list[dict]:
    """
    Fetch all BTC price threshold markets from the April 2026 event.
    Returns list sorted by threshold ascending.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{POLYMARKET_GAMMA_BASE}/events",
                params={"slug": POLYMARKET_APRIL_SLUG, "closed": "false"},
            )
            resp.raise_for_status()
            events = resp.json()

        if not events:
            return []

        event = events[0] if isinstance(events, list) else events
        markets = event.get("markets", [])
        if not markets:
            return []

        results = []
        for m in markets:
            # Skip closed/resolved markets
            if m.get("closed", False):
                continue

            question = m.get("question", "")
            match = re.search(r'\$?([\d,]+)', question)
            if not match:
                continue
            threshold = int(match.group(1).replace(",", ""))

            outcome_prices = m.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except Exception:
                    outcome_prices = [0.5, 0.5]
            yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

            q_lower = question.lower()
            if "dip" in q_lower or "drop" in q_lower or "fall" in q_lower or "below" in q_lower:
                direction = "down"
            else:
                direction = "up"

            results.append({
                "threshold": threshold,
                "direction": direction,
                "yes_price": round(yes_price, 4),
                "yes_pct": round(yes_price * 100, 1),
                "volume": float(m.get("volume", 0)),
                "volume_24h": float(m.get("volume24hr", 0) or 0),
                "question": question,
                "liquidity": float(m.get("liquidity", 0) or 0),
            })

        results.sort(key=lambda x: x["threshold"])
        return results

    except Exception as e:
        print(f"[polymarket] Error: {e}")
        return []
