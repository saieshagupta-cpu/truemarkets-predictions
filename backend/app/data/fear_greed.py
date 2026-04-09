import httpx
from app.config import FEAR_GREED_BASE


async def fetch_fear_greed(limit: int = 30) -> dict:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(FEAR_GREED_BASE, params={"limit": limit, "format": "json"})
        resp.raise_for_status()
        data = resp.json()["data"]

    history = [
        {
            "value": int(entry["value"]),
            "classification": entry["value_classification"],
            "timestamp": int(entry["timestamp"]),
        }
        for entry in data
    ]

    current = history[0] if history else {"value": 50, "classification": "Neutral"}

    return {
        "current": current,
        "history": history,
        "average_30d": sum(h["value"] for h in history) / max(len(history), 1),
    }
