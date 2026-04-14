"""
True Markets Gateway API integration.
Docs: https://docs.truemarkets.co/apis/gateway/v1
Mock: https://docs.truemarkets.co/_mock/apis/gateway/v1/
Prod: https://api.truemarkets.co/v1/conductor/
"""

import httpx
from app.config import TRUEMARKETS_API_BASE
from app.data.truemarkets_mcp import _make_jwt


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    token = _make_jwt()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def get_quote(base_asset: str, quote_asset: str, side: str, qty: str, qty_unit: str = "base") -> dict:
    """POST /quotes — Get real-time price quote."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{TRUEMARKETS_API_BASE}/quotes",
            headers=_headers(),
            json={
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "side": side,
                "qty": qty,
                "qty_unit": qty_unit,
            },
        )
        resp.raise_for_status()
        return resp.json()


async def place_order(
    base_asset: str,
    quote_asset: str,
    side: str,
    qty: str,
    order_type: str = "market",
    price: str | None = None,
) -> dict:
    """POST /orders — Place a trade order."""
    body: dict = {
        "base_asset": base_asset,
        "quote_asset": quote_asset,
        "side": side,
        "qty": qty,
        "qty_unit": "base",
        "type": order_type,
    }
    if price and order_type == "limit":
        body["price"] = price

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{TRUEMARKETS_API_BASE}/orders",
            headers=_headers(),
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


async def list_orders(limit: int = 20) -> dict:
    """GET /orders — List recent orders."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{TRUEMARKETS_API_BASE}/orders",
            headers=_headers(),
            params={"limit": limit},
        )
        resp.raise_for_status()
        return resp.json()


async def cancel_order(order_id: str) -> dict:
    """DELETE /orders/{id} — Cancel an order."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(
            f"{TRUEMARKETS_API_BASE}/orders/{order_id}",
            headers=_headers(),
        )
        resp.raise_for_status()
        return resp.json() if resp.text else {"status": "cancelled"}


async def get_balances() -> dict:
    """GET /balances — List account balances."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{TRUEMARKETS_API_BASE}/balances",
            headers=_headers(),
        )
        resp.raise_for_status()
        return resp.json()
