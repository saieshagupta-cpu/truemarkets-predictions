"""
Minimal streamable-HTTP MCP client for mcp.truemarkets.co/marketdata.
Used by the backend to poll fresh price/sentiment data on a timer.
"""

import json
import time
import httpx

MCP_URL = "https://mcp.truemarkets.co/marketdata/mcp"
PROTOCOL_VERSION = "2025-03-26"

_session_id: str | None = None
_session_created_at: float = 0
_SESSION_TTL = 1800  # 30 min — re-init past this


async def _ensure_session(client: httpx.AsyncClient) -> str:
    global _session_id, _session_created_at
    if _session_id and (time.time() - _session_created_at) < _SESSION_TTL:
        return _session_id

    resp = await client.post(
        MCP_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        },
        json={
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "tmbackend", "version": "1.0"},
            },
        },
        timeout=10,
    )
    resp.raise_for_status()
    sid = resp.headers.get("mcp-session-id")
    if not sid:
        raise RuntimeError("No mcp-session-id returned from initialize")

    await client.post(
        MCP_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
            "mcp-session-id": sid,
        },
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        timeout=5,
    )

    _session_id = sid
    _session_created_at = time.time()
    return sid


async def _parse_response(resp: httpx.Response) -> dict:
    """MCP responses may be plain JSON or SSE-framed. Handle both."""
    ct = resp.headers.get("content-type", "")
    if "text/event-stream" in ct:
        # Parse SSE frames: "event: message\ndata: {...}\n\n"
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                return json.loads(line[6:])
        raise RuntimeError("No data frame in SSE response")
    return resp.json()


async def call_tool(name: str, arguments: dict) -> dict:
    async with httpx.AsyncClient() as client:
        sid = await _ensure_session(client)
        resp = await client.post(
            MCP_URL,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
                "mcp-session-id": sid,
            },
            json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            timeout=15,
        )
        # Session expired → retry once with fresh session
        if resp.status_code in (400, 404):
            global _session_id
            _session_id = None
            sid = await _ensure_session(client)
            resp = await client.post(
                MCP_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": PROTOCOL_VERSION,
                    "mcp-session-id": sid,
                },
                json={
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
                timeout=15,
            )
        resp.raise_for_status()
        payload = await _parse_response(resp)

    result = payload.get("result", {})
    content = result.get("content", [])
    if content and content[0].get("type") == "text":
        return json.loads(content[0]["text"])
    return result


async def fetch_price_history(symbol: str = "BTC", window: str = "1d", resolution: str = "1h") -> dict:
    return await call_tool("get_price_history", {
        "symbols": symbol, "window": window, "resolution": resolution,
    })


async def fetch_asset_summary(symbol: str = "BTC") -> dict:
    return await call_tool("get_asset_summary", {"symbol": symbol})
