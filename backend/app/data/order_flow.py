"""
BTC Order Flow from Binance BTCUSDT.
Source: Binance public API (no auth required).
Endpoints:
  - GET api.binance.com/api/v3/trades?symbol=BTCUSDT&limit=1000
  - GET api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=20
"""

import httpx
from app.config import BINANCE_BASE


async def fetch_binance_order_flow() -> dict:
    """
    Fetch real BTC order flow from Binance.
    Returns buy/sell volume ratio + order book imbalance.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            trades_resp, depth_resp = await _fetch_parallel(client)

        # --- Recent trades: buy vs sell volume ---
        trades = trades_resp.json() if trades_resp.status_code == 200 else []
        buy_volume = 0.0
        sell_volume = 0.0
        buy_count = 0
        sell_count = 0
        for t in trades:
            qty = float(t.get("qty", 0))
            # isBuyerMaker=true means the buyer was the maker (passive) = a sell aggressor
            # isBuyerMaker=false means the seller was the maker = a buy aggressor
            if t.get("isBuyerMaker", False):
                sell_volume += qty
                sell_count += 1
            else:
                buy_volume += qty
                buy_count += 1

        total_vol = buy_volume + sell_volume
        buy_sell_ratio = buy_volume / total_vol if total_vol > 0 else 0.5

        # --- Order book: bid vs ask depth ---
        depth = depth_resp.json() if depth_resp.status_code == 200 else {"bids": [], "asks": []}
        bid_depth = sum(float(b[1]) for b in depth.get("bids", []))
        ask_depth = sum(float(a[1]) for a in depth.get("asks", []))
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Combined signal: volume ratio (60%) + book imbalance (40%)
        vol_signal = (buy_sell_ratio - 0.5) * 2  # map 0-1 to -1 to +1
        signal = vol_signal * 0.6 + imbalance * 0.4
        signal = max(-1.0, min(1.0, signal))

        if signal > 0.15:
            pressure = "strong_buy"
        elif signal > 0.03:
            pressure = "buy"
        elif signal < -0.15:
            pressure = "strong_sell"
        elif signal < -0.03:
            pressure = "sell"
        else:
            pressure = "neutral"

        return {
            "buy_volume": round(buy_volume, 4),
            "sell_volume": round(sell_volume, 4),
            "buy_sell_ratio": round(buy_sell_ratio, 4),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "bid_depth": round(bid_depth, 4),
            "ask_depth": round(ask_depth, 4),
            "imbalance": round(imbalance, 4),
            "signal": round(signal, 4),
            "pressure": pressure,
            "source": "Binance BTCUSDT",
        }

    except Exception as e:
        print(f"[order_flow] Binance error: {e}")
        return {
            "buy_volume": 0, "sell_volume": 0, "buy_sell_ratio": 0.5,
            "buy_count": 0, "sell_count": 0,
            "bid_depth": 0, "ask_depth": 0, "imbalance": 0,
            "signal": 0, "pressure": "neutral",
            "source": "Binance BTCUSDT (error)",
        }


async def _fetch_parallel(client: httpx.AsyncClient):
    """Fetch trades and depth in parallel."""
    import asyncio
    trades_task = client.get(f"{BINANCE_BASE}/trades", params={"symbol": "BTCUSDT", "limit": 1000})
    depth_task = client.get(f"{BINANCE_BASE}/depth", params={"symbol": "BTCUSDT", "limit": 20})
    return await asyncio.gather(trades_task, depth_task)
