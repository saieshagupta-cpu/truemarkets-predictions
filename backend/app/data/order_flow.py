"""
Order flow analysis from Polymarket market microstructure + True Markets order history.

Polymarket provides: volume trends, bid/ask spreads, liquidity, price momentum.
True Markets provides: order history with buy/sell side and executed quantities.

Combined into a single order flow signal: buy pressure vs sell pressure.
"""

import numpy as np
import httpx
from app.config import POLYMARKET_GAMMA_BASE, TRUEMARKETS_API_BASE
from app.data.truemarkets_mcp import _make_jwt


async def fetch_order_flow(coin: str = "bitcoin", polymarket_markets: list | None = None) -> dict:
    """
    Aggregate order flow signals from Polymarket microstructure + True Markets orders.
    Returns a consolidated view of buy/sell pressure.
    """
    poly_flow = _analyze_polymarket_flow(polymarket_markets or [])
    tm_flow = await _analyze_truemarkets_flow()

    # Combine signals
    # Polymarket flow: based on volume momentum + price action across markets
    # TM flow: based on actual buy/sell order ratio
    poly_signal = poly_flow["signal"]  # -1 (sell pressure) to +1 (buy pressure)
    tm_signal = tm_flow["signal"]

    # Weight: Polymarket has real data. TM mock returns fake orders — only
    # use TM if there are enough real orders to be meaningful (>5).
    if tm_flow["order_count"] > 5:
        combined = poly_signal * 0.7 + tm_signal * 0.3
    else:
        combined = poly_signal  # TM mock data is not meaningful

    # Pressure follows the money — if more volume on one side, that's the pressure
    # Only call it neutral if volume is truly even (<2% difference)
    if combined > 0.15:
        pressure = "strong_buy"
    elif combined > 0.02:
        pressure = "buy"
    elif combined < -0.15:
        pressure = "strong_sell"
    elif combined < -0.02:
        pressure = "sell"
    else:
        pressure = "neutral"

    return {
        "combined_signal": round(combined, 4),
        "pressure": pressure,
        "polymarket_flow": poly_flow,
        "truemarkets_flow": tm_flow,
    }


def _analyze_polymarket_flow(markets: list) -> dict:
    """
    Analyze order flow from Polymarket market microstructure.
    Uses volume trends, bid-ask dynamics, and price momentum.
    """
    if not markets:
        return {"signal": 0.0, "details": {}}

    # Separate upside (reach) vs downside (dip) markets
    up_markets = [m for m in markets if "reach" in m.get("question", "").lower()]
    down_markets = [m for m in markets if "dip" in m.get("question", "").lower()]

    # 1. Volume momentum: is more money flowing into upside or downside bets?
    up_vol_24h = sum(float(m.get("volume24hr", 0) or 0) for m in up_markets)
    down_vol_24h = sum(float(m.get("volume24hr", 0) or 0) for m in down_markets)
    total_vol = up_vol_24h + down_vol_24h

    if total_vol > 0:
        vol_ratio = (up_vol_24h - down_vol_24h) / total_vol  # -1 to +1
    else:
        vol_ratio = 0.0

    # 2. Volume acceleration: is volume increasing or decreasing?
    up_vol_1w = sum(float(m.get("volume1wk", 0) or 0) for m in up_markets)
    down_vol_1w = sum(float(m.get("volume1wk", 0) or 0) for m in down_markets)
    daily_avg_1w = (up_vol_1w + down_vol_1w) / 7 if (up_vol_1w + down_vol_1w) > 0 else 1
    vol_accel = (total_vol / max(daily_avg_1w, 1)) - 1  # >0 means accelerating

    # 3. Bid-ask spread analysis: tight spreads = conviction, wide = uncertainty
    spreads = [float(m.get("spread", 0) or 0) for m in markets if m.get("spread")]
    avg_spread = sum(spreads) / max(len(spreads), 1)

    # 4. Price momentum: are upside markets gaining or losing?
    up_momentum = sum(float(m.get("oneDayPriceChange", 0) or 0) for m in up_markets) / max(len(up_markets), 1)
    down_momentum = sum(float(m.get("oneDayPriceChange", 0) or 0) for m in down_markets) / max(len(down_markets), 1)
    # If upside prices rising = bullish flow, if downside prices rising = bearish flow
    momentum_signal = up_momentum - down_momentum  # positive = bullish

    # 5. Liquidity depth: more liquidity in upside vs downside
    up_liq = sum(float(m.get("liquidity", 0) or 0) for m in up_markets)
    down_liq = sum(float(m.get("liquidity", 0) or 0) for m in down_markets)
    total_liq = up_liq + down_liq
    liq_ratio = (up_liq - down_liq) / max(total_liq, 1)

    # Combined Polymarket flow signal
    # Volume IS the order flow. If more money flows up, it's buy pressure.
    # Other signals are minor confirmation only — they cannot flip the direction.
    base = vol_ratio  # This IS the signal: -1 to +1
    # Small adjustments from other data (cannot exceed ±0.03 total)
    adj = (
        np.clip(momentum_signal * 2, -0.01, 0.01)
        + np.clip(liq_ratio * 0.1, -0.01, 0.01)
        + np.clip(min(max(vol_accel, -1), 1) * 0.1, -0.01, 0.01)
    )
    signal = max(-1, min(1, base + adj))

    return {
        "signal": round(signal, 4),
        "details": {
            "up_volume_24h": round(up_vol_24h, 2),
            "down_volume_24h": round(down_vol_24h, 2),
            "volume_ratio": round(vol_ratio, 4),
            "volume_acceleration": round(vol_accel, 4),
            "avg_spread": round(avg_spread, 4),
            "up_momentum": round(up_momentum, 4),
            "down_momentum": round(down_momentum, 4),
            "liquidity_ratio": round(liq_ratio, 4),
        },
    }


async def _analyze_truemarkets_flow() -> dict:
    """
    Analyze buy vs sell pressure from True Markets order history.
    """
    try:
        token = _make_jwt()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{TRUEMARKETS_API_BASE}/orders",
                headers=headers,
                params={"limit": 50},
            )
            if resp.status_code != 200:
                return {"signal": 0.0, "order_count": 0, "buy_count": 0, "sell_count": 0}

            data = resp.json()
            orders = data.get("data", [])
    except Exception:
        return {"signal": 0.0, "order_count": 0, "buy_count": 0, "sell_count": 0}

    if not orders:
        return {"signal": 0.0, "order_count": 0, "buy_count": 0, "sell_count": 0}

    buy_count = sum(1 for o in orders if o.get("side") == "buy")
    sell_count = sum(1 for o in orders if o.get("side") == "sell")
    total = buy_count + sell_count

    if total > 0:
        signal = (buy_count - sell_count) / total  # -1 to +1
    else:
        signal = 0.0

    return {
        "signal": round(signal, 4),
        "order_count": total,
        "buy_count": buy_count,
        "sell_count": sell_count,
    }
