"""
BTC Order Flow from real exchanges.
Sources:
  1. Binance.US — GET api.binance.us/api/v3/trades?symbol=BTCUSDT&limit=1000
  2. Coinbase   — GET api.exchange.coinbase.com/products/BTC-USD/trades?limit=100
No auth required for either.
"""

import asyncio
import httpx

BINANCE_US = "https://api.binance.us/api/v3"
COINBASE = "https://api.exchange.coinbase.com"


async def fetch_binance_order_flow() -> dict:
    """
    Fetch real BTC order flow from Binance.US + Coinbase.
    Returns buy/sell volume ratio + order book imbalance.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Fetch in parallel: Binance.US trades + depth, Coinbase trades
            bn_trades_t = client.get(f"{BINANCE_US}/trades", params={"symbol": "BTCUSDT", "limit": 1000})
            bn_depth_t = client.get(f"{BINANCE_US}/depth", params={"symbol": "BTCUSDT", "limit": 20})
            cb_trades_t = client.get(f"{COINBASE}/products/BTC-USD/trades", params={"limit": 100})

            results = await asyncio.gather(bn_trades_t, bn_depth_t, cb_trades_t, return_exceptions=True)
            bn_trades_resp, bn_depth_resp, cb_trades_resp = results

        # ── Binance.US trades ──
        buy_volume = 0.0
        sell_volume = 0.0
        buy_count = 0
        sell_count = 0

        if not isinstance(bn_trades_resp, Exception) and bn_trades_resp.status_code == 200:
            trades = bn_trades_resp.json()
            if isinstance(trades, list):
                for t in trades:
                    qty = float(t.get("qty", 0))
                    # isBuyerMaker=true → sell aggressor, false → buy aggressor
                    if t.get("isBuyerMaker", False):
                        sell_volume += qty
                        sell_count += 1
                    else:
                        buy_volume += qty
                        buy_count += 1

        # ── Coinbase trades (supplement) ──
        if not isinstance(cb_trades_resp, Exception) and cb_trades_resp.status_code == 200:
            cb_trades = cb_trades_resp.json()
            if isinstance(cb_trades, list):
                for t in cb_trades:
                    qty = float(t.get("size", 0))
                    side = t.get("side", "")
                    # Coinbase "side" is the maker side: "buy" = maker was buyer = sell aggressor
                    if side == "buy":
                        sell_volume += qty
                        sell_count += 1
                    elif side == "sell":
                        buy_volume += qty
                        buy_count += 1

        total_vol = buy_volume + sell_volume
        buy_sell_ratio = buy_volume / total_vol if total_vol > 0 else 0.5

        # ── Binance.US order book depth ──
        bid_depth = 0.0
        ask_depth = 0.0

        if not isinstance(bn_depth_resp, Exception) and bn_depth_resp.status_code == 200:
            depth = bn_depth_resp.json()
            if isinstance(depth, dict):
                bid_depth = sum(float(b[1]) for b in depth.get("bids", []))
                ask_depth = sum(float(a[1]) for a in depth.get("asks", []))

        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Simple: buy vs sell ratio determines pressure
        # buy > sell = buy, buy >> sell by 10%+ = strong_buy, vice versa
        vol_signal = (buy_sell_ratio - 0.5) * 2  # 0-1 → -1 to +1
        signal = vol_signal * 0.8 + imbalance * 0.2
        signal = max(-1.0, min(1.0, signal))

        if buy_sell_ratio >= 0.60:       # buy >> sell by 10%+
            pressure = "strong_buy"
        elif buy_sell_ratio > 0.50:      # buy > sell
            pressure = "buy"
        elif buy_sell_ratio <= 0.40:     # sell >> buy by 10%+
            pressure = "strong_sell"
        elif buy_sell_ratio < 0.50:      # sell > buy
            pressure = "sell"
        else:
            pressure = "neutral"

        sources = []
        if buy_count + sell_count > 0:
            if not isinstance(bn_trades_resp, Exception) and bn_trades_resp.status_code == 200:
                sources.append("Binance.US")
            if not isinstance(cb_trades_resp, Exception) and cb_trades_resp.status_code == 200:
                sources.append("Coinbase")

        return {
            "buy_volume": round(buy_volume, 6),
            "sell_volume": round(sell_volume, 6),
            "buy_sell_ratio": round(buy_sell_ratio, 4),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "bid_depth": round(bid_depth, 6),
            "ask_depth": round(ask_depth, 6),
            "imbalance": round(imbalance, 4),
            "signal": round(signal, 4),
            "pressure": pressure,
            "source": " + ".join(sources) if sources else "No exchange data",
        }

    except Exception as e:
        print(f"[order_flow] Error: {e}")
        return {
            "buy_volume": 0, "sell_volume": 0, "buy_sell_ratio": 0.5,
            "buy_count": 0, "sell_count": 0,
            "bid_depth": 0, "ask_depth": 0, "imbalance": 0,
            "signal": 0, "pressure": "neutral",
            "source": "Error fetching exchange data",
        }
