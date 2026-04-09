"use client";

import { useState } from "react";
import type { RecommendedTrade as TradeType } from "@/lib/api";
import { placeOrder } from "@/lib/api";

interface RecommendedTradeProps {
  trade: TradeType;
  onOrderPlaced: () => void;
}

export default function RecommendedTrade({ trade, onOrderPlaced }: RecommendedTradeProps) {
  const [qty, setQty] = useState("1");
  const [placing, setPlacing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isBuy = trade.side === "buy";
  const price = trade.quote ? parseFloat(trade.quote.price) : 0;
  const total = price * parseFloat(qty || "0");

  const handleOrder = async () => {
    setPlacing(true);
    setError(null);
    try {
      const res = await placeOrder(trade.base_asset, trade.side, qty);
      setResult(res.order_id || "Order placed");
      onOrderPlaced();
    } catch {
      setError("Order failed. Try again.");
    } finally {
      setPlacing(false);
    }
  };

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <p className="text-xs text-tm-muted uppercase tracking-wider">Recommended</p>
          <span className={`text-sm font-bold ${isBuy ? "text-tm-green" : "text-tm-red"}`}>
            {trade.side.toUpperCase()} {trade.symbol}
          </span>
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
            isBuy ? "bg-tm-green/10 text-tm-green" : "bg-tm-red/10 text-tm-red"
          }`}>{trade.based_on_mispricing ? "mispricing" : "consensus"}</span>
        </div>
      </div>

      {/* Reasons — compact */}
      <div className="space-y-1 mb-3">
        {trade.reasons.map((reason, i) => (
          <p key={i} className={`text-xs ${i === 0 ? "text-tm-text" : "text-tm-muted"}`}>
            {i === 0 ? "\u25CF " : "\u25CB "}{reason}
          </p>
        ))}
      </div>

      {/* Quote + action — inline */}
      {trade.quote && !result && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 flex-1 text-xs">
            <span className="text-tm-muted">${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}/unit</span>
            <span className="text-tm-border">|</span>
            <input
              type="number"
              min="0.01"
              step="0.1"
              value={qty}
              onChange={(e) => { setQty(e.target.value); setResult(null); }}
              className="w-16 bg-tm-bg border border-tm-border rounded px-2 py-1 text-xs text-right focus:border-tm-accent focus:outline-none"
            />
            <span className="text-tm-muted">{trade.symbol}</span>
            <span className="text-tm-border">|</span>
            <span className="font-medium">${total.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          </div>
          <button
            onClick={handleOrder}
            disabled={placing || !qty}
            className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all shrink-0 ${
              isBuy
                ? "bg-tm-green/15 text-tm-green border border-tm-green/25 hover:bg-tm-green/25"
                : "bg-tm-red/15 text-tm-red border border-tm-red/25 hover:bg-tm-red/25"
            } disabled:opacity-50`}
          >
            {placing ? "..." : trade.side.toUpperCase()}
          </button>
        </div>
      )}

      {result && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-tm-green font-medium">Placed</span>
          <span className="text-tm-muted">ID: {result}</span>
          <button onClick={() => setResult(null)} className="text-tm-accent hover:underline ml-auto">Again</button>
        </div>
      )}

      {error && <p className="text-[10px] text-tm-red mt-1">{error}</p>}
    </div>
  );
}
