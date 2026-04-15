"use client";

import { useState } from "react";
import type { Signal } from "@/lib/api";
import { placeOrder } from "@/lib/api";

interface BuySellPanelProps {
  recommendedSide: "buy" | "sell";
  confidence: number;
  buySignals: Signal[];
  sellSignals: Signal[];
  neutralSignals: Signal[];
  buyCount: number;
  sellCount: number;
  neutralCount: number;
  totalSignals: number;
  currentPrice: number;
  onOrderPlaced?: () => void;
}

export default function BuySellPanel({
  recommendedSide, confidence, buySignals, sellSignals, neutralSignals,
  buyCount, sellCount, neutralCount, totalSignals, currentPrice, onOrderPlaced,
}: BuySellPanelProps) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <SideCard
          side="buy"
          isRecommended={recommendedSide === "buy"}
          signals={buySignals}
          count={buyCount}
          total={totalSignals}
          price={currentPrice}
          onOrderPlaced={onOrderPlaced}
        />
        <SideCard
          side="sell"
          isRecommended={recommendedSide === "sell"}
          signals={sellSignals}
          count={sellCount}
          total={totalSignals}
          price={currentPrice}
          onOrderPlaced={onOrderPlaced}
        />
      </div>
      {neutralSignals.length > 0 && (
        <div className="bg-tm-card border border-tm-border rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-bold text-tm-muted">NEUTRAL</span>
            <span className="text-xs text-tm-muted">{neutralCount}/{totalSignals}</span>
          </div>
          <div className="space-y-2">
            {neutralSignals.map((s, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-tm-muted mt-0.5 text-xs">●</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 mb-0.5">
                    <span className="text-xs font-semibold text-tm-text">{s.name}</span>
                    <span className="text-[9px] text-tm-muted">({Math.round(s.weight * 100)}%)</span>
                  </div>
                  <p className="text-[11px] text-tm-muted leading-snug">{s.reason}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SideCard({
  side, isRecommended, signals, count, total, price, onOrderPlaced,
}: {
  side: "buy" | "sell";
  isRecommended: boolean;
  signals: Signal[];
  count: number;
  total: number;
  price: number;
  onOrderPlaced?: () => void;
}) {
  const [qty, setQty] = useState("0.001");
  const [placing, setPlacing] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const isBuy = side === "buy";
  const color = isBuy ? "tm-green" : "tm-red";
  const borderClass = isRecommended
    ? isBuy ? "border-tm-green/50 shadow-[0_0_15px_rgba(0,212,170,0.15)]" : "border-tm-red/50 shadow-[0_0_15px_rgba(255,107,107,0.15)]"
    : "border-tm-border";

  const handleOrder = async () => {
    if (!qty || parseFloat(qty) <= 0) return;
    setPlacing(true);
    setResult(null);
    try {
      const res = await placeOrder("BTC", side, qty);
      setResult(`Order placed: ${res.order_id}`);
      onOrderPlaced?.();
    } catch (e: unknown) {
      setResult(`Failed: ${e instanceof Error ? e.message : "unknown error"}`);
    } finally {
      setPlacing(false);
    }
  };

  const total_usd = (parseFloat(qty) || 0) * price;

  return (
    <div className={`bg-tm-card border ${borderClass} rounded-xl p-4 relative flex flex-col`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`text-lg font-bold text-${color}`}>
            {isBuy ? "BUY" : "SELL"} BTC
          </span>
          {isRecommended && (
            <span className="text-[9px] font-bold bg-tm-accent text-white px-1.5 py-0.5 rounded uppercase tracking-wider">
              Recommended
            </span>
          )}
        </div>
        <span className="text-xs text-tm-muted">{count}/{total}</span>
      </div>

      {/* Signal reasons */}
      <div className="space-y-2 flex-1">
        {signals.map((s, i) => (
          <div key={i} className="flex items-start gap-2">
            <span className={`text-${color} mt-0.5 text-xs`}>●</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className="text-xs font-semibold text-tm-text">{s.name}</span>
                <span className="text-[9px] text-tm-muted">({Math.round(s.weight * 100)}%)</span>
              </div>
              <p className="text-[11px] text-tm-muted leading-snug">{s.reason}</p>
            </div>
          </div>
        ))}
        {signals.length === 0 && (
          <p className="text-xs text-tm-muted italic">No signals on this side</p>
        )}
      </div>

      {/* Trade section */}
      <div className="mt-3 pt-3 border-t border-tm-border">
        <div className="flex items-center gap-2 mb-2">
          <input
            type="number"
            step="0.001"
            min="0"
            value={qty}
            onChange={(e) => setQty(e.target.value)}
            className="w-24 bg-tm-bg border border-tm-border rounded px-2 py-1 text-xs text-tm-text focus:outline-none focus:border-tm-accent"
            placeholder="BTC qty"
          />
          <span className="text-[10px] text-tm-muted">BTC</span>
          <span className="text-[10px] text-tm-muted ml-auto">
            ≈ ${total_usd.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
        </div>
        <button
          onClick={handleOrder}
          disabled={placing || !qty || parseFloat(qty) <= 0}
          className={`w-full py-2 rounded-lg text-xs font-bold transition-all ${
            isBuy
              ? "bg-tm-green hover:bg-tm-green/80 text-black"
              : "bg-tm-red hover:bg-tm-red/80 text-white"
          } disabled:opacity-40 disabled:cursor-not-allowed`}
        >
          {placing ? "Placing..." : `${side.toUpperCase()} BTC`}
        </button>
        {result && (
          <p className={`text-[10px] mt-1 ${result.startsWith("Failed") ? "text-tm-red" : "text-tm-green"}`}>
            {result}
          </p>
        )}
      </div>
    </div>
  );
}
