"use client";

import type { Signal } from "@/lib/api";

interface BuySellPanelProps {
  recommendedSide: "buy" | "sell";
  confidence: number;
  buySignals: Signal[];
  sellSignals: Signal[];
  buyCount: number;
  sellCount: number;
  totalSignals: number;
  currentPrice: number;
}

export default function BuySellPanel({
  recommendedSide, confidence, buySignals, sellSignals,
  buyCount, sellCount, totalSignals, currentPrice,
}: BuySellPanelProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {/* BUY side */}
      <SideCard
        side="buy"
        isRecommended={recommendedSide === "buy"}
        signals={buySignals}
        count={buyCount}
        total={totalSignals}
        price={currentPrice}
      />
      {/* SELL side */}
      <SideCard
        side="sell"
        isRecommended={recommendedSide === "sell"}
        signals={sellSignals}
        count={sellCount}
        total={totalSignals}
        price={currentPrice}
      />
    </div>
  );
}

function SideCard({
  side, isRecommended, signals, count, total, price,
}: {
  side: "buy" | "sell";
  isRecommended: boolean;
  signals: Signal[];
  count: number;
  total: number;
  price: number;
}) {
  const isBuy = side === "buy";
  const color = isBuy ? "tm-green" : "tm-red";
  const borderClass = isRecommended
    ? isBuy ? "border-tm-green/50 shadow-[0_0_15px_rgba(0,212,170,0.15)]" : "border-tm-red/50 shadow-[0_0_15px_rgba(255,107,107,0.15)]"
    : "border-tm-border";

  return (
    <div className={`bg-tm-card border ${borderClass} rounded-xl p-4 relative`}>
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
        <span className="text-xs text-tm-muted">
          {count}/{total}
        </span>
      </div>

      {/* Price */}
      <div className="text-sm text-tm-muted mb-3">
        ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </div>

      {/* Signal reasons */}
      <div className="space-y-2">
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
    </div>
  );
}
