"use client";

import type { PolymarketThreshold } from "@/lib/api";

interface PolymarketTableProps {
  thresholds: PolymarketThreshold[];
  currentPrice: number;
}

export default function PolymarketTable({ thresholds, currentPrice }: PolymarketTableProps) {
  if (!thresholds.length) {
    return (
      <div className="bg-tm-card border border-tm-border rounded-xl p-4">
        <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-2">Polymarket</h3>
        <p className="text-xs text-tm-muted">No Polymarket data available</p>
      </div>
    );
  }

  // Split into above and below current price, take 5 closest each
  const above = [...thresholds]
    .filter(t => t.threshold > currentPrice)
    .sort((a, b) => a.threshold - b.threshold)
    .slice(0, 5)
    .reverse(); // highest first

  const below = [...thresholds]
    .filter(t => t.threshold <= currentPrice)
    .sort((a, b) => b.threshold - a.threshold)
    .slice(0, 5); // highest first

  const sorted = [...above, ...below];

  // Current price goes between above and below
  const priceInsertIdx = above.length;

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider">
          Polymarket BTC Thresholds
        </h3>
        <span className="text-[10px] text-tm-muted">April 2026</span>
      </div>

      <div className="space-y-0.5">
        {sorted.map((t, idx) => (
          <div key={`${t.direction}-${t.threshold}`}>
            {/* Insert current price marker between thresholds */}
            {idx === priceInsertIdx && (
              <PriceMarker price={currentPrice} />
            )}
            <ThresholdRow t={t} />
          </div>
        ))}
        {/* If current price is below ALL thresholds */}
        {priceInsertIdx === -1 && (
          <PriceMarker price={currentPrice} />
        )}
      </div>

      <p className="text-[9px] text-tm-muted mt-2">
        Source: Polymarket Gamma API &bull; Probabilities = Yes price &bull; Updated every 30s
      </p>
    </div>
  );
}

function PriceMarker({ price }: { price: number }) {
  return (
    <div className="flex items-center gap-2 py-1.5 px-2 my-1 bg-tm-accent/10 rounded-lg border border-tm-accent/30">
      <span className="text-[10px] text-tm-accent">▸</span>
      <span className="text-xs font-bold text-tm-accent">Current Price</span>
      <span className="text-xs font-bold text-tm-accent flex-1 text-right">
        ${price.toLocaleString(undefined, { maximumFractionDigits: 0 })}
      </span>
    </div>
  );
}

function ThresholdRow({ t }: { t: PolymarketThreshold }) {
  const isUp = t.direction === "up";
  const color = isUp ? "text-tm-green" : "text-tm-red";
  const bgColor = isUp ? "bg-tm-green" : "bg-tm-red";
  const pct = t.yes_pct;

  return (
    <div className="flex items-center gap-2 py-1 px-2 rounded hover:bg-tm-bg/50 transition-colors">
      <span className={`text-[10px] ${color}`}>{isUp ? "\u2191" : "\u2193"}</span>
      <span className="text-xs font-medium w-20">${t.threshold.toLocaleString()}</span>
      <div className="flex-1 bg-tm-border/30 rounded-full h-1.5 overflow-hidden">
        <div className={`${bgColor} h-full rounded-full transition-all`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className={`text-xs font-bold w-10 text-right ${pct > 10 ? color : "text-tm-muted"}`}>
        {pct < 1 ? "<1%" : `${pct.toFixed(0)}%`}
      </span>
      <span className="text-[9px] text-tm-muted w-14 text-right">
        ${(t.volume / 1e6).toFixed(1)}M
      </span>
    </div>
  );
}
