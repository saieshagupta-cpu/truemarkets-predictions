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

  // Split into above and below current price
  const above = thresholds.filter(t => t.direction === "up").sort((a, b) => a.threshold - b.threshold);
  const below = thresholds.filter(t => t.direction === "down").sort((a, b) => b.threshold - a.threshold);

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider">
          Polymarket BTC Thresholds
        </h3>
        <span className="text-[10px] text-tm-muted">April 2026</span>
      </div>

      <div className="space-y-1">
        {/* Above current price */}
        {above.map((t) => (
          <ThresholdRow key={`up-${t.threshold}`} t={t} currentPrice={currentPrice} />
        ))}

        {/* Current price marker */}
        <div className="flex items-center gap-2 py-1.5 px-2 bg-tm-accent/10 rounded-lg border border-tm-accent/20">
          <span className="text-xs font-bold text-tm-accent">Current</span>
          <span className="text-xs font-bold text-tm-accent flex-1 text-right">
            ${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>

        {/* Below current price */}
        {below.map((t) => (
          <ThresholdRow key={`down-${t.threshold}`} t={t} currentPrice={currentPrice} />
        ))}
      </div>

      <p className="text-[9px] text-tm-muted mt-2">
        Source: Polymarket Gamma API. Probabilities = Yes price. Updated every 30s.
      </p>
    </div>
  );
}

function ThresholdRow({ t, currentPrice }: { t: PolymarketThreshold; currentPrice: number }) {
  const isUp = t.direction === "up";
  const color = isUp ? "text-tm-green" : "text-tm-red";
  const bgColor = isUp ? "bg-tm-green" : "bg-tm-red";
  const pct = t.yes_pct;

  return (
    <div className="flex items-center gap-2 py-1 px-2 rounded hover:bg-tm-bg/50 transition-colors">
      <span className={`text-[10px] ${color}`}>{isUp ? "\u2191" : "\u2193"}</span>
      <span className="text-xs font-medium w-16">${t.threshold.toLocaleString()}</span>
      <div className="flex-1 bg-tm-border/30 rounded-full h-1.5 overflow-hidden">
        <div className={`${bgColor} h-full rounded-full transition-all`} style={{ width: `${Math.min(pct, 100)}%` }} />
      </div>
      <span className={`text-xs font-bold w-10 text-right ${pct > 10 ? color : "text-tm-muted"}`}>
        {pct < 1 ? "<1%" : `${pct.toFixed(0)}%`}
      </span>
      <span className="text-[9px] text-tm-muted w-16 text-right">
        ${(t.volume / 1e6).toFixed(1)}M
      </span>
    </div>
  );
}
