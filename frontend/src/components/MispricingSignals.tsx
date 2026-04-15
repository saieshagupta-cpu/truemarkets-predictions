"use client";

import type { MispricingSignal } from "@/lib/api";

interface MispricingSignalsProps {
  signals: MispricingSignal[];
  currentPrice: number;
  symbol: string;
  hasPolymarket: boolean;
}

export default function MispricingSignals({ signals, currentPrice, symbol, hasPolymarket }: MispricingSignalsProps) {
  const actionable = signals.filter((s) => s.poly_prob !== null);
  const modelOnly = signals.filter((s) => s.poly_prob === null);

  return (
    <div>
      {/* Explainer */}
      <div className="bg-tm-card border border-tm-border rounded-xl p-4 mb-4">
        <h2 className="text-sm font-semibold mb-1">
          {hasPolymarket ? "Model vs Polymarket" : "Model Predictions"}
        </h2>
        <p className="text-xs text-tm-muted leading-relaxed">
          {hasPolymarket ? (
            <>
              Side-by-side comparison of our <span className="text-tm-accent">GRU + XGBoost ensemble</span> vs
              <span className="text-tm-blue"> Polymarket</span> crowd predictions.
              Both estimate the probability of {symbol} reaching each price target within 30 days.
            </>
          ) : (
            <>
              Our ensemble model estimates the probability that {symbol} reaches each price target within 30 days.
              Based on 5 years of price patterns, on-chain data, and True Markets sentiment.
            </>
          )}
        </p>
      </div>

      {/* Actionable signals (have Polymarket comparison) */}
      {actionable.length > 0 && (
        <div className="space-y-2.5 mb-4">
          {actionable
            .sort((a, b) => Number(a.threshold) - Number(b.threshold))
            .map((signal) => (
              <ComparisonRow key={signal.threshold} signal={signal} symbol={symbol} />
            ))}
        </div>
      )}

      {/* Model-only predictions */}
      {modelOnly.length > 0 && (
        <div>
          {hasPolymarket && (
            <p className="text-xs text-tm-muted mb-2 uppercase tracking-wider">Model-only (no Polymarket match)</p>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {modelOnly
              .sort((a, b) => Number(a.threshold) - Number(b.threshold))
              .map((s) => (
                <ModelOnlyCard key={s.threshold} signal={s} symbol={symbol} />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ComparisonRow({ signal, symbol }: { signal: MispricingSignal; symbol: string }) {
  const ourPct = Math.round(signal.our_prob * 100);
  const polyPct = Math.round((signal.poly_prob ?? 0) * 100);
  const threshold = Number(signal.threshold).toLocaleString();
  const isUp = signal.direction === "up";

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      {/* Threshold header */}
      <div className="flex items-center gap-2 mb-3">
        <span className={`text-sm ${isUp ? "text-tm-green" : "text-tm-red"}`}>
          {isUp ? "\u2191" : "\u2193"}
        </span>
        <span className="font-semibold">${threshold}</span>
        <span className="text-xs text-tm-muted">
          {isUp ? "reaches" : "drops to"} within 30d
        </span>
      </div>

      {/* Dual bars */}
      <div className="space-y-1.5 mb-2">
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-tm-accent font-medium w-16">Our Model</span>
          <div className="flex-1 bg-tm-border/40 rounded-full h-2.5 overflow-hidden">
            <div className="bg-tm-accent h-full rounded-full transition-all" style={{ width: `${ourPct}%` }} />
          </div>
          <span className="text-sm font-bold text-tm-accent w-10 text-right">{ourPct}%</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-tm-blue font-medium w-16">Polymarket</span>
          <div className="flex-1 bg-tm-border/40 rounded-full h-2.5 overflow-hidden">
            <div className="bg-tm-blue h-full rounded-full transition-all" style={{ width: `${polyPct}%` }} />
          </div>
          <span className="text-sm font-bold text-tm-blue w-10 text-right">{polyPct}%</span>
        </div>
      </div>

      {/* Model info */}
      <div className="text-[10px] text-tm-muted">
        <span>GRU + XGBoost agreement ensemble (61.2% OOS)</span>
      </div>
    </div>
  );
}

function ModelOnlyCard({ signal, symbol }: { signal: MispricingSignal; symbol: string }) {
  const pct = Math.round(signal.our_prob * 100);
  const threshold = Number(signal.threshold).toLocaleString();
  const isUp = signal.direction === "up";

  return (
    <div className="bg-tm-card border border-tm-border rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <span className={isUp ? "text-tm-green" : "text-tm-red"}>{isUp ? "\u2191" : "\u2193"}</span>
          <span className="font-semibold">${threshold}</span>
        </div>
        <span className={`text-lg font-bold ${pct > 50 ? "text-tm-green" : pct > 20 ? "text-tm-yellow" : "text-tm-muted"}`}>
          {pct}%
        </span>
      </div>
      <div className="w-full bg-tm-border/40 rounded-full h-2 mb-1.5">
        <div className={`h-full rounded-full ${isUp ? "bg-tm-green" : "bg-tm-red"}`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-[10px] text-tm-muted">
        {pct}% chance {symbol} {isUp ? "reaches" : "drops to"} ${threshold} within 30 days
      </p>
    </div>
  );
}
