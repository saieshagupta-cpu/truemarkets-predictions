"use client";

import type { SentimentSignal, OrderFlow } from "@/lib/api";

interface SignalBreakdownProps {
  sentiment: SentimentSignal;
  indicators: { rsi: number; macd: number; volatility: number; fear_greed: number };
  weights: Record<string, number>;
  orderFlow: OrderFlow | null;
}

export default function SignalBreakdown({ sentiment, indicators, weights, orderFlow }: SignalBreakdownProps) {
  const getRSILabel = (v: number) => v > 70 ? "Overbought" : v < 30 ? "Oversold" : "Neutral";

  const getPressureColor = (p: string) => {
    if (p.includes("buy")) return "text-tm-green";
    if (p.includes("sell")) return "text-tm-red";
    return "text-tm-muted";
  };

  const getPressureLabel = (p: string) => {
    const labels: Record<string, string> = {
      strong_buy: "Strong Buy",
      buy: "Buy",
      neutral: "Neutral",
      sell: "Sell",
      strong_sell: "Strong Sell",
    };
    return labels[p] || "Neutral";
  };

  const of = orderFlow;
  const details = of?.polymarket_flow?.details;

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-5">
      <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-3">
        Model Signals
      </h3>

      <div className="grid grid-cols-2 gap-3 text-xs">
        <div className="bg-tm-bg rounded-lg px-3 py-2">
          <span className="text-tm-muted">RSI (14)</span>
          <div className="flex items-center justify-between mt-0.5">
            <span className="font-medium">{indicators.rsi.toFixed(1)}</span>
            <span className={indicators.rsi > 70 ? "text-tm-red" : indicators.rsi < 30 ? "text-tm-green" : "text-tm-muted"}>
              {getRSILabel(indicators.rsi)}
            </span>
          </div>
        </div>

        <div className="bg-tm-bg rounded-lg px-3 py-2">
          <span className="text-tm-muted">MACD</span>
          <div className="mt-0.5">
            <span className={`font-medium ${indicators.macd > 0 ? "text-tm-green" : "text-tm-red"}`}>
              {indicators.macd > 0 ? "+" : ""}{indicators.macd.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="bg-tm-bg rounded-lg px-3 py-2">
          <span className="text-tm-muted">Volatility (20d)</span>
          <div className="mt-0.5">
            <span className="font-medium">{(indicators.volatility * 100).toFixed(2)}%</span>
          </div>
        </div>

        <div className="bg-tm-bg rounded-lg px-3 py-2">
          <span className="text-tm-muted">Sentiment</span>
          <div className="mt-0.5">
            <span className={`font-medium ${
              sentiment.overall_signal.toLowerCase().includes("bullish") ? "text-tm-green" :
              sentiment.overall_signal.toLowerCase().includes("bearish") ? "text-tm-red" : "text-tm-muted"
            }`}>{sentiment.overall_signal}</span>
          </div>
        </div>
      </div>

      {/* Order Flow */}
      {of && (
        <div className="mt-3 pt-3 border-t border-tm-border">
          <p className="text-[10px] text-tm-muted uppercase tracking-wider mb-2">Order Flow</p>
          <div className="bg-tm-bg rounded-lg px-3 py-2">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-tm-muted">Pressure</span>
              <span className={`text-xs font-bold ${getPressureColor(of.pressure)}`}>
                {getPressureLabel(of.pressure)}
              </span>
            </div>
            {/* Buy/Sell pressure bar — based on actual volume ratio */}
            <div className="flex h-2 rounded-full overflow-hidden bg-tm-border/50">
              {(() => {
                const upVol = details?.up_volume_24h || 0;
                const downVol = details?.down_volume_24h || 0;
                const totalVol = upVol + downVol;
                const buyWidth = totalVol > 0 ? Math.max(5, Math.min(95, (upVol / totalVol) * 100)) : 50;
                return (
                  <>
                    <div className="bg-tm-green h-full transition-all" style={{ width: `${buyWidth}%` }} />
                    <div className="bg-tm-red h-full transition-all" style={{ width: `${100 - buyWidth}%` }} />
                  </>
                );
              })()}
            </div>
            <div className="flex justify-between mt-1 text-[9px] text-tm-muted">
              <span>Buy pressure</span>
              <span>Sell pressure</span>
            </div>
            {details && (details.up_volume_24h || 0) + (details.down_volume_24h || 0) > 0 && (
              <div className="flex justify-between mt-1.5 text-[10px] text-tm-muted">
                <span>24h vol: <span className="text-tm-green">${((details.up_volume_24h || 0) / 1000).toFixed(0)}K up</span></span>
                <span><span className="text-tm-red">${((details.down_volume_24h || 0) / 1000).toFixed(0)}K down</span></span>
              </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
}
