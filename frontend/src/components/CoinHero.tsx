"use client";

import type { MispricingData, MarketData } from "@/lib/api";

interface CoinHeroProps {
  mispricing: MispricingData;
  market: MarketData | null;
}

export default function CoinHero({ mispricing, market }: CoinHeroProps) {
  const price = market?.price ?? mispricing.current_price;

  // 24h change: prefer mispricing data (always available), fallback to market
  const changePct = (mispricing as any).change_24h_pct ?? market?.change_24h ?? null;
  const changeUsd = (mispricing as any).change_24h_usd ?? null;
  const hasChange = changePct !== null && changePct !== 0;

  const sentiment = mispricing.sentiment_signal;

  const getSentimentColor = (s: string) => {
    if (s.toLowerCase().includes("bullish")) return "text-tm-green";
    if (s.toLowerCase().includes("bearish")) return "text-tm-red";
    return "text-tm-muted";
  };

  const strongSignals = mispricing.signals.filter((s) => s.severity === "strong").length;
  const moderateSignals = mispricing.signals.filter((s) => s.severity === "moderate").length;

  const fmt = (n: number) => n.toLocaleString(undefined, { maximumFractionDigits: 2 });

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-5 mb-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-5">
          <div>
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold">{mispricing.symbol}</span>
              <span className="text-xl font-semibold">
                ${price.toLocaleString(undefined, { maximumFractionDigits: price > 100 ? 0 : 2 })}
              </span>
            </div>
            {hasChange && (
              <p className={`text-sm mt-0.5 ${changePct >= 0 ? "text-tm-green" : "text-tm-red"}`}>
                {changeUsd !== null && (
                  <span>{changeUsd >= 0 ? "+" : ""}${fmt(Math.abs(changeUsd))} </span>
                )}
                ({changePct >= 0 ? "+" : ""}{changePct.toFixed(2)}%) <span className="text-tm-muted">Today</span>
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="text-center">
            <p className={`text-sm font-semibold ${getSentimentColor(sentiment.overall_signal)}`}>
              {sentiment.overall_signal}
            </p>
            <p className="text-[10px] text-tm-muted uppercase">Sentiment</p>
          </div>

          <div className="text-center">
            <p className="text-sm font-semibold">{sentiment.fear_greed_value}</p>
            <p className="text-[10px] text-tm-muted uppercase">Fear/Greed</p>
          </div>

        </div>
      </div>
    </div>
  );
}
