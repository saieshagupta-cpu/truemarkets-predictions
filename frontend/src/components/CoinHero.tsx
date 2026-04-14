"use client";

import type { MispricingData, MarketData } from "@/lib/api";

interface CoinHeroProps {
  mispricing: MispricingData;
  market: MarketData | null;
}

export default function CoinHero({ mispricing, market }: CoinHeroProps) {
  const price = market?.price ?? mispricing.current_price;
  const change = market?.change_24h && market.change_24h !== 0 ? market.change_24h : null;
  const conf = Math.round(mispricing.confidence * 100);
  const sentiment = mispricing.sentiment_signal;

  const getSentimentColor = (s: string) => {
    if (s.toLowerCase().includes("bullish")) return "text-tm-green";
    if (s.toLowerCase().includes("bearish")) return "text-tm-red";
    return "text-tm-muted";
  };

  const getConfColor = (c: number) => {
    if (c >= 70) return "bg-tm-green";
    if (c >= 40) return "bg-tm-yellow";
    return "bg-tm-red";
  };

  const strongSignals = mispricing.signals.filter((s) => s.severity === "strong").length;
  const moderateSignals = mispricing.signals.filter((s) => s.severity === "moderate").length;

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
              {change !== null && (
                <span className={`text-sm font-medium ${change >= 0 ? "text-tm-green" : "text-tm-red"}`}>
                  {change >= 0 ? "+" : ""}{change.toFixed(2)}% <span className="text-tm-muted font-normal">24h</span>
                </span>
              )}
            </div>
            {market && (
              <div className="flex gap-4 mt-1 text-xs text-tm-muted">
                <span>Vol: ${(market.volume_24h / 1e9).toFixed(1)}B</span>
                <span>MCap: ${(market.market_cap / 1e12).toFixed(2)}T</span>
              </div>
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
