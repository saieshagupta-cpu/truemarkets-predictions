"use client";

import { useState } from "react";
import type { PredictionData } from "@/lib/api";
import BuySellPanel from "@/components/BuySellPanel";
import PolymarketTable from "@/components/PolymarketTable";
import TechnicalIndicators from "@/components/TechnicalIndicators";
import OrderFlowPanel from "@/components/OrderFlowPanel";
import PortfolioBox from "@/components/PortfolioBox";
import HowItWorks from "@/components/HowItWorks";

interface PredictionViewProps {
  data: PredictionData;
  loading: boolean;
}

export default function PredictionView({ data, loading }: PredictionViewProps) {
  const [portfolioKey, setPortfolioKey] = useState(0);

  if (loading && !data) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-tm-accent border-t-transparent rounded-full animate-spin mb-3" />
        <p className="text-sm text-tm-muted">Analyzing BTC...</p>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      {/* Price header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold">
              ${data.current_price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </span>
            <span className={`text-sm font-medium ${data.change_24h >= 0 ? "text-tm-green" : "text-tm-red"}`}>
              {data.change_24h >= 0 ? "+" : ""}{data.change_24h.toFixed(2)}%
            </span>
          </div>
          <p className="text-xs text-tm-muted mt-0.5">Bitcoin (BTC/USD) &bull; Source: TrueMarkets</p>
        </div>
        <div className="text-right">
          {(() => {
            const dir = (data.sentiment_direction || "neutral").toLowerCase();
            const isBullish = dir === "bullish";
            const isBearish = dir === "bearish";
            return (
              <>
                <div className={`text-xs font-bold px-2.5 py-1 rounded inline-flex items-center gap-1.5 ${
                  isBullish ? "bg-tm-green/15 text-tm-green border border-tm-green/20" :
                  isBearish ? "bg-tm-red/15 text-tm-red border border-tm-red/20" :
                  "bg-tm-card text-tm-muted border border-tm-border"
                }`}>
                  <span className="text-[9px] uppercase tracking-wider opacity-70">TM Sentiment</span>
                  <span>{isBullish ? "Bullish" : isBearish ? "Bearish" : "Neutral"}</span>
                </div>
                {data.sentiment_summary && (
                  <p className="text-[9px] text-tm-muted mt-0.5 max-w-[200px] leading-tight truncate">
                    {data.sentiment_summary.slice(0, 80)}
                  </p>
                )}
              </>
            );
          })()}
        </div>
      </div>

      {/* Row 1: Buy / Sell */}
      <BuySellPanel
        recommendedSide={data.recommended_side}
        confidence={data.confidence}
        buySignals={data.buy_signals}
        sellSignals={data.sell_signals}
        neutralSignals={data.neutral_signals || []}
        buyCount={data.buy_count}
        sellCount={data.sell_count}
        neutralCount={data.neutral_count || 0}
        totalSignals={data.total_signals}
        currentPrice={data.current_price}
        onOrderPlaced={() => setPortfolioKey((k) => k + 1)}
      />

      {/* Row 2: Polymarket (left) | Portfolio + Tech + Order Flow (right, stacked) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-7">
          <PolymarketTable
            thresholds={data.polymarket_thresholds}
            currentPrice={data.current_price}
          />
        </div>
        <div className="lg:col-span-5 space-y-4">
          <PortfolioBox refreshKey={portfolioKey} />
          <TechnicalIndicators
            indicators={data.technical_indicators}
            fearGreed={data.fear_greed}
          />
          <OrderFlowPanel orderFlow={data.order_flow} />
        </div>
      </div>

      {/* How it works */}
      <HowItWorks
        weights={data.weights}
        backtest={data.backtest_results}
      />
    </div>
  );
}
