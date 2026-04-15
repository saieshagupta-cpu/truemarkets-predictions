"use client";

import type { TechnicalIndicators as TechType, OrderFlowData } from "@/lib/api";

interface TechnicalIndicatorsProps {
  indicators: TechType;
  orderFlow: OrderFlowData;
  fearGreed: { current?: { value: number; classification: string } };
}

export default function TechnicalIndicators({ indicators, orderFlow, fearGreed }: TechnicalIndicatorsProps) {
  const fg = fearGreed?.current;

  return (
    <div className="space-y-3">
      {/* Technical Indicators */}
      <div className="bg-tm-card border border-tm-border rounded-xl p-4">
        <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-3">
          Technical Indicators
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {/* RSI */}
          <div className="bg-tm-bg rounded-lg px-3 py-2">
            <span className="text-[10px] text-tm-muted">RSI (14)</span>
            <div className="flex items-center justify-between mt-0.5">
              <span className="text-sm font-bold">{indicators.rsi.toFixed(1)}</span>
              <span className={`text-[10px] font-medium ${
                indicators.rsi > 70 ? "text-tm-red" : indicators.rsi < 30 ? "text-tm-green" : "text-tm-muted"
              }`}>{indicators.rsi_label}</span>
            </div>
            <div className="w-full bg-tm-border/40 rounded-full h-1 mt-1">
              <div className={`h-full rounded-full ${
                indicators.rsi > 70 ? "bg-tm-red" : indicators.rsi < 30 ? "bg-tm-green" : "bg-tm-accent"
              }`} style={{ width: `${indicators.rsi}%` }} />
            </div>
          </div>

          {/* MACD */}
          <div className="bg-tm-bg rounded-lg px-3 py-2">
            <span className="text-[10px] text-tm-muted">MACD</span>
            <div className="mt-0.5">
              <span className={`text-sm font-bold ${indicators.macd_histogram > 0 ? "text-tm-green" : "text-tm-red"}`}>
                {indicators.macd_histogram > 0 ? "+" : ""}{indicators.macd_histogram.toFixed(0)}
              </span>
            </div>
            <div className="text-[9px] text-tm-muted mt-0.5">
              {indicators.macd_histogram > 0 ? "Bullish" : "Bearish"} momentum
            </div>
          </div>

          {/* Bollinger */}
          <div className="bg-tm-bg rounded-lg px-3 py-2">
            <span className="text-[10px] text-tm-muted">Bollinger Position</span>
            <div className="flex items-center justify-between mt-0.5">
              <span className="text-sm font-bold">{(indicators.bollinger_position * 100).toFixed(0)}%</span>
              <span className={`text-[10px] ${
                indicators.bollinger_position < 0.2 ? "text-tm-green" : indicators.bollinger_position > 0.8 ? "text-tm-red" : "text-tm-muted"
              }`}>{indicators.bollinger_label}</span>
            </div>
            <div className="w-full bg-tm-border/40 rounded-full h-1 mt-1 relative">
              <div className="absolute h-3 w-0.5 bg-tm-accent rounded -top-1"
                style={{ left: `${indicators.bollinger_position * 100}%` }} />
            </div>
          </div>

          {/* Fear & Greed */}
          <div className="bg-tm-bg rounded-lg px-3 py-2">
            <span className="text-[10px] text-tm-muted">Fear & Greed</span>
            <div className="flex items-center justify-between mt-0.5">
              <span className="text-sm font-bold">{fg?.value ?? 50}</span>
              <span className={`text-[10px] font-medium ${
                (fg?.value ?? 50) < 30 ? "text-tm-red" : (fg?.value ?? 50) > 70 ? "text-tm-green" : "text-tm-muted"
              }`}>{fg?.classification ?? "Neutral"}</span>
            </div>
            <div className="w-full h-1 mt-1 rounded-full" style={{
              background: "linear-gradient(to right, #ff6b6b, #ffd93d, #00d4aa)"
            }}>
              <div className="relative">
                <div className="absolute h-3 w-0.5 bg-white rounded -top-1"
                  style={{ left: `${fg?.value ?? 50}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Order Flow */}
      <div className="bg-tm-card border border-tm-border rounded-xl p-4">
        <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-2">
          BTC Order Flow
        </h3>
        <p className="text-[9px] text-tm-muted mb-2">Source: {orderFlow.source}</p>

        {/* Buy/Sell volume bars */}
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-tm-green w-8">Buy</span>
            <div className="flex-1 bg-tm-border/30 rounded-full h-2 overflow-hidden">
              <div className="bg-tm-green h-full rounded-full transition-all"
                style={{ width: `${orderFlow.buy_sell_ratio * 100}%` }} />
            </div>
            <span className="text-[10px] text-tm-green w-10 text-right font-medium">
              {(orderFlow.buy_sell_ratio * 100).toFixed(0)}%
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-tm-red w-8">Sell</span>
            <div className="flex-1 bg-tm-border/30 rounded-full h-2 overflow-hidden">
              <div className="bg-tm-red h-full rounded-full transition-all"
                style={{ width: `${(1 - orderFlow.buy_sell_ratio) * 100}%` }} />
            </div>
            <span className="text-[10px] text-tm-red w-10 text-right font-medium">
              {((1 - orderFlow.buy_sell_ratio) * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Book imbalance */}
        <div className="mt-2 flex items-center justify-between text-[10px] text-tm-muted">
          <span>Book imbalance: <span className={orderFlow.imbalance > 0 ? "text-tm-green" : "text-tm-red"}>
            {orderFlow.imbalance > 0 ? "+" : ""}{(orderFlow.imbalance * 100).toFixed(1)}%
          </span></span>
          <span className={`font-medium ${
            orderFlow.pressure.includes("buy") ? "text-tm-green" :
            orderFlow.pressure.includes("sell") ? "text-tm-red" : "text-tm-muted"
          }`}>
            {orderFlow.pressure.replace("_", " ").toUpperCase()}
          </span>
        </div>

        {/* Volume counts */}
        <div className="mt-1.5 flex justify-between text-[9px] text-tm-muted">
          <span>{orderFlow.buy_count} buy trades ({orderFlow.buy_volume.toFixed(2)} BTC)</span>
          <span>{orderFlow.sell_count} sell trades ({orderFlow.sell_volume.toFixed(2)} BTC)</span>
        </div>
      </div>
    </div>
  );
}
