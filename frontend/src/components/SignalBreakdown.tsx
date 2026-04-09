"use client";

import type { SentimentSignal } from "@/lib/api";

interface SignalBreakdownProps {
  sentiment: SentimentSignal;
  indicators: { rsi: number; macd: number; volatility: number; fear_greed: number };
  weights: Record<string, number>;
}

export default function SignalBreakdown({ sentiment, indicators, weights }: SignalBreakdownProps) {
  const getRSILabel = (v: number) => v > 70 ? "Overbought" : v < 30 ? "Oversold" : "Neutral";

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

      {/* Weights bar */}
      <div className="mt-3 pt-3 border-t border-tm-border">
        <div className="flex items-center gap-1 h-2 rounded-full overflow-hidden">
          <div className="bg-tm-blue h-full rounded-l-full" style={{ width: `${(weights.lstm || 0.4) * 100}%` }} />
          <div className="bg-tm-yellow h-full" style={{ width: `${(weights.xgboost || 0.45) * 100}%` }} />
          <div className="bg-tm-purple h-full rounded-r-full" style={{ width: `${(weights.sentiment || 0.15) * 100}%` }} />
        </div>
        <div className="flex justify-between mt-1 text-[10px] text-tm-muted">
          <span className="text-tm-blue">LSTM {Math.round((weights.lstm || 0.4) * 100)}%</span>
          <span className="text-tm-yellow">XGBoost {Math.round((weights.xgboost || 0.45) * 100)}%</span>
          <span className="text-tm-purple">Sentiment {Math.round((weights.sentiment || 0.15) * 100)}%</span>
        </div>
      </div>
    </div>
  );
}
