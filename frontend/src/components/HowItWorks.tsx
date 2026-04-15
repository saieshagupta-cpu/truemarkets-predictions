"use client";

interface HowItWorksProps {
  weights: Record<string, number>;
  backtest: Record<string, unknown>;
}

export default function HowItWorks({ weights, backtest }: HowItWorksProps) {
  const accuracy = ((backtest.test_accuracy as number) * 100 || 0).toFixed(1);
  const trainPeriod = (backtest.train_period as string) || "N/A";
  const testPeriod = (backtest.test_period as string) || "N/A";
  const nFeatures = (backtest.n_features as number) || 35;
  const topFeatures = (backtest.top_10_features as string[]) || [];

  return (
    <details className="bg-tm-card border border-tm-border rounded-xl">
      <summary className="px-4 py-2.5 cursor-pointer text-xs text-tm-muted hover:text-tm-text transition-colors">
        How it works
      </summary>
      <div className="px-4 pb-4 text-[11px] text-tm-muted border-t border-tm-border pt-3 space-y-3">

        {/* 6 Signals */}
        <div>
          <p className="text-white font-semibold text-xs mb-1.5">6 Signals, Weighted by Backtest</p>
          <div className="space-y-1 pl-1">
            <SignalRow name="Polymarket" weight={weights.polymarket} color="text-tm-blue"
              desc="Prediction market probabilities from Gamma API. 18 BTC price thresholds for April 2026." source="gamma-api.polymarket.com" />
            <SignalRow name="Order Flow" weight={weights.order_flow} color="text-tm-purple"
              desc="Real BTC buy/sell volume + order book depth from Binance BTCUSDT. 1000 recent trades + top 20 book levels." source="api.binance.com/api/v3" />
            <SignalRow name="Our Model" weight={weights.lightgbm} color="text-tm-green"
              desc={`CNN-LSTM (Omole & Enke 2024) with ${nFeatures} on-chain features + Boruta selection. Next-day direction. ${accuracy}% test accuracy.`} source="BGeometrics on-chain API" />
            <SignalRow name="Technical" weight={weights.technical} color="text-tm-yellow"
              desc="RSI (14), MACD (12,26,9), Bollinger Band position. Computed from TrueMarkets MCP price data." source="TrueMarkets MCP" />
            <SignalRow name="TM Sentiment" weight={weights.sentiment} color="text-tm-accent"
              desc="AI-generated sentiment analysis from 30+ crypto news sources." source="TrueMarkets MCP get_asset_summary" />
            <SignalRow name="Fear & Greed" weight={weights.fear_greed} color="text-tm-red"
              desc="Contrarian at extremes: extreme fear = buy, extreme greed = sell. Daily index from alternative.me." source="api.alternative.me/fng" />
          </div>
        </div>

        {/* Model Details */}
        <div>
          <p className="text-white font-semibold text-xs mb-1">Our Model Details</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 pl-1">
            <span>Test Accuracy:</span><span className="text-white font-medium">{accuracy}%</span>
            <span>Training Data:</span><span className="text-white">{trainPeriod}</span>
            <span>Test Period:</span><span className="text-white">{testPeriod}</span>
            <span>Features:</span><span className="text-white">{nFeatures} (Boruta-selected from 42)</span>
            <span>Horizon:</span><span className="text-white">Next-day direction</span>
            <span>Architecture:</span><span className="text-white">CNN-LSTM (Omole & Enke 2024)</span>
            <span>Data Source:</span><span className="text-white">BGeometrics on-chain API</span>
            <span>Window:</span><span className="text-white">5 days</span>
          </div>
          {topFeatures.length > 0 && (
            <div className="mt-1 pl-1">
              <span>Top features: </span>
              <span className="text-white">{topFeatures.slice(0, 5).join(", ")}</span>
            </div>
          )}
        </div>

        {/* Data Sources */}
        <div>
          <p className="text-white font-semibold text-xs mb-1">Data Sources</p>
          <table className="w-full text-[10px]">
            <tbody>
              <tr><td className="pr-2 py-0.5">BTC Price</td><td className="text-white">TrueMarkets MCP (same as market page)</td></tr>
              <tr><td className="pr-2 py-0.5">Polymarket</td><td className="text-white">Gamma API (public, no auth)</td></tr>
              <tr><td className="pr-2 py-0.5">Order Flow</td><td className="text-white">Binance BTCUSDT (public, no auth)</td></tr>
              <tr><td className="pr-2 py-0.5">Model Training</td><td className="text-white">BGeometrics 5yr on-chain (HODL waves, MVRV, SOPR, etc.)</td></tr>
              <tr><td className="pr-2 py-0.5">Fear & Greed</td><td className="text-white">alternative.me/fng (daily)</td></tr>
              <tr><td className="pr-2 py-0.5">Sentiment</td><td className="text-white">TrueMarkets MCP AI summary</td></tr>
            </tbody>
          </table>
        </div>

        <p className="text-[10px]">
          All data refreshes every <span className="text-white">30 seconds</span>.
          Price on this page is from the <span className="text-white">same source</span> as the market page (TrueMarkets MCP).
        </p>
      </div>
    </details>
  );
}

function SignalRow({ name, weight, color, desc, source }: {
  name: string; weight: number; color: string; desc: string; source: string;
}) {
  const pct = weight ? `${Math.round(weight * 100)}%` : "—";
  return (
    <div className="py-0.5">
      <span className={`${color} font-medium`}>{name}</span>
      <span className="text-white ml-1">{pct}</span>
      <span className="mx-1">&mdash;</span>
      <span>{desc}</span>
      <span className="text-tm-muted/60 ml-1">({source})</span>
    </div>
  );
}
