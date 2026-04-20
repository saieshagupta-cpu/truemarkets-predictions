"use client";

interface HowItWorksProps {
  weights: Record<string, number>;
  backtest: Record<string, unknown>;
}

export default function HowItWorks({ weights }: HowItWorksProps) {
  return (
    <details className="bg-tm-card border border-tm-border rounded-xl">
      <summary className="px-4 py-2.5 cursor-pointer text-xs text-tm-muted hover:text-tm-text transition-colors">
        How it works
      </summary>
      <div className="px-4 pb-4 text-[11px] text-tm-muted border-t border-tm-border pt-3 space-y-3">

        {/* 6 Signals */}
        <div>
          <p className="text-white font-semibold text-xs mb-1.5">6 Signals, Weighted</p>
          <div className="space-y-1 pl-1">
            <SignalRow name="Polymarket" weight={weights.polymarket} color="text-tm-blue"
              desc="18 BTC price thresholds ($20K-$150K) for April 2026." source="Gamma API" />
            <SignalRow name="Order Flow" weight={weights.order_flow} color="text-tm-purple"
              desc="Real BTC buy/sell volume from Binance.US + Coinbase + order book depth." source="Binance.US + Coinbase" />
            <SignalRow name="Our Model" weight={weights.lightgbm} color="text-tm-green"
              desc="Gradient Boosting on 30 Boruta-selected on-chain features. Next-day BTC direction." source="BGeometrics Premium" />
            <SignalRow name="Technical" weight={weights.technical} color="text-tm-yellow"
              desc="RSI (14), MACD (12,26,9), Bollinger Band position." source="TrueMarkets MCP" />
            <SignalRow name="BTC Sentiment" weight={weights.sentiment} color="text-tm-accent"
              desc="AI sentiment from 30+ crypto news sources." source="TrueMarkets MCP" />
            <SignalRow name="Fear & Greed" weight={weights.fear_greed} color="text-tm-red"
              desc="Contrarian: extreme fear = buy, extreme greed = sell." source="alternative.me" />
          </div>
        </div>

        {/* Data Sources */}
        <div>
          <p className="text-white font-semibold text-xs mb-1">Data Sources</p>
          <table className="w-full text-[10px]">
            <tbody>
              <tr><td className="pr-2 py-0.5">BTC Price</td><td className="text-white">TrueMarkets MCP (same as market page)</td></tr>
              <tr><td className="pr-2 py-0.5">On-chain (model)</td><td className="text-white">BGeometrics Premium &mdash; HODL waves, MVRV, SOPR, NVT, Puell, exchange flows</td></tr>
              <tr><td className="pr-2 py-0.5">Order Flow</td><td className="text-white">Binance.US + Coinbase (real BTC trades + order book)</td></tr>
              <tr><td className="pr-2 py-0.5">Polymarket</td><td className="text-white">Gamma API (18 threshold markets)</td></tr>
              <tr><td className="pr-2 py-0.5">Fear & Greed</td><td className="text-white">alternative.me/fng</td></tr>
              <tr><td className="pr-2 py-0.5">Sentiment</td><td className="text-white">TrueMarkets MCP (30+ news sources)</td></tr>
            </tbody>
          </table>
        </div>

        <p className="text-[10px]">
          All data refreshes every <span className="text-white">30 seconds</span>.
        </p>
      </div>
    </details>
  );
}

function SignalRow({ name, weight, color, desc, source }: {
  name: string; weight: number; color: string; desc: string; source: string;
}) {
  const pct = weight ? `${Math.round(weight * 100)}%` : "\u2014";
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
