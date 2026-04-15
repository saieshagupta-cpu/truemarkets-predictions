"use client";

interface HowItWorksProps {
  weights: Record<string, number>;
  backtest: Record<string, unknown>;
}

export default function HowItWorks({ weights, backtest }: HowItWorksProps) {
  const accuracy = ((backtest.test_accuracy as number) * 100 || 0).toFixed(1);
  const precision = ((backtest.test_precision as number) * 100 || 0).toFixed(1);
  const recall = ((backtest.test_recall as number) * 100 || 0).toFixed(1);
  const f1 = ((backtest.test_f1 as number) * 100 || 0).toFixed(1);
  const auc = ((backtest.test_auc as number) || 0).toFixed(4);
  const mcc = ((backtest.test_mcc as number) || 0).toFixed(3);
  const trainPeriod = (backtest.train_period as string) || "N/A";
  const testPeriod = (backtest.test_period as string) || "N/A";
  const nFeatSelected = (backtest.n_features_selected as number) || 30;
  const nFeatTotal = (backtest.n_features_total as number) || 216;
  const testSamples = (backtest.test_samples as number) || 359;
  const topFeatures = (backtest.top_10_features as string[]) || [];
  const cm = (backtest.confusion_matrix as number[][]) || [[0,0],[0,0]];

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
              desc="Real BTC buy/sell volume from Binance.US (1000 trades) + Coinbase (100 trades) + order book depth." source="Binance.US + Coinbase" />
            <SignalRow name="Our Model" weight={weights.lightgbm} color="text-tm-green"
              desc={`On-chain ensemble with ${nFeatSelected} Boruta-selected features. Next-day BTC direction. ${accuracy}% test accuracy on ${testSamples} OOS days.`} source="BGeometrics Premium" />
            <SignalRow name="Technical" weight={weights.technical} color="text-tm-yellow"
              desc="RSI (14), MACD (12,26,9), Bollinger Band position." source="TrueMarkets MCP" />
            <SignalRow name="TM Sentiment" weight={weights.sentiment} color="text-tm-accent"
              desc="AI sentiment from 30+ crypto news sources." source="TrueMarkets MCP" />
            <SignalRow name="Fear & Greed" weight={weights.fear_greed} color="text-tm-red"
              desc="Contrarian: extreme fear = buy, extreme greed = sell." source="alternative.me" />
          </div>
        </div>

        {/* Model Backtest */}
        <div>
          <p className="text-white font-semibold text-xs mb-1">Our Model &mdash; Backtest Results</p>
          <div className="grid grid-cols-3 gap-x-3 gap-y-0.5 pl-1 mb-1.5">
            <div><span>Accuracy: </span><span className="text-tm-green font-bold">{accuracy}%</span></div>
            <div><span>Precision: </span><span className="text-white">{precision}%</span></div>
            <div><span>Recall: </span><span className="text-white">{recall}%</span></div>
            <div><span>F1 Score: </span><span className="text-white">{f1}%</span></div>
            <div><span>AUC-ROC: </span><span className="text-white">{auc}</span></div>
            <div><span>MCC: </span><span className="text-white">{mcc}</span></div>
          </div>
          <div className="pl-1 mb-1.5">
            <span>Confusion Matrix: </span>
            <span className="text-white font-mono text-[10px]">TN={cm[0]?.[0]} FP={cm[0]?.[1]} FN={cm[1]?.[0]} TP={cm[1]?.[1]}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 pl-1">
            <span>Architecture:</span><span className="text-white">GradientBoosting + Boruta (Omole & Enke 2024)</span>
            <span>Features:</span><span className="text-white">{nFeatSelected} selected from {nFeatTotal} (42 on-chain + 168 rate-of-change + 6 price)</span>
            <span>Key Innovation:</span><span className="text-white">Rate-of-change features (1d/3d/7d/14d change of each on-chain metric)</span>
            <span>Training:</span><span className="text-white">{trainPeriod}</span>
            <span>Test (OOS):</span><span className="text-white">{testPeriod} ({testSamples} days)</span>
            <span>On-chain Data:</span><span className="text-white">BGeometrics Premium (27 endpoints, 5 years daily)</span>
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
              <tr><td className="pr-2 py-0.5">On-chain (model)</td><td className="text-white">BGeometrics Premium API &mdash; HODL waves, MVRV, SOPR, NVT, Puell, exchange flows, etc.</td></tr>
              <tr><td className="pr-2 py-0.5">Order Flow</td><td className="text-white">Binance.US + Coinbase (real BTC trades + order book)</td></tr>
              <tr><td className="pr-2 py-0.5">Polymarket</td><td className="text-white">Gamma API (18 threshold markets, public)</td></tr>
              <tr><td className="pr-2 py-0.5">Fear & Greed</td><td className="text-white">alternative.me/fng (daily)</td></tr>
              <tr><td className="pr-2 py-0.5">Sentiment</td><td className="text-white">TrueMarkets MCP get_asset_summary (30+ news sources)</td></tr>
            </tbody>
          </table>
        </div>

        <p className="text-[10px]">
          All data refreshes every <span className="text-white">30 seconds</span>.
          Price is from the <span className="text-white">same source</span> on both pages (TrueMarkets MCP).
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
