"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

interface MarketStats {
  price: number;
  change_24h_pct: number;
  change_24h_usd: number;
  market_cap: number;
  volume_24h: number;
  high_24h: number;
  low_24h: number;
  ath: number;
  atl: number;
  circulating_supply: number;
  max_supply: number | null;
  price_change_7d: number;
  price_change_30d: number;
  price_change_1y: number;
  fear_greed: { current?: { value: number; classification: string } };
}

const PERIODS = [
  { label: "1D", days: "1" },
  { label: "5D", days: "5" },
  { label: "1M", days: "30" },
  { label: "6M", days: "180" },
  { label: "YTD", days: "ytd" },
  { label: "1Y", days: "365" },
];

interface EnhancedSignals {
  fear_greed: number;
  fear_greed_class: string;
  sentiment: string;
  order_flow_pressure: string;
  recommendation: string;
}

export default function MarketView() {
  const [stats, setStats] = useState<MarketStats | null>(null);
  const [enhanced, setEnhanced] = useState<EnhancedSignals | null>(null);
  const [chart, setChart] = useState<number[][]>([]);
  const [activePeriod, setActivePeriod] = useState("1");
  const [loading, setLoading] = useState(true);
  const [chartLoading, setChartLoading] = useState(false);

  // Initial load: full stats + chart + enhanced signals from prediction engine
  useEffect(() => {
    async function load() {
      try {
        const [statsRes, chartRes, mispRes] = await Promise.allSettled([
          fetch(`${API_BASE}/market-stats/bitcoin`).then(r => r.json()),
          fetch(`${API_BASE}/chart/bitcoin?days=1`).then(r => r.json()),
          fetch(`${API_BASE}/mispricing/bitcoin?_t=${Date.now()}`, { cache: "no-store" }).then(r => r.json()),
        ]);
        if (statsRes.status === "fulfilled") setStats(statsRes.value);
        if (chartRes.status === "fulfilled" && chartRes.value.prices) setChart(chartRes.value.prices);
        if (mispRes.status === "fulfilled" && mispRes.value.indicators) {
          const m = mispRes.value;
          // Use TM MCP sentiment from market-stats (same source as prediction page)
          const tmSentiment = statsRes.status === "fulfilled" ? statsRes.value.tm_sentiment : null;
          setEnhanced({
            fear_greed: m.indicators.fear_greed ?? m.indicators.rsi,
            fear_greed_class: m.indicators.fear_greed_classification || m.sentiment_signal?.fear_greed || "Neutral",
            sentiment: tmSentiment ? (tmSentiment.charAt(0).toUpperCase() + tmSentiment.slice(1)) : (m.sentiment_signal?.overall_signal || "Neutral"),
            order_flow_pressure: m.order_flow?.pressure || "neutral",
            recommendation: m.recommended_trade?.primary_side || m.recommended_trade?.side || "hold",
          });
        } else if (statsRes.status === "fulfilled" && statsRes.value.tm_sentiment) {
          // Fallback: use market-stats TM sentiment even without mispricing data
          const s = statsRes.value;
          setEnhanced({
            fear_greed: s.fear_greed?.current?.value || 50,
            fear_greed_class: s.fear_greed?.current?.classification || "Neutral",
            sentiment: s.tm_sentiment.charAt(0).toUpperCase() + s.tm_sentiment.slice(1),
            order_flow_pressure: "neutral",
            recommendation: "hold",
          });
        }
      } catch { /* */ }
      finally { setLoading(false); }
    }
    load();
  }, []);

  // Price poll every 30 seconds
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/price/bitcoin?_t=${Date.now()}`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (data.price) {
            setStats(prev => {
              if (!prev) return prev;
              // Compute actual USD change from percentage
              const basePrice = prev.price / (1 + prev.change_24h_pct / 100);
              const newUsdChange = data.price - basePrice;
              return {
                ...prev,
                price: data.price,
                change_24h_pct: data.change_24h,
                change_24h_usd: newUsdChange,
                volume_24h: data.volume_24h || prev.volume_24h,
              };
            });
          }
        }
      } catch { /* silent */ }
    }, 15_000);  // 15 seconds
    return () => clearInterval(interval);
  }, []);

  // Full stats refresh every 2 minutes
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/market-stats/bitcoin?_t=${Date.now()}`, { cache: "no-store" });
        if (res.ok) setStats(await res.json());
      } catch { /* silent */ }
    }, 120_000);
    return () => clearInterval(interval);
  }, []);

  // Auto-refresh 1D chart every 2 minutes
  useEffect(() => {
    if (activePeriod !== "1") return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/chart/bitcoin?days=1&_t=${Date.now()}`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (data.prices && data.prices.length > 3) setChart(data.prices);
        }
      } catch { /* silent */ }
    }, 120_000);
    return () => clearInterval(interval);
  }, [activePeriod]);

  const loadChart = async (days: string) => {
    setActivePeriod(days);
    setChartLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chart/bitcoin?days=${days}`);
      if (res.ok) {
        const data = await res.json();
        if (data.prices?.length) setChart(data.prices);
      }
    } catch { /* */ }
    finally { setChartLoading(false); }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="w-8 h-8 border-2 border-tm-accent border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!stats) return <p className="text-center text-tm-muted py-20">Failed to load market data</p>;

  // Use enhanced FG from prediction engine (includes order flow), fallback to raw
  const fgValue = enhanced?.fear_greed ?? stats.fear_greed?.current?.value ?? 50;
  const fgClass = enhanced?.fear_greed_class ?? stats.fear_greed?.current?.classification ?? "Neutral";

  const fmt = (n: number, dec = 2) => n.toLocaleString(undefined, { maximumFractionDigits: dec });
  const fmtB = (n: number) => n >= 1e12 ? `$${(n / 1e12).toFixed(2)}T` : `$${(n / 1e9).toFixed(1)}B`;
  const fmtPct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(2)}%`;
  const pctColor = (n: number) => n >= 0 ? "text-tm-green" : "text-tm-red";

  const chartFirst = chart.length > 0 ? chart[0][1] : 0;
  const chartLast = chart.length > 0 ? chart[chart.length - 1][1] : 0;
  const chartUp = chartLast >= chartFirst;

  return (
    <div className="max-w-2xl mx-auto">
      {/* Price Hero */}
      <div className="text-center pt-6 pb-3">
        <div className="flex items-center justify-center gap-2 mb-2">
          <div className="w-7 h-7 rounded-full bg-orange-500/20 flex items-center justify-center text-xs font-bold text-orange-400">B</div>
          <span className="text-sm text-tm-muted">Bitcoin <span className="text-tm-muted/50">BTC</span></span>
        </div>
        <p className="text-4xl font-bold mb-1">${fmt(stats.price)}</p>
        <p className={`text-sm ${pctColor(stats.change_24h_pct)}`}>
          {stats.change_24h_usd >= 0 ? "+" : ""}${fmt(Math.abs(stats.change_24h_usd))} ({fmtPct(stats.change_24h_pct)}) <span className="text-tm-muted">Today</span>
        </p>
      </div>

      {/* Period tabs */}
      <div className="flex justify-center items-center gap-1 mb-1 px-2">
        {PERIODS.map((p) => (
          <button
            key={p.days}
            onClick={() => loadChart(p.days)}
            className={`px-3 py-1.5 text-xs font-medium transition-all border-b-2 ${
              activePeriod === p.days
                ? "text-tm-accent border-tm-accent"
                : "text-tm-muted border-transparent hover:text-tm-text"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="h-44 mb-4 relative">
        {chartLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-tm-bg/60 z-10">
            <div className="w-5 h-5 border-2 border-tm-accent border-t-transparent rounded-full animate-spin" />
          </div>
        )}
        {chart.length > 1 ? (
          <PriceChart data={chart} isUp={chartUp} />
        ) : (
          <div className="h-full flex items-center justify-center text-xs text-tm-muted">Loading chart data...</div>
        )}
      </div>

      {/* Key Stats */}
      <div className="space-y-2.5">
        <h3 className="text-base font-semibold">Key Stats</h3>

        <div className="bg-tm-card border border-tm-border rounded-xl p-3 text-center">
          <p className="text-[10px] text-tm-muted mb-0.5">Market Cap</p>
          <p className="text-xl font-bold">{stats.market_cap ? fmtB(stats.market_cap) : "N/A"}</p>
        </div>

        <div className="bg-tm-card border border-tm-border rounded-xl p-3 grid grid-cols-3 gap-3 text-center">
          <div>
            <p className="text-[10px] text-tm-muted mb-0.5">24h Volume</p>
            <p className="text-base font-bold">{fmtB(stats.volume_24h)}</p>
          </div>
          <div>
            <p className="text-[10px] text-tm-muted mb-0.5">Day High</p>
            <p className="text-base font-bold">${fmt(stats.high_24h, 0)}</p>
          </div>
          <div>
            <p className="text-[10px] text-tm-muted mb-0.5">Day Low</p>
            <p className="text-base font-bold">${fmt(stats.low_24h, 0)}</p>
          </div>
        </div>

        {/* Fear & Greed */}
        <div className="bg-tm-card border border-tm-border rounded-xl p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-[10px] text-tm-muted uppercase tracking-wider">Fear & Greed</span>
            <span className={`text-sm font-bold ${
              fgValue <= 25 ? "text-tm-red" : fgValue <= 45 ? "text-tm-yellow" : fgValue <= 55 ? "text-tm-muted" : fgValue <= 75 ? "text-tm-green/80" : "text-tm-green"
            }`}>
              {fgValue} &mdash; {fgClass}
            </span>
          </div>
          <div className="relative h-2.5 rounded-full overflow-visible">
            <div className="absolute inset-0 rounded-full" style={{
              background: "linear-gradient(90deg, #ff6b6b 0%, #ffa06b 25%, #ffd93d 50%, #a8e06c 75%, #00d4aa 100%)"
            }} />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full border-2 border-white bg-tm-bg shadow-md transition-all"
              style={{ left: `calc(${Math.max(2, Math.min(98, fgValue))}% - 7px)` }}
            />
          </div>
          <div className="flex justify-between text-[9px] text-tm-muted mt-1">
            <span>Extreme Fear</span>
            <span>Extreme Greed</span>
          </div>
        </div>

        {/* Prediction signals — synced from prediction engine */}
        {enhanced && (
          <div className="bg-tm-card border border-tm-border rounded-xl p-3">
            <div className="grid grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-[10px] text-tm-muted mb-0.5">Sentiment</p>
                <p className={`text-sm font-bold ${
                  enhanced.sentiment.toLowerCase().includes("bullish") ? "text-tm-green" :
                  enhanced.sentiment.toLowerCase().includes("bearish") ? "text-tm-red" : "text-tm-muted"
                }`}>{enhanced.sentiment}</p>
              </div>
              <div>
                <p className="text-[10px] text-tm-muted mb-0.5">Order Flow</p>
                <p className={`text-sm font-bold ${
                  enhanced.order_flow_pressure.includes("buy") ? "text-tm-green" :
                  enhanced.order_flow_pressure.includes("sell") ? "text-tm-red" : "text-tm-muted"
                }`}>{enhanced.order_flow_pressure.replace("_", " ").replace(/\b\w/g, c => c.toUpperCase())}</p>
              </div>
              <div>
                <p className="text-[10px] text-tm-muted mb-0.5">Signal</p>
                <p className={`text-sm font-bold ${
                  enhanced.recommendation === "buy" ? "text-tm-green" : "text-tm-red"
                }`}>{enhanced.recommendation.toUpperCase()}</p>
              </div>
            </div>
          </div>
        )}

        {/* Stats rows */}
        <div className="bg-tm-card border border-tm-border rounded-xl p-3 space-y-2">
          {([
            ["24H Change", fmtPct(stats.change_24h_pct), pctColor(stats.change_24h_pct)],
            ["Absolute Change", `${stats.change_24h_usd >= 0 ? "+" : ""}$${fmt(Math.abs(stats.change_24h_usd))}`, pctColor(stats.change_24h_usd)],
            ["Day High", `$${fmt(stats.high_24h)}`, ""],
            ["Day Low", `$${fmt(stats.low_24h)}`, ""],
            ["All-Time High", `$${fmt(stats.ath)}`, ""],
            ["All-Time Low", `$${fmt(stats.atl)}`, ""],
            ["Circulating Supply", stats.circulating_supply ? `${(stats.circulating_supply / 1e6).toFixed(2)}M BTC` : "N/A", ""],
            ["Max Supply", stats.max_supply ? `${(stats.max_supply / 1e6).toFixed(0)}M BTC` : "N/A", ""],
          ] as [string, string, string][]).map(([label, value, color]) => (
            <div key={label} className="flex justify-between text-sm">
              <span className="text-tm-muted">{label}</span>
              <span className={`font-medium ${color}`}>{value}</span>
            </div>
          ))}
        </div>

        {/* Performance */}
        <div className="bg-tm-card border border-tm-border rounded-xl p-3">
          <p className="text-[10px] text-tm-muted uppercase tracking-wider mb-2">Performance</p>
          <div className="grid grid-cols-3 gap-3 text-center">
            {([
              ["7 Days", stats.price_change_7d],
              ["30 Days", stats.price_change_30d],
              ["1 Year", stats.price_change_1y],
            ] as [string, number][]).map(([label, val]) => (
              <div key={label}>
                <p className="text-[10px] text-tm-muted mb-0.5">{label}</p>
                <p className={`text-base font-bold ${pctColor(val)}`}>{fmtPct(val)}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function PriceChart({ data, isUp }: { data: number[][]; isUp: boolean }) {
  const prices = data.map(d => d[1]);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;

  const w = 600;
  const h = 170;
  const pad = 1;

  const points = prices.map((p, i) => {
    const x = pad + (i / (prices.length - 1)) * (w - pad * 2);
    const y = h - pad - ((p - min) / range) * (h - pad * 2 - 4) - 2;
    return `${x},${y}`;
  });

  const linePath = `M${points.join(" L")}`;
  const areaPath = `${linePath} L${w - pad},${h} L${pad},${h} Z`;
  const color = isUp ? "#00d4aa" : "#ff6b6b";

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-full" preserveAspectRatio="none">
      <defs>
        <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={areaPath} fill="url(#chartGrad)" />
      <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
