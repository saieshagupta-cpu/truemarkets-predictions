"use client";

import { useEffect, useState, useCallback } from "react";
import Header from "@/components/Header";
import MarketView from "@/components/MarketView";
import CoinHero from "@/components/CoinHero";
import MispricingSignals from "@/components/MispricingSignals";
import RecommendedTrade from "@/components/RecommendedTrade";
import PortfolioImpact from "@/components/PortfolioImpact";
import SignalBreakdown from "@/components/SignalBreakdown";
import type { MispricingData } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const PREDICTION_INTERVAL = 60 * 1000;  // 60 seconds

export default function Home() {
  const [activeTab, setActiveTab] = useState<"market" | "prediction">("market");
  const [mispricing, setMispricing] = useState<MispricingData | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [nextRefresh, setNextRefresh] = useState(PREDICTION_INTERVAL / 1000);
  const [portfolioKey, setPortfolioKey] = useState(0);

  const fetchPredictions = useCallback(async (isRefresh = false) => {
    if (!isRefresh) setPredLoading(true);
    setPredError(false);
    try {
      const res = await fetch(`${API_BASE}/mispricing/bitcoin?_t=${Date.now()}`, { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        if (data.signals) {
          setMispricing(data);
          setLastUpdated(new Date());
        }
      } else {
        if (!isRefresh) setPredError(true);
      }
    } catch {
      if (!isRefresh) setPredError(true);
    } finally {
      setPredLoading(false);
      setNextRefresh(PREDICTION_INTERVAL / 1000);
    }
  }, []);

  // Fetch on tab switch
  useEffect(() => {
    if (activeTab === "prediction" && !mispricing && !predLoading) fetchPredictions();
  }, [activeTab, mispricing, predLoading, fetchPredictions]);

  // Auto-refresh predictions every 30s
  useEffect(() => {
    if (activeTab !== "prediction") return;
    const i = setInterval(() => fetchPredictions(true), PREDICTION_INTERVAL);
    return () => clearInterval(i);
  }, [activeTab, fetchPredictions]);

  // Countdown timer
  useEffect(() => {
    const t = setInterval(() => setNextRefresh((p) => Math.max(0, p - 1)), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="min-h-screen bg-tm-bg">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        lastUpdated={lastUpdated}
        nextRefresh={nextRefresh}
        onRefresh={() => activeTab === "prediction" ? fetchPredictions(true) : window.location.reload()}
      />

      <main className="px-4 sm:px-6 py-4">
        {activeTab === "market" ? (
          <MarketView />
        ) : (
          <PredictionView
            mispricing={mispricing}
            loading={predLoading}
            error={predError}
            onRetry={() => fetchPredictions()}
            onOrderPlaced={() => setPortfolioKey((k) => k + 1)}
            portfolioKey={portfolioKey}
          />
        )}

        <footer className="max-w-2xl mx-auto text-center text-[10px] text-tm-muted py-4 border-t border-tm-border mt-4">
          True Markets Prediction Engine &bull; Not financial advice &bull; Powered by True Markets Gateway API
        </footer>
      </main>
    </div>
  );
}

function PredictionView({
  mispricing,
  loading,
  error,
  onRetry,
  onOrderPlaced,
  portfolioKey,
}: {
  mispricing: MispricingData | null;
  loading: boolean;
  error: boolean;
  onRetry: () => void;
  onOrderPlaced: () => void;
  portfolioKey: number;
}) {
  if (loading && !mispricing) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-tm-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-sm text-tm-muted">Analyzing BTC...</p>
        </div>
      </div>
    );
  }

  if (error && !mispricing) {
    return (
      <div className="flex items-center justify-center py-20 text-center">
        <div>
          <p className="text-sm text-tm-muted mb-2">Failed to load prediction data</p>
          <button onClick={onRetry} className="text-xs text-tm-accent hover:underline">Retry</button>
        </div>
      </div>
    );
  }

  if (!mispricing) return null;

  return (
    <div className="max-w-4xl mx-auto">
      <CoinHero mispricing={mispricing} market={null} />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div className="lg:col-span-8 space-y-4">
          {mispricing.recommended_trade && (
            <RecommendedTrade
              trade={mispricing.recommended_trade}
              onOrderPlaced={onOrderPlaced}
            />
          )}
          <MispricingSignals
            signals={mispricing.signals}
            currentPrice={mispricing.current_price}
            symbol="BTC"
            hasPolymarket={mispricing.polymarket_count > 0}
          />
        </div>

        <div className="lg:col-span-4 space-y-4">
          <SignalBreakdown
            sentiment={mispricing.sentiment_signal}
            indicators={mispricing.indicators}
            weights={{ lstm: 0.55, xgboost: 0.30, sentiment: 0.15 }}
            orderFlow={mispricing.order_flow}
          />
          <PortfolioImpact refreshKey={portfolioKey} />

          <details className="bg-tm-card border border-tm-border rounded-xl">
            <summary className="px-4 py-2.5 cursor-pointer text-xs text-tm-muted hover:text-tm-text">
              How it works
            </summary>
            <div className="px-4 pb-3 text-[11px] text-tm-muted border-t border-tm-border pt-2 space-y-1.5">
              <p>
                Ensemble of <span className="text-tm-blue">LSTM (40%)</span>,
                <span className="text-tm-yellow"> XGBoost (45%)</span>,
                <span className="text-tm-purple"> Sentiment (15%)</span>.
                Compares our 30-day probability forecasts against Polymarket crowd odds to find mispricings.
                Order flow from Polymarket microstructure + True Markets orders feeds into the recommendation.
              </p>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
