"use client";

import { useEffect, useState, useCallback } from "react";
import Header from "@/components/Header";
import MarketView from "@/components/MarketView";
import PredictionView from "@/components/PredictionView";
import type { PredictionData } from "@/lib/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";
const PREDICTION_INTERVAL = 30 * 1000;
const PRICE_POLL_INTERVAL = 15 * 1000;

export default function Home() {
  const [activeTab, setActiveTab] = useState<"market" | "prediction">("market");
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [nextRefresh, setNextRefresh] = useState(PREDICTION_INTERVAL / 1000);
  const [livePrice, setLivePrice] = useState<number>(0);

  const fetchPredictions = useCallback(async (isRefresh = false) => {
    if (!isRefresh) setPredLoading(true);
    try {
      const res = await fetch(`${API_BASE}/prediction/bitcoin?_t=${Date.now()}`, { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        setPrediction(data);
        setLivePrice(data.current_price);
        setLastUpdated(new Date());
      }
    } catch {
      // Keep stale data on error
    } finally {
      setPredLoading(false);
      setNextRefresh(PREDICTION_INTERVAL / 1000);
    }
  }, []);

  // Poll /price/bitcoin every 15s — SAME endpoint as MarketView
  // This ensures both pages always show the same price
  useEffect(() => {
    if (activeTab !== "prediction") return;
    const pollPrice = async () => {
      try {
        const res = await fetch(`${API_BASE}/price/bitcoin?_t=${Date.now()}`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (data.price > 0) setLivePrice(data.price);
        }
      } catch { /* keep last price */ }
    };
    pollPrice();
    const i = setInterval(pollPrice, PRICE_POLL_INTERVAL);
    return () => clearInterval(i);
  }, [activeTab]);

  // Fetch on tab switch (always fetch fresh, even if we have stale data)
  useEffect(() => {
    if (activeTab === "prediction" && !predLoading) fetchPredictions(!prediction);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

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

  // Override prediction price with live-polled price (same source as market page)
  const predictionWithLivePrice = prediction && livePrice > 0
    ? { ...prediction, current_price: livePrice }
    : prediction;

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
            data={predictionWithLivePrice!}
            loading={predLoading}
          />
        )}

        <footer className="max-w-2xl mx-auto text-center text-[10px] text-tm-muted py-4 border-t border-tm-border mt-4">
          True Markets Prediction Engine &bull; Not financial advice &bull; Powered by True Markets Gateway API
        </footer>
      </main>
    </div>
  );
}
