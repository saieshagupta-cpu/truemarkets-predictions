"use client";

import { useEffect, useState } from "react";
import { getBalances, getOrders } from "@/lib/api";
import type { BalancesResponse, OrdersResponse } from "@/lib/api";

interface PortfolioImpactProps {
  refreshKey: number;
}

export default function PortfolioImpact({ refreshKey }: PortfolioImpactProps) {
  const [balances, setBalances] = useState<BalancesResponse | null>(null);
  const [orders, setOrders] = useState<OrdersResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [b, o] = await Promise.allSettled([getBalances(), getOrders()]);
        if (b.status === "fulfilled") setBalances(b.value);
        if (o.status === "fulfilled") setOrders(o.value);
      } catch {
        // Silently fail
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [refreshKey]);

  const recentOrders = orders?.data?.slice(0, 5) || [];
  const assets = balances?.balances || [];

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-5">
      <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-4">
        Portfolio & Orders
      </h3>

      {loading ? (
        <div className="text-center py-4">
          <div className="w-6 h-6 border-2 border-tm-accent border-t-transparent rounded-full animate-spin mx-auto" />
        </div>
      ) : (
        <>
          {/* Balances */}
          <div className="mb-4">
            <p className="text-xs text-tm-muted uppercase tracking-wider mb-2">Balances</p>
            {assets.length > 0 ? (
              <div className="space-y-1">
                {assets.map((a, i) => (
                  <div key={i} className="flex justify-between text-sm bg-tm-bg rounded-lg px-3 py-2">
                    <span className="text-tm-muted">{a.asset_name || a.asset_id}</span>
                    <span className="font-medium">{a.balance}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-tm-muted bg-tm-bg rounded-lg px-3 py-2">
                Connect API key to view balances
              </p>
            )}
          </div>

          {/* Recent Orders */}
          <div>
            <p className="text-xs text-tm-muted uppercase tracking-wider mb-2">Recent Orders</p>
            {recentOrders.length > 0 ? (
              <div className="space-y-1">
                {recentOrders.map((o, i) => (
                  <div key={i} className="flex items-center justify-between text-xs bg-tm-bg rounded-lg px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className={`font-semibold ${o.side === "buy" ? "text-tm-green" : "text-tm-red"}`}>
                        {o.side?.toUpperCase()}
                      </span>
                      <span className="text-tm-muted">{o.base_asset}/{o.quote_asset}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span>{o.qty} @ ${o.price}</span>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                        o.status === "complete"
                          ? "bg-tm-green/10 text-tm-green"
                          : "bg-tm-yellow/10 text-tm-yellow"
                      }`}>
                        {o.status}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-tm-muted bg-tm-bg rounded-lg px-3 py-2">
                No orders yet. Use mispricing signals to trade.
              </p>
            )}
          </div>

          {/* API Info */}
          <div className="mt-4 pt-3 border-t border-tm-border">
            <p className="text-[10px] text-tm-muted">
              Trading via True Markets Gateway API
            </p>
            <p className="text-[10px] text-tm-muted">
              {process.env.NEXT_PUBLIC_API_URL?.includes("mock") || !process.env.NEXT_PUBLIC_API_URL
                ? "Mock server (demo mode)"
                : "Production"}
            </p>
          </div>
        </>
      )}
    </div>
  );
}
