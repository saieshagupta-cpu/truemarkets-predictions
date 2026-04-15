"use client";

import { useEffect, useState } from "react";
import { getBalances, getOrders } from "@/lib/api";
import type { BalancesResponse, OrdersResponse } from "@/lib/api";

interface PortfolioBoxProps {
  refreshKey: number;
}

export default function PortfolioBox({ refreshKey }: PortfolioBoxProps) {
  const [balances, setBalances] = useState<BalancesResponse | null>(null);
  const [orders, setOrders] = useState<OrdersResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.allSettled([getBalances(), getOrders()])
      .then(([b, o]) => {
        if (b.status === "fulfilled") setBalances(b.value);
        if (o.status === "fulfilled") setOrders(o.value);
      })
      .finally(() => setLoading(false));
  }, [refreshKey]);

  const recentOrders = (orders?.data || []).slice(0, 5);
  const hasBalances = balances && balances.balances && balances.balances.length > 0;

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      <h3 className="text-sm font-semibold text-tm-muted uppercase tracking-wider mb-3">
        Portfolio
      </h3>

      {loading ? (
        <div className="flex items-center justify-center py-4">
          <div className="w-4 h-4 border-2 border-tm-accent border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          {/* Balances */}
          {hasBalances ? (
            <div className="space-y-1.5 mb-3">
              {balances!.balances.map((b) => (
                <div key={b.asset_id} className="flex items-center justify-between">
                  <span className="text-xs text-tm-text font-medium">{b.asset_name || b.asset_id}</span>
                  <span className="text-xs text-tm-muted font-mono">{parseFloat(b.balance).toFixed(4)}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-[10px] text-tm-muted mb-3">No balances</p>
          )}

          {/* Recent Orders */}
          <div className="border-t border-tm-border pt-2">
            <p className="text-[10px] text-tm-muted uppercase tracking-wider mb-1.5">Recent Orders</p>
            {recentOrders.length > 0 ? (
              <div className="space-y-1">
                {recentOrders.map((o) => (
                  <div key={o.order_id} className="flex items-center justify-between text-[10px]">
                    <div className="flex items-center gap-1.5">
                      <span className={`font-bold ${o.side === "buy" ? "text-tm-green" : "text-tm-red"}`}>
                        {o.side.toUpperCase()}
                      </span>
                      <span className="text-tm-muted">{o.base_asset}/{o.quote_asset}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-tm-muted">{parseFloat(o.qty).toFixed(4)}</span>
                      <StatusBadge status={o.status} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[10px] text-tm-muted italic">No recent orders</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    complete: "bg-tm-green/20 text-tm-green",
    pending: "bg-tm-yellow/20 text-tm-yellow",
    active: "bg-tm-blue/20 text-tm-blue",
    failed: "bg-tm-red/20 text-tm-red",
    canceled: "bg-tm-muted/20 text-tm-muted",
    cancelled: "bg-tm-muted/20 text-tm-muted",
  };
  return (
    <span className={`px-1.5 py-0.5 rounded text-[8px] font-medium ${colors[status] || "bg-tm-border text-tm-muted"}`}>
      {status}
    </span>
  );
}
