"use client";

import type { OrderFlowData } from "@/lib/api";

interface OrderFlowPanelProps {
  orderFlow: OrderFlowData;
}

export default function OrderFlowPanel({ orderFlow }: OrderFlowPanelProps) {
  return (
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
  );
}
