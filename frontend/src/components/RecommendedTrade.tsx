"use client";

import { useState } from "react";
import { placeOrder } from "@/lib/api";

interface TradeCase {
  side: string;
  reasons: string[];
  vote_count: number;
}

interface TradeData {
  mode: "split" | "consensus";
  // Consensus fields
  side?: string;
  reasons?: string[];
  vote_count?: number;
  total_signals?: number;
  // Split fields
  buy_case?: TradeCase;
  sell_case?: TradeCase;
  primary_side?: string;
  // Shared
  symbol: string;
  base_asset: string;
  quote?: { price: string; qty: string; total: string } | null;
}

interface Props {
  trade: TradeData;
  onOrderPlaced: () => void;
}

export default function RecommendedTrade({ trade, onOrderPlaced }: Props) {
  if (trade.mode === "split") {
    return <SplitView trade={trade} onOrderPlaced={onOrderPlaced} />;
  }
  return <ConsensusView trade={trade} onOrderPlaced={onOrderPlaced} />;
}

function ConsensusView({ trade, onOrderPlaced }: Props) {
  const [qty, setQty] = useState("1");
  const [placing, setPlacing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const isBuy = trade.side === "buy";
  const price = trade.quote ? parseFloat(trade.quote.price) : 0;
  const total = price * parseFloat(qty || "0");

  const handleOrder = async () => {
    setPlacing(true);
    try {
      const res = await placeOrder(trade.base_asset, trade.side!, qty);
      setResult(res.order_id || "Placed"); onOrderPlaced();
    } catch {} finally { setPlacing(false); }
  };

  return (
    <div className="bg-tm-card border border-tm-border rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <p className="text-xs text-tm-muted uppercase tracking-wider">Recommended</p>
        <span className={`text-sm font-bold ${isBuy ? "text-tm-green" : "text-tm-red"}`}>
          {trade.side?.toUpperCase()} {trade.symbol}
        </span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-tm-border text-tm-muted">
          {trade.vote_count}/{trade.total_signals} signals
        </span>
      </div>
      <div className="space-y-1 mb-3">
        {(trade.reasons || []).map((r, i) => (
          <p key={i} className={`text-xs ${i === 0 ? "text-tm-text" : "text-tm-muted"}`}>{i === 0 ? "\u25CF " : "\u25CB "}{r}</p>
        ))}
      </div>
      {trade.quote && !result && (
        <div className="flex items-center gap-3">
          <span className="text-xs text-tm-muted">${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}/unit</span>
          <span className="text-tm-border">|</span>
          <input type="number" min="0.01" step="0.1" value={qty}
            onChange={(e) => { setQty(e.target.value); setResult(null); }}
            className="w-16 bg-tm-bg border border-tm-border rounded px-2 py-1 text-xs text-right focus:border-tm-accent focus:outline-none" />
          <span className="text-xs text-tm-muted">{trade.symbol}</span>
          <span className="text-tm-border">|</span>
          <span className="text-xs font-medium">${total.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          <button onClick={handleOrder} disabled={placing}
            className={`ml-auto px-4 py-1.5 rounded-lg text-xs font-semibold transition-all shrink-0 ${
              isBuy ? "bg-tm-green/15 text-tm-green border border-tm-green/25 hover:bg-tm-green/25"
                    : "bg-tm-red/15 text-tm-red border border-tm-red/25 hover:bg-tm-red/25"
            } disabled:opacity-50`}>
            {placing ? "..." : trade.side?.toUpperCase()}
          </button>
        </div>
      )}
      {result && (
        <div className="flex items-center gap-2 text-xs">
          <span className="text-tm-green font-medium">Placed</span>
          <span className="text-tm-muted">ID: {result}</span>
          <button onClick={() => setResult(null)} className="text-tm-accent hover:underline ml-auto">Again</button>
        </div>
      )}
    </div>
  );
}

function SplitView({ trade, onOrderPlaced }: Props) {
  const price = trade.quote ? parseFloat(trade.quote.price) : 0;

  return (
    <div className="space-y-3">
      <div className="bg-tm-card border border-tm-border rounded-xl p-3">
        <p className="text-xs text-tm-muted text-center">Signals are split — both cases shown</p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <CaseCard case_data={trade.buy_case!} symbol={trade.symbol} baseAsset={trade.base_asset} price={price} onOrderPlaced={onOrderPlaced} />
        <CaseCard case_data={trade.sell_case!} symbol={trade.symbol} baseAsset={trade.base_asset} price={price} onOrderPlaced={onOrderPlaced} />
      </div>
    </div>
  );
}

function CaseCard({ case_data, symbol, baseAsset, price, onOrderPlaced }: {
  case_data: TradeCase; symbol: string; baseAsset: string; price: number; onOrderPlaced: () => void;
}) {
  const [qty, setQty] = useState("1");
  const [placing, setPlacing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const isBuy = case_data.side === "buy";
  const total = price * parseFloat(qty || "0");

  const handleOrder = async () => {
    setPlacing(true);
    try {
      const res = await placeOrder(baseAsset, case_data.side, qty);
      setResult(res.order_id || "Placed"); onOrderPlaced();
    } catch {} finally { setPlacing(false); }
  };

  return (
    <div className={`border rounded-xl p-4 ${isBuy ? "border-tm-green/30 bg-tm-green/5" : "border-tm-red/30 bg-tm-red/5"}`}>
      <div className="flex items-center justify-between mb-3">
        <span className={`text-sm font-bold ${isBuy ? "text-tm-green" : "text-tm-red"}`}>
          {case_data.side.toUpperCase()} {symbol}
        </span>
        <span className="text-[10px] text-tm-muted">{case_data.vote_count} signal{case_data.vote_count !== 1 ? "s" : ""}</span>
      </div>
      <div className="space-y-1 mb-3">
        {case_data.reasons.map((r, i) => (
          <p key={i} className="text-xs text-tm-muted">{"\u25CF "}{r}</p>
        ))}
      </div>
      {!result ? (
        <div className="flex items-center gap-2">
          <input type="number" min="0.01" step="0.1" value={qty}
            onChange={(e) => { setQty(e.target.value); setResult(null); }}
            className="w-14 bg-tm-bg border border-tm-border rounded px-2 py-1 text-xs text-right focus:border-tm-accent focus:outline-none" />
          <span className="text-[10px] text-tm-muted">{symbol}</span>
          <span className="text-[10px] text-tm-muted">${total.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          <button onClick={handleOrder} disabled={placing}
            className={`ml-auto px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              isBuy ? "bg-tm-green/15 text-tm-green border border-tm-green/25 hover:bg-tm-green/25"
                    : "bg-tm-red/15 text-tm-red border border-tm-red/25 hover:bg-tm-red/25"
            } disabled:opacity-50`}>
            {placing ? "..." : case_data.side.toUpperCase()}
          </button>
        </div>
      ) : (
        <p className="text-xs text-tm-green">Placed (ID: {result})</p>
      )}
    </div>
  );
}
