"use client";

import { useState } from "react";
import { placeOrder } from "@/lib/api";

interface TradeCase {
  side: string;
  reasons: string[];
  vote_count: number;
}

interface TradeData {
  mode: string;
  symbol: string;
  base_asset: string;
  primary_side: string;
  buy_case: TradeCase;
  sell_case: TradeCase;
  total_signals: number;
  quote?: { price: string; qty: string; total: string } | null;
  // Legacy consensus fields
  side?: string;
  reasons?: string[];
  vote_count?: number;
}

interface Props {
  trade: TradeData;
  onOrderPlaced: () => void;
}

export default function RecommendedTrade({ trade, onOrderPlaced }: Props) {
  const price = trade.quote ? parseFloat(trade.quote.price) : 0;
  const primary = trade.primary_side || trade.side || "buy";
  const buyCase = trade.buy_case || { side: "buy", reasons: trade.reasons || [], vote_count: trade.vote_count || 0 };
  const sellCase = trade.sell_case || { side: "sell", reasons: [], vote_count: 0 };
  const total = trade.total_signals || (buyCase.vote_count + sellCase.vote_count);

  return (
    <div>
      <div className="grid grid-cols-2 gap-3">
        <CaseCard
          caseData={buyCase}
          symbol={trade.symbol}
          baseAsset={trade.base_asset}
          price={price}
          isPrimary={primary === "buy"}
          totalSignals={total}
          onOrderPlaced={onOrderPlaced}
        />
        <CaseCard
          caseData={sellCase}
          symbol={trade.symbol}
          baseAsset={trade.base_asset}
          price={price}
          isPrimary={primary === "sell"}
          totalSignals={total}
          onOrderPlaced={onOrderPlaced}
        />
      </div>
    </div>
  );
}

function CaseCard({ caseData, symbol, baseAsset, price, isPrimary, totalSignals, onOrderPlaced }: {
  caseData: TradeCase; symbol: string; baseAsset: string; price: number;
  isPrimary: boolean; totalSignals: number; onOrderPlaced: () => void;
}) {
  const [qty, setQty] = useState("1");
  const [placing, setPlacing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const isBuy = caseData.side === "buy";
  const total = price * parseFloat(qty || "0");
  const allPointOther = caseData.reasons.length === 1 && caseData.reasons[0].startsWith("All signals");

  const handleOrder = async () => {
    setPlacing(true);
    try {
      const res = await placeOrder(baseAsset, caseData.side, qty);
      setResult(res.order_id || "Placed"); onOrderPlaced();
    } catch {} finally { setPlacing(false); }
  };

  return (
    <div className={`border rounded-xl p-4 transition-all ${
      isPrimary
        ? isBuy ? "border-tm-green/40 bg-tm-green/5" : "border-tm-red/40 bg-tm-red/5"
        : "border-tm-border bg-tm-card opacity-70"
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold ${isBuy ? "text-tm-green" : "text-tm-red"}`}>
            {caseData.side.toUpperCase()} {symbol}
          </span>
          {isPrimary && (
            <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
              isBuy ? "bg-tm-green/15 text-tm-green" : "bg-tm-red/15 text-tm-red"
            }`}>recommended</span>
          )}
        </div>
        <span className="text-[10px] text-tm-muted">
          {caseData.vote_count}/{totalSignals}
        </span>
      </div>

      {/* Reasons */}
      <div className="space-y-1 mb-3 min-h-[40px]">
        {caseData.reasons.map((r, i) => (
          <p key={i} className={`text-xs ${allPointOther ? "text-tm-muted italic" : "text-tm-muted"}`}>
            {allPointOther ? "" : "\u25CF "}{r}
          </p>
        ))}
      </div>

      {/* Trade action */}
      {!result ? (
        <div className="flex items-center gap-2">
          <input type="number" min="0.01" step="0.1" value={qty}
            onChange={(e) => { setQty(e.target.value); setResult(null); }}
            className="w-14 bg-tm-bg border border-tm-border rounded px-2 py-1 text-xs text-right focus:border-tm-accent focus:outline-none" />
          <span className="text-[10px] text-tm-muted">${total.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
          <button onClick={handleOrder} disabled={placing}
            className={`ml-auto px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              isBuy ? "bg-tm-green/15 text-tm-green border border-tm-green/25 hover:bg-tm-green/25"
                    : "bg-tm-red/15 text-tm-red border border-tm-red/25 hover:bg-tm-red/25"
            } disabled:opacity-50`}>
            {placing ? "..." : caseData.side.toUpperCase()}
          </button>
        </div>
      ) : (
        <p className="text-xs text-tm-green">Placed (ID: {result})</p>
      )}
    </div>
  );
}
