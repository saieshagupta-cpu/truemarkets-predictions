const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface MispricingSignal {
  threshold: string;
  direction: "up" | "down";
  distance_pct: number;
  our_prob: number;
  poly_prob: number | null;
  diff: number | null;
  signal: "OVERPRICED" | "UNDERPRICED" | "FAIR" | "NO_MARKET";
  severity: "strong" | "moderate" | "fair" | "unknown";
  action: "buy" | "sell" | "hold" | "monitor";
  description: string;
  poly_question: string | null;
  poly_volume: number | null;
  model_signals: { lstm: number; xgboost: number; sentiment: number };
}

export interface SentimentSignal {
  overall_signal: string;
  social_sentiment: string;
  fear_greed: string;
  fear_greed_value: number;
  sentiment_score: number;
  bullish_ratio: number;
}

export interface RecommendedTrade {
  side: "buy" | "sell";
  symbol: string;
  base_asset: string;
  confidence: number;
  reasons: string[];
  based_on_mispricing: boolean;
  strongest_signal: { threshold: string; diff: number; signal: string } | null;
  quote: { price: string; qty: string; total: string } | null;
}

export interface OrderFlow {
  combined_signal: number;
  pressure: "strong_buy" | "buy" | "neutral" | "sell" | "strong_sell";
  polymarket_flow: {
    signal: number;
    details: {
      up_volume_24h?: number;
      down_volume_24h?: number;
      volume_ratio?: number;
      volume_acceleration?: number;
      avg_spread?: number;
      up_momentum?: number;
      down_momentum?: number;
      liquidity_ratio?: number;
    };
  };
  truemarkets_flow: {
    signal: number;
    order_count: number;
    buy_count: number;
    sell_count: number;
  };
}

export interface MispricingData {
  coin: string;
  symbol: string;
  current_price: number;
  confidence: number;
  sentiment_signal: SentimentSignal;
  indicators: { rsi: number; macd: number; volatility: number; fear_greed: number };
  signals: MispricingSignal[];
  polymarket_count: number;
  order_flow: OrderFlow | null;
  recommended_trade: RecommendedTrade | null;
}

export interface MarketData {
  price: number;
  change_24h: number;
  market_cap: number;
  volume_24h: number;
  fear_greed: { current?: { value: number; classification: string } };
  onchain: Record<string, number>;
  sentiment: { sentiment_score?: number; classification?: string };
}

export interface QuoteResponse {
  qty: string;
  price: string;
}

export interface OrderResponse {
  order_id: string;
  status: string;
  quote?: { base_asset: string; quote_asset: string; qty: string; qty_out: string; fee: string; fee_asset: string };
}

export interface BalancesResponse {
  balances: Array<{ asset_name: string; asset_id: string; balance: string }>;
}

export interface OrdersResponse {
  data: Array<{
    order_id: string;
    base_asset: string;
    quote_asset: string;
    side: string;
    qty: string;
    executed_qty: string;
    price: string;
    status: string;
    created_at: string;
  }>;
}

export interface CoinInfo {
  symbol: string;
  base_asset: string;
}

async function fetchAPI<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

async function postAPI<T>(endpoint: string, body: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const getCoins = () => fetchAPI<Record<string, CoinInfo>>("/coins");
export const getMispricing = (coin: string) => fetchAPI<MispricingData>(`/mispricing/${coin}`);
export const getMarketData = (coin: string) => fetchAPI<MarketData>(`/market-data/${coin}`);
export const getBalances = () => fetchAPI<BalancesResponse>("/trade/balances");
export const getOrders = () => fetchAPI<OrdersResponse>("/trade/orders");

export const getQuote = (base_asset: string, side: string, qty: string) =>
  postAPI<QuoteResponse>("/trade/quote", { base_asset, quote_asset: "USD", side, qty, qty_unit: "base" });

export const placeOrder = (base_asset: string, side: string, qty: string, order_type = "market") =>
  postAPI<OrderResponse>("/trade/order", { base_asset, quote_asset: "USD", side, qty, order_type });
