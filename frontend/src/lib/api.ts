const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// ─── NEW: Prediction types ────────────────────────────

export interface Signal {
  name: string;
  direction: "bullish" | "bearish" | "neutral";
  strength: number;
  reason: string;
  weight: number;
  raw_data: Record<string, unknown>;
}

export interface PolymarketThreshold {
  threshold: number;
  direction: "up" | "down";
  yes_price: number;
  yes_pct: number;
  volume: number;
  volume_24h: number;
  question: string;
  liquidity: number;
}

export interface OrderFlowData {
  buy_volume: number;
  sell_volume: number;
  buy_sell_ratio: number;
  buy_count: number;
  sell_count: number;
  bid_depth: number;
  ask_depth: number;
  imbalance: number;
  signal: number;
  pressure: string;
  source: string;
}

export interface TechnicalIndicators {
  rsi: number;
  rsi_label: string;
  macd_line: number;
  macd_signal: number;
  macd_histogram: number;
  bollinger_position: number;
  bollinger_label: string;
}

export interface PredictionData {
  current_price: number;
  change_24h: number;
  recommended_side: "buy" | "sell";
  confidence: number;
  weighted_strength: number;
  buy_signals: Signal[];
  sell_signals: Signal[];
  buy_count: number;
  sell_count: number;
  total_signals: number;
  polymarket_thresholds: PolymarketThreshold[];
  order_flow: OrderFlowData;
  technical_indicators: TechnicalIndicators;
  sentiment_summary: string;
  fear_greed: { current?: { value: number; classification: string } };
  backtest_results: Record<string, unknown>;
  weights: Record<string, number>;
  updated_at: number;
}

// ─── KEPT: Legacy types for MarketView compatibility ──

export interface MispricingSignal {
  threshold: string;
  direction: "up" | "down";
  distance_pct: number;
  our_prob: number;
  poly_prob: number | null;
  diff: number | null;
  signal: string;
  severity: string;
  action: string;
  description: string;
  poly_question: string | null;
  poly_volume: number | null;
  model_signals: Record<string, number>;
}

export interface SentimentSignal {
  overall_signal: string;
  social_sentiment?: string;
  fear_greed: string;
  fear_greed_value: number;
  sentiment_score: number;
  bullish_ratio: number;
}

export interface OrderFlow {
  combined_signal?: number;
  signal?: number;
  pressure: string;
  polymarket_flow?: Record<string, unknown>;
  truemarkets_flow?: Record<string, unknown>;
  buy_volume?: number;
  sell_volume?: number;
  buy_sell_ratio?: number;
  imbalance?: number;
  source?: string;
}

export interface RecommendedTrade {
  mode: string;
  symbol: string;
  base_asset: string;
  primary_side: string;
  buy_case: { side: string; reasons: string[]; vote_count: number };
  sell_case: { side: string; reasons: string[]; vote_count: number };
  total_signals: number;
  confidence: number;
  quote: { price: string; qty: string; total: string } | null;
}

export interface MispricingData {
  coin: string;
  symbol: string;
  current_price: number;
  change_24h_pct?: number;
  confidence: number;
  sentiment_signal: SentimentSignal;
  indicators: Record<string, number>;
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

// ─── API helpers ──────────────────────────────────────

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

// ─── Exports ──────────────────────────────────────────

export const getPrediction = () => fetchAPI<PredictionData>("/prediction/bitcoin");
export const getMispricing = (coin: string) => fetchAPI<MispricingData>(`/mispricing/${coin}`);
export const getCoins = () => fetchAPI<Record<string, CoinInfo>>("/coins");
export const getMarketData = (coin: string) => fetchAPI<MarketData>(`/market-data/${coin}`);
export const getBalances = () => fetchAPI<BalancesResponse>("/trade/balances");
export const getOrders = () => fetchAPI<OrdersResponse>("/trade/orders");

export const getQuote = (base_asset: string, side: string, qty: string) =>
  postAPI<QuoteResponse>("/trade/quote", { base_asset, quote_asset: "USD", side, qty, qty_unit: "base" });

export const placeOrder = (base_asset: string, side: string, qty: string, order_type = "market") =>
  postAPI<OrderResponse>("/trade/order", { base_asset, quote_asset: "USD", side, qty, order_type });
