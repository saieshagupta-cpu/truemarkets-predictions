from pydantic import BaseModel


class ThresholdPrediction(BaseModel):
    probability: float
    direction: str
    distance_pct: float


class ModelSignals(BaseModel):
    lstm: dict[str, float]
    xgboost: dict[str, float]
    sentiment: dict[str, float]


class SentimentSignal(BaseModel):
    overall_signal: str
    social_sentiment: str
    fear_greed: str
    fear_greed_value: int | float
    sentiment_score: float
    bullish_ratio: float


class Indicators(BaseModel):
    rsi: float
    macd: float
    volatility: float
    fear_greed: float


class PredictionResponse(BaseModel):
    coin: str
    current_price: float
    thresholds: dict[str, ThresholdPrediction]
    confidence: float
    model_signals: ModelSignals
    weights: dict[str, float]
    sentiment_signal: SentimentSignal
    indicators: Indicators


class MarketDataResponse(BaseModel):
    price: float
    change_24h: float
    market_cap: float
    volume_24h: float
    fear_greed: dict
    onchain: dict
    sentiment: dict


class ComparisonResponse(BaseModel):
    coin: str
    our_predictions: dict[str, float]
    polymarket_markets: list[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
