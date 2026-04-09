import numpy as np
import pandas as pd
from app.models.lstm_model import LSTMPredictor
from app.models.xgboost_model import XGBoostPredictor
from app.models.sentiment import SentimentPredictor
from app.config import LSTM_WEIGHT, XGBOOST_WEIGHT, SENTIMENT_WEIGHT, SEQUENCE_LENGTH


class EnsemblePredictionEngine:
    def __init__(self, num_thresholds: int = 6):
        self.lstm = LSTMPredictor(num_thresholds=num_thresholds)
        self.xgboost = XGBoostPredictor()
        self.sentiment_model = SentimentPredictor()

    def predict(
        self,
        price_df: pd.DataFrame,
        sentiment_data: dict,
        fear_greed_data: dict,
        onchain_data: dict,
        thresholds: list[float],
    ) -> dict:
        current_price = float(price_df["price"].iloc[-1])
        volatility = float(price_df["volatility_20d"].iloc[-1]) if "volatility_20d" in price_df.columns else 0.02

        # LSTM: prepare normalized sequences, but pass actual price for heuristic
        feature_cols = ["price", "volume", "rsi", "macd", "bb_width", "volatility_20d"]
        available_cols = [c for c in feature_cols if c in price_df.columns]
        seq_data = price_df[available_cols].values[-SEQUENCE_LENGTH:]
        means = seq_data.mean(axis=0)
        stds = seq_data.std(axis=0)
        stds[stds == 0] = 1
        seq_normalized = (seq_data - means) / stds

        lstm_probs = self.lstm.predict(seq_normalized, thresholds, current_price, volatility)

        # XGBoost: pass actual feature values
        xgb_features = {
            "price": current_price,
            "rsi": float(price_df["rsi"].iloc[-1]) if "rsi" in price_df.columns else 50,
            "macd": float(price_df["macd"].iloc[-1]) if "macd" in price_df.columns else 0,
            "bb_width": float(price_df["bb_width"].iloc[-1]) if "bb_width" in price_df.columns else 0,
            "volatility": volatility,
            "volume_change": float(price_df["volume_change"].iloc[-1]) if "volume_change" in price_df.columns else 0,
            "fear_greed": fear_greed_data.get("current", {}).get("value", 50),
            "sentiment_score": sentiment_data.get("sentiment_score", 0),
            "hash_rate": onchain_data.get("hash_rate", 0),
        }
        xgb_probs = self.xgboost.predict(xgb_features, thresholds)

        # Sentiment: pass actual price + volatility
        sent_probs = self.sentiment_model.predict(
            sentiment_data, fear_greed_data, thresholds, current_price, volatility
        )

        # Ensemble: weighted average
        ensemble_probs = {}
        model_agreement = []
        for threshold in thresholds:
            key = str(threshold)
            lstm_p = lstm_probs.get(key, 0.5)
            xgb_p = xgb_probs.get(key, 0.5)
            sent_p = sent_probs.get(key, 0.5)

            combined = lstm_p * LSTM_WEIGHT + xgb_p * XGBOOST_WEIGHT + sent_p * SENTIMENT_WEIGHT
            ensemble_probs[key] = round(float(np.clip(combined, 0.01, 0.99)), 4)

            # Confidence: decisiveness + agreement
            decisiveness = abs(combined - 0.5) * 2
            pairwise_diffs = [abs(lstm_p - xgb_p), abs(lstm_p - sent_p), abs(xgb_p - sent_p)]
            agreement = 1 - np.mean(pairwise_diffs)
            model_agreement.append(0.5 * decisiveness + 0.5 * agreement)

        confidence = round(float(np.clip(np.mean(model_agreement), 0.1, 0.99)), 4)
        sentiment_signal = self.sentiment_model.get_signal_breakdown(sentiment_data, fear_greed_data)

        return {
            "coin": "bitcoin",
            "current_price": current_price,
            "thresholds": {
                key: {
                    "probability": ensemble_probs[key],
                    "direction": "up" if float(key) > current_price else "down",
                    "distance_pct": round((float(key) - current_price) / current_price * 100, 2),
                }
                for key in ensemble_probs
            },
            "confidence": confidence,
            "model_signals": {
                "lstm": lstm_probs,
                "xgboost": xgb_probs,
                "sentiment": sent_probs,
            },
            "weights": {
                "lstm": LSTM_WEIGHT,
                "xgboost": XGBOOST_WEIGHT,
                "sentiment": SENTIMENT_WEIGHT,
            },
            "sentiment_signal": sentiment_signal,
            "indicators": {
                "rsi": xgb_features["rsi"],
                "macd": xgb_features["macd"],
                "volatility": xgb_features["volatility"],
                "fear_greed": xgb_features["fear_greed"],
            },
        }
