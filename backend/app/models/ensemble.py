import numpy as np
import pandas as pd
import os
import pickle
from app.models.lstm_model import LSTMPredictor
from app.models.xgboost_model import XGBoostPredictor, GBM_FEATURES
from app.models.sentiment import SentimentPredictor
from app.config import LSTM_WEIGHT, XGBOOST_WEIGHT, SENTIMENT_WEIGHT, SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class MetaLearner:
    """
    Stacking meta-learner: trained logistic regression per threshold.
    Input: [lstm_prob, gbm_prob, sent_prob, volatility, rsi]
    Learns WHEN to trust each sub-model.
    """
    def __init__(self):
        self.models = {}
        self.trained = False
        path = os.path.join(MODEL_WEIGHTS_DIR, "meta_learner.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.models = pickle.load(f)
                self.trained = bool(self.models)
            except Exception:
                pass

    def predict(self, lstm_p: float, gbm_p: float, sent_p: float,
                volatility: float, rsi: float, threshold_key: str) -> float:
        if not self.trained or threshold_key not in self.models:
            return lstm_p * LSTM_WEIGHT + gbm_p * XGBOOST_WEIGHT + sent_p * SENTIMENT_WEIGHT

        model = self.models[threshold_key]
        x = np.array([[lstm_p, gbm_p, sent_p, volatility, rsi]])
        try:
            prob = model.predict_proba(x)[0, 1]
        except Exception:
            prob = lstm_p * LSTM_WEIGHT + gbm_p * XGBOOST_WEIGHT + sent_p * SENTIMENT_WEIGHT
        return float(np.clip(prob, 0.01, 0.99))


class EnsemblePredictionEngine:
    def __init__(self, num_thresholds: int = 6):
        self.lstm = LSTMPredictor(num_thresholds=num_thresholds)
        self.xgboost = XGBoostPredictor()
        self.sentiment_model = SentimentPredictor()
        self.meta_learner = MetaLearner()

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

        # ── LSTM input: normalized sequences ──
        feature_cols = ["price", "volume", "rsi", "macd", "bb_width", "volatility_20d"]
        available_cols = [c for c in feature_cols if c in price_df.columns]
        seq_data = price_df[available_cols].values[-SEQUENCE_LENGTH:]
        # Pad if not enough data
        if len(seq_data) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq_data), len(available_cols)))
            seq_data = np.vstack([pad, seq_data])
        means = seq_data.mean(axis=0)
        stds = seq_data.std(axis=0)
        stds[stds == 0] = 1
        seq_normalized = (seq_data - means) / stds

        lstm_probs = self.lstm.predict(seq_normalized, thresholds, current_price, volatility)

        # ── XGBoost input: expanded features ──
        def _safe_get(col, default=0):
            return float(price_df[col].iloc[-1]) if col in price_df.columns else default

        xgb_features = {
            "price": current_price,
            "rsi": _safe_get("rsi", 50),
            "macd": _safe_get("macd", 0),
            "bb_width": _safe_get("bb_width", 0),
            "volatility": volatility,
            "volume_change": _safe_get("volume_change", 0),
            "return_1d": _safe_get("return_1d", 0),
            "return_3d": _safe_get("return_3d", 0),
            "return_7d": _safe_get("return_7d", 0),
            "return_14d": _safe_get("return_14d", 0),
            "return_30d": _safe_get("return_30d", 0),
            "relative_volume": _safe_get("relative_volume", 1.0),
            "volatility_ratio": _safe_get("volatility_ratio", 1.0),
            "rsi_momentum": _safe_get("rsi_momentum", 0),
            "price_position": _safe_get("price_position", 0.5),
            "bollinger_position": _safe_get("bollinger_position", 0.5),
            "fear_greed": fear_greed_data.get("current", {}).get("value", 50),
            "sentiment_score": sentiment_data.get("sentiment_score", 0),
            "hash_rate": onchain_data.get("hash_rate", 0),
        }
        xgb_probs = self.xgboost.predict(xgb_features, thresholds)

        # ── Sentiment ──
        sent_probs = self.sentiment_model.predict(
            sentiment_data, fear_greed_data, thresholds, current_price, volatility
        )

        # ── Ensemble: meta-learner or fixed weights ──
        ensemble_probs = {}
        model_agreement = []
        rsi_val = xgb_features["rsi"]

        for threshold in thresholds:
            key = str(threshold)
            lstm_p = lstm_probs.get(key, 0.5)
            xgb_p = xgb_probs.get(key, 0.5)
            sent_p = sent_probs.get(key, 0.5)

            if self.meta_learner.trained:
                combined = self.meta_learner.predict(lstm_p, xgb_p, sent_p, volatility, rsi_val, key)
            else:
                combined = lstm_p * LSTM_WEIGHT + xgb_p * XGBOOST_WEIGHT + sent_p * SENTIMENT_WEIGHT

            ensemble_probs[key] = round(float(np.clip(combined, 0.01, 0.99)), 4)

            # Confidence
            decisiveness = abs(combined - 0.5) * 2
            pairwise_diffs = [abs(lstm_p - xgb_p), abs(lstm_p - sent_p), abs(xgb_p - sent_p)]
            agreement = 1 - np.mean(pairwise_diffs)
            model_agreement.append(0.5 * decisiveness + 0.5 * agreement)

        # ── Directional signal for near-money thresholds ──
        up_probs = []
        down_probs = []
        near_money = []

        for threshold in thresholds:
            key = str(threshold)
            distance_pct = abs((threshold - current_price) / current_price * 100)

            if threshold > current_price:
                up_probs.append(ensemble_probs[key])
            else:
                down_probs.append(ensemble_probs[key])

            if distance_pct <= 7.0:  # within 7% is "near money"
                near_money.append(key)

        avg_up = np.mean(up_probs) if up_probs else 0.5
        avg_down = np.mean(down_probs) if down_probs else 0.5
        directional_ratio = avg_up / max(avg_down, 0.01)

        # Override near-money with directional signal
        for key in near_money:
            threshold = float(key)
            if threshold > current_price:
                adjusted = directional_ratio / (1 + directional_ratio)
                ensemble_probs[key] = round(float(np.clip(adjusted, 0.2, 0.8)), 4)
            else:
                adjusted = 1 / (1 + directional_ratio)
                ensemble_probs[key] = round(float(np.clip(adjusted, 0.2, 0.8)), 4)

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
                "meta_learner": self.meta_learner.trained,
            },
            "sentiment_signal": sentiment_signal,
            "indicators": {
                "rsi": xgb_features["rsi"],
                "macd": xgb_features["macd"],
                "volatility": xgb_features["volatility"],
                "fear_greed": xgb_features["fear_greed"],
            },
            "directional_signal": {
                "avg_up_prob": round(float(avg_up), 4),
                "avg_down_prob": round(float(avg_down), 4),
                "ratio": round(float(directional_ratio), 2),
                "bias": "bullish" if directional_ratio > 1.2 else "bearish" if directional_ratio < 0.8 else "neutral",
                "near_money_count": len(near_money),
            },
        }
