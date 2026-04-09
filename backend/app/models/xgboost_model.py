import numpy as np
import os
import json
from app.config import MODEL_WEIGHTS_DIR


class XGBoostPredictor:
    def __init__(self):
        self.models = {}
        self._load_models()

    def _load_models(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "xgboost_models.json")
        if os.path.exists(path):
            with open(path) as f:
                self.models = json.load(f)
            self.trained = True
        else:
            self.trained = False

    def predict(self, features: dict, thresholds: list[float]) -> dict[str, float]:
        if not self.trained:
            return self._heuristic_predict(features, thresholds)

        results = {}
        for threshold in thresholds:
            key = str(threshold)
            if key in self.models:
                weights = self.models[key]
                prob = self._score(features, weights)
            else:
                prob = 0.5
            results[key] = float(np.clip(prob, 0.01, 0.99))
        return results

    def _heuristic_predict(self, features: dict, thresholds: list[float]) -> dict[str, float]:
        """
        Feature-driven probability estimation.
        Uses current price + indicators to adjust the base probability.
        """
        current_price = features.get("price", 0)
        rsi = features.get("rsi", 50)
        fear_greed = features.get("fear_greed", 50)
        sentiment = features.get("sentiment_score", 0)
        volatility = features.get("volatility", 0.02)

        if current_price <= 0:
            return {str(t): 0.5 for t in thresholds}

        # Directional bias from indicators: -1 (bearish) to +1 (bullish)
        bias = 0.0
        bias += np.clip((rsi - 50) / 50, -1, 1) * 0.3       # RSI momentum
        bias += np.clip(sentiment * 2, -1, 1) * 0.2           # Social sentiment
        bias += np.clip((fear_greed - 50) / 50, -1, 1) * 0.3  # Fear/Greed
        # bias range: roughly -0.8 to +0.8

        daily_vol = max(volatility, 0.01)
        horizon_vol = daily_vol * np.sqrt(30)

        results = {}
        for threshold in thresholds:
            pct_move = (threshold - current_price) / current_price

            if threshold > current_price:
                # Upside target — bullish bias increases probability
                z = (pct_move - bias * 0.05) / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))
            else:
                # Downside target — bearish bias increases probability
                z = (abs(pct_move) + bias * 0.05) / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))

            results[str(threshold)] = float(np.clip(prob, 0.01, 0.99))
        return results

    def _score(self, features: dict, weights: dict) -> float:
        score = weights.get("intercept", 0)
        for key, weight in weights.items():
            if key != "intercept" and key in features:
                score += features[key] * weight
        return 1 / (1 + np.exp(-score))


def _norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
