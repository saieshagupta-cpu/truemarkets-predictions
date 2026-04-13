"""
Next-day BTC direction prediction model.
Separate from the 30-day threshold model — this predicts tomorrow's direction
for the recommendation engine.

Trained on 3 years of daily BTC data via RandomForest.
Walk-forward backtest: 54.4% accuracy, AUC 0.5257.
"""

import os
import pickle
import numpy as np
from app.config import MODEL_WEIGHTS_DIR

FEATURES = [
    "rsi", "macd_hist", "bb_position", "volatility_20d", "volatility_ratio",
    "volume_change", "relative_volume", "vol_trend",
    "return_1d", "return_2d", "return_3d", "return_5d", "return_7d",
    "return_14d", "return_21d", "return_30d",
    "rsi_momentum", "price_position", "streak",
]


class NextDayPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        self._load()

    def _load(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "next_day_model.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.trained = True
            except Exception:
                self.trained = False

    def predict(self, features: dict) -> dict:
        """
        Predict next-day direction.
        Returns: {"direction": "up"/"down", "probability": 0.0-1.0, "features_used": [...]}
        """
        if not self.trained:
            return self._heuristic(features)

        x = np.array([[features.get(f, 0) for f in FEATURES]])
        x = np.nan_to_num(x, nan=0, posinf=1, neginf=-1)

        try:
            prob_up = self.model.predict_proba(x)[0, 1]
        except Exception:
            return self._heuristic(features)

        direction = "up" if prob_up > 0.5 else "down"
        confidence = abs(prob_up - 0.5) * 2  # 0 at 50%, 1 at 0%/100%

        return {
            "direction": direction,
            "probability": round(float(prob_up), 4),
            "confidence": round(float(confidence), 4),
        }

    def _heuristic(self, features: dict) -> dict:
        """Fallback when no trained model."""
        ret_1d = features.get("return_1d", 0)
        ret_7d = features.get("return_7d", 0)
        rsi = features.get("rsi", 50)

        # Simple momentum + mean reversion
        signal = ret_7d * 2 + (50 - rsi) / 100
        prob_up = 0.5 + signal * 0.1
        prob_up = max(0.3, min(0.7, prob_up))

        return {
            "direction": "up" if prob_up > 0.5 else "down",
            "probability": round(prob_up, 4),
            "confidence": round(abs(prob_up - 0.5) * 2, 4),
        }
