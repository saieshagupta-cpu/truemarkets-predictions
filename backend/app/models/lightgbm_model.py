"""
LightGBM wrapper for BTC 3-day direction prediction.
Loads trained model from disk and provides predict() interface.
"""

import os
import json
import pickle
import numpy as np
from app.config import LIGHTGBM_MODEL_PATH, BACKTEST_RESULTS_PATH
from app.models.feature_engineering import FEATURE_NAMES


class LightGBMPredictor:
    """Loads and runs the trained LightGBM model."""

    def __init__(self):
        self.model = None
        self.metrics = None
        self._load()

    def _load(self):
        if os.path.exists(LIGHTGBM_MODEL_PATH):
            try:
                with open(LIGHTGBM_MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"[lightgbm] Failed to load model: {e}")

        if os.path.exists(BACKTEST_RESULTS_PATH):
            try:
                with open(BACKTEST_RESULTS_PATH) as f:
                    self.metrics = json.load(f)
            except Exception:
                pass

    def predict(self, features: dict) -> float:
        """
        Predict probability of BTC being UP in 3 days.
        features: dict of feature_name -> value (matching FEATURE_NAMES).
        Returns: float in [0.05, 0.95].
        """
        if self.model is None:
            return 0.5

        try:
            x = np.array([[features.get(f, 0.0) for f in FEATURE_NAMES]])
            prob = self.model.predict(x)[0]
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            return 0.5

    def get_accuracy(self) -> float:
        if self.metrics:
            return self.metrics.get("test_accuracy", 0.5)
        return 0.5

    def get_feature_importance(self) -> dict:
        if self.metrics:
            return self.metrics.get("feature_importance", {})
        return {}
