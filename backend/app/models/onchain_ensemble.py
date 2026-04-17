"""
On-chain ensemble model for BTC next-day direction prediction.
Uses GradientBoosting + RandomForest on Boruta-selected features from
BGeometrics on-chain data + rate-of-change engineering.

Based on: Omole & Enke (2024) feature selection methodology
with gradient boosting (better than CNN-LSTM on 1825 samples).
"""

import os
import json
import pickle
import numpy as np

WINDOW_SIZE = 1  # Tabular model, no window needed


class OnchainEnsemblePredictor:
    """Loads and runs the trained on-chain GB + RF ensemble (Boruta feature-selected)."""

    def __init__(self):
        self.gb_model = None
        self.rf_model = None
        self.trained = False
        self.norm_params = None
        self.selected_features = None
        self.metrics = None
        self._load()

    def _load(self):
        from app.config import SAVED_DIR
        gb_path = os.path.join(SAVED_DIR, "model_gb.pkl")
        rf_path = os.path.join(SAVED_DIR, "model_rf.pkl")
        norm_path = os.path.join(SAVED_DIR, "onchain_ensemble_norm.json")
        metrics_path = os.path.join(SAVED_DIR, "onchain_ensemble_metrics.json")

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)
                self.selected_features = self.norm_params.get("selected_features", [])

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        if os.path.exists(gb_path):
            try:
                with open(gb_path, "rb") as f:
                    self.gb_model = pickle.load(f)
            except Exception as e:
                print(f"[model] Failed to load GB: {e}")

        if os.path.exists(rf_path):
            try:
                with open(rf_path, "rb") as f:
                    self.rf_model = pickle.load(f)
            except Exception as e:
                print(f"[model] Failed to load RF: {e}")

        self.trained = self.gb_model is not None or self.rf_model is not None

    def predict(self, features: dict) -> float:
        """
        Predict probability of BTC UP tomorrow.
        features: dict of {feature_name: value} (latest on-chain values).
        """
        if not self.trained or not self.selected_features:
            return 0.5

        try:
            x = np.array([[features.get(f, 0.0) for f in self.selected_features]])

            probs = []
            if self.gb_model:
                probs.append(self.gb_model.predict_proba(x)[0][1])
            if self.rf_model:
                probs.append(self.rf_model.predict_proba(x)[0][1])

            if probs:
                return float(np.clip(np.mean(probs), 0.05, 0.95))
            return 0.5
        except Exception:
            return 0.5

    def get_accuracy(self) -> float:
        if self.metrics:
            return self.metrics.get("test_accuracy", 0.5)
        return 0.5
