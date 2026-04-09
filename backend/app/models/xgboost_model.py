import numpy as np
import os
import pickle
from app.config import MODEL_WEIGHTS_DIR

# Full feature list for the GBM model
GBM_FEATURES = [
    "rsi", "macd", "bb_width", "volatility", "volume_change",
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    "relative_volume", "volatility_ratio", "rsi_momentum",
    "price_position", "bollinger_position",
    "fear_greed", "sentiment_score",
]


class XGBoostPredictor:
    def __init__(self):
        self.models = []
        self._load_models()

    def _load_models(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "gbm_models.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.models = pickle.load(f)
                self.trained = True
            except Exception:
                self.trained = False
        else:
            self.trained = False

    def predict(self, features: dict, thresholds: list[float]) -> dict[str, float]:
        if self.trained and self.models:
            return self._trained_predict(features, thresholds)
        return self._heuristic_predict(features, thresholds)

    def _trained_predict(self, features: dict, thresholds: list[float]) -> dict[str, float]:
        # Build feature vector in the correct order
        x = np.array([[features.get(f, 0) for f in GBM_FEATURES]])
        # Handle NaN/inf
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        results = {}
        for i, threshold in enumerate(thresholds):
            if i < len(self.models) and self.models[i] is not None:
                try:
                    prob = self.models[i].predict_proba(x)[0, 1] if len(self.models[i].classes_) > 1 else 0.5
                except Exception:
                    prob = 0.5
            else:
                prob = 0.5
            results[str(threshold)] = float(np.clip(prob, 0.01, 0.99))
        return results

    def _heuristic_predict(self, features: dict, thresholds: list[float]) -> dict[str, float]:
        """Feature-driven heuristic using all available features."""
        current_price = features.get("price", 0)
        if current_price <= 0:
            return {str(t): 0.5 for t in thresholds}

        rsi = features.get("rsi", 50)
        fear_greed = features.get("fear_greed", 50)
        sentiment = features.get("sentiment_score", 0)
        volatility = features.get("volatility", 0.02)
        vol_ratio = features.get("volatility_ratio", 1.0)
        rsi_mom = features.get("rsi_momentum", 0)
        price_pos = features.get("price_position", 0.5)
        ret_7d = features.get("return_7d", 0)
        ret_30d = features.get("return_30d", 0)
        rel_vol = features.get("relative_volume", 1.0)

        # Multi-factor directional bias: -1 (bearish) to +1 (bullish)
        bias = 0.0
        bias += np.clip((rsi - 50) / 50, -1, 1) * 0.15           # RSI level
        bias += np.clip(rsi_mom / 20, -1, 1) * 0.10               # RSI momentum
        bias += np.clip(sentiment * 2, -1, 1) * 0.10              # Social sentiment
        bias += np.clip((fear_greed - 50) / 50, -1, 1) * 0.15    # Fear/Greed
        bias += np.clip(ret_7d * 10, -1, 1) * 0.15               # 7d momentum
        bias += np.clip(ret_30d * 5, -1, 1) * 0.10               # 30d trend
        bias += np.clip((price_pos - 0.5) * 2, -1, 1) * 0.10    # Range position
        # Volume expansion with trend = conviction
        vol_signal = np.clip((rel_vol - 1) * 2, -1, 1) * np.sign(ret_7d)
        bias += vol_signal * 0.10
        # Volatility regime: high vol ratio = expanding vol = bigger moves
        vol_regime = np.clip(vol_ratio - 1, -0.5, 0.5) * 0.05

        daily_vol = max(volatility, 0.005)
        horizon_vol = daily_vol * np.sqrt(30)

        results = {}
        for threshold in thresholds:
            pct_move = (threshold - current_price) / current_price
            # Adjust volatility for regime
            adj_vol = horizon_vol * (1 + vol_regime)

            if threshold > current_price:
                z = (pct_move - bias * 0.06) / max(adj_vol, 0.01)
                prob = 2 * (1 - _norm_cdf(z))
            else:
                z = (abs(pct_move) + bias * 0.06) / max(adj_vol, 0.01)
                prob = 2 * (1 - _norm_cdf(z))

            results[str(threshold)] = float(np.clip(prob, 0.01, 0.99))
        return results


def _norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
