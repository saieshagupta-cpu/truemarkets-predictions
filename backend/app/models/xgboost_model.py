"""
XGBoost regime model for BTC direction prediction.
Uses features that DON'T overlap with CNN-LSTM (diversification).

CNN-LSTM handles: sequential price patterns (log_return, volatility, momentum, on-chain)
XGBoost handles: regime indicators (RSI, MACD, Fear&Greed, mean-reversion, streaks, day-of-week)

This diversification reduces sub-model correlation and improves ensemble accuracy.
"""

import os
import json
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from app.config import MODEL_WEIGHTS_DIR


class DirectionXGBoostPredictor:
    """Wrapper for trained XGBoost regime model."""
    def __init__(self):
        self.model = None
        self.trained = False
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_xgb.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.trained = True
            except Exception:
                self.trained = False

    def predict_direction(self, features: dict) -> float:
        """
        features: dict with keys matching training features.
        Returns probability of UP move.
        """
        if not self.trained or self.model is None:
            return 0.5

        try:
            # Build feature vector in training order
            feat_order = ["rsi", "macd_hist", "fear_greed", "fg_momentum",
                         "vol_regime", "mean_rev_z", "trend_strength",
                         "streak", "day_of_week", "hour_of_day"]
            x = np.array([[features.get(k, 0.0) for k in feat_order]])
            prob = self.model.predict_proba(x)[0][1]
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            return 0.5


def build_xgb_features(prices, rsi_raw, macd_hist, fear_greed=50, timestamps=None):
    """
    Build XGBoost features for a single prediction point.
    All features are regime/state indicators (not sequential patterns).
    """
    n = len(prices)
    if n < 5:
        return {"rsi": 0.5, "macd_hist": 0, "fear_greed": 50, "fg_momentum": 0,
                "vol_regime": 0.5, "mean_rev_z": 0, "trend_strength": 0,
                "streak": 0, "day_of_week": 0, "hour_of_day": 12}

    # RSI (0-1 scale)
    rsi = rsi_raw[-1] / 100.0 if isinstance(rsi_raw, (list, np.ndarray)) else rsi_raw / 100.0

    # MACD histogram (normalized)
    mh = macd_hist[-1] if isinstance(macd_hist, (list, np.ndarray)) else macd_hist

    # Fear & Greed (0-1 scale)
    fg = fear_greed / 100.0

    # F&G momentum (change over recent period, proxy)
    fg_mom = 0  # Will be filled in from actual data if available

    # Volatility regime: current vol vs 60-day median
    log_ret = np.diff(np.log(np.maximum(prices[-61:], 1e-10)))
    if len(log_ret) >= 20:
        vol_5 = np.std(log_ret[-5:])
        vol_60 = np.std(log_ret)
        vol_regime = vol_5 / max(vol_60, 1e-10)
    else:
        vol_regime = 1.0

    # Mean reversion z-score (20-period)
    window = prices[-21:]
    if len(window) >= 10:
        mean_rev_z = (prices[-1] - np.mean(window)) / max(np.std(window), 1e-10)
    else:
        mean_rev_z = 0

    # Trend strength: slope of 20-period linear regression
    if len(prices) >= 20:
        x_lr = np.arange(20)
        y_lr = prices[-20:]
        slope = np.polyfit(x_lr, y_lr, 1)[0]
        trend_strength = slope / max(np.mean(y_lr), 1e-10) * 20  # Normalized
    else:
        trend_strength = 0

    # Streak: consecutive up/down days
    streak = 0
    for i in range(len(prices)-1, 0, -1):
        if prices[i] > prices[i-1]:
            if streak >= 0:
                streak += 1
            else:
                break
        elif prices[i] < prices[i-1]:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break

    # Day of week (0-6)
    import pandas as pd
    if timestamps is not None:
        try:
            ts = pd.Timestamp(timestamps[-1] if hasattr(timestamps, '__len__') else timestamps)
            dow = ts.dayofweek / 6.0
            hod = ts.hour / 23.0
        except Exception:
            dow, hod = 0.5, 0.5
    else:
        dow, hod = 0.5, 0.5

    return {
        "rsi": float(rsi),
        "macd_hist": float(np.clip(mh, -3, 3)),
        "fear_greed": float(fg),
        "fg_momentum": float(fg_mom),
        "vol_regime": float(np.clip(vol_regime, 0, 5)),
        "mean_rev_z": float(np.clip(mean_rev_z, -4, 4)),
        "trend_strength": float(np.clip(trend_strength, -2, 2)),
        "streak": float(np.clip(streak / 10.0, -1, 1)),
        "day_of_week": float(dow),
        "hour_of_day": float(hod),
    }
