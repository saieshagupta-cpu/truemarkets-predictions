"""
Direction XGBoost: predicts next-period direction using regime + technical features.
Deliberately uses DIFFERENT features from LSTM to reduce ensemble correlation.
"""

import os
import pickle
import numpy as np
from app.config import MODEL_WEIGHTS_DIR


# Features for the GBM model — no overlap with LSTM's price-only features
DIRECTION_XGB_FEATURES = [
    "rsi", "macd_hist", "volatility_regime", "mean_reversion_z",
    "trend_sign", "streak", "hour_of_day", "rsi_momentum",
    "bb_position", "price_acceleration",
]


class DirectionXGBPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        self._load()

    def _load(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "direction_xgb.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                self.trained = True
            except Exception:
                self.trained = False

    def predict_direction(self, features: dict) -> float:
        """
        Predict probability of next-period UP.
        features: dict with keys from DIRECTION_XGB_FEATURES
        Returns: float 0-1
        """
        if self.trained and self.model is not None:
            x = np.array([[features.get(f, 0) for f in DIRECTION_XGB_FEATURES]])
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            try:
                if hasattr(self.model, "predict_proba"):
                    prob = self.model.predict_proba(x)[0, 1]
                else:
                    prob = self.model.predict(x)[0]
                return float(np.clip(prob, 0.05, 0.95))
            except Exception:
                pass

        return self._heuristic(features)

    def _heuristic(self, features: dict) -> float:
        """Multi-factor regime heuristic."""
        rsi = features.get("rsi", 50)
        macd_hist = features.get("macd_hist", 0)
        vol_regime = features.get("volatility_regime", 1.0)
        mean_rev_z = features.get("mean_reversion_z", 0)
        trend = features.get("trend_sign", 0)
        streak = features.get("streak", 0)

        signal = 0.0

        # RSI mean-reversion (strongest at extremes)
        if rsi < 30:
            signal += 0.25  # oversold → expect bounce
        elif rsi > 70:
            signal -= 0.25  # overbought → expect pullback
        else:
            signal += (rsi - 50) / 200  # mild momentum in middle zone

        # MACD histogram momentum
        signal += np.clip(macd_hist / 500, -0.2, 0.2)

        # Mean reversion: far from mean → expect reversion
        signal -= np.clip(mean_rev_z * 0.1, -0.15, 0.15)

        # Trend following (mild)
        signal += trend * 0.05

        # Streak: long streaks tend to reverse
        if abs(streak) >= 4:
            signal -= np.sign(streak) * 0.1

        # Volatility regime: expanding vol reduces confidence
        if vol_regime > 1.5:
            signal *= 0.7  # dampen in high vol

        prob = 0.5 + signal
        return float(np.clip(prob, 0.2, 0.8))

    @staticmethod
    def build_features(prices, rsi_values, macd_values, macd_signal_values,
                       bb_positions, volatility_5d, volatility_20d, timestamps=None):
        """
        Build feature dict from raw indicator arrays.
        All inputs should be arrays of same length.
        """
        n = len(prices)
        features_list = []

        for i in range(n):
            rsi = float(rsi_values[i]) if not np.isnan(rsi_values[i]) else 50
            macd_hist = float(macd_values[i] - macd_signal_values[i]) if not (np.isnan(macd_values[i]) or np.isnan(macd_signal_values[i])) else 0

            vol_5 = float(volatility_5d[i]) if not np.isnan(volatility_5d[i]) else 0.02
            vol_20 = float(volatility_20d[i]) if not np.isnan(volatility_20d[i]) else 0.02
            vol_regime = vol_5 / vol_20 if vol_20 > 0 else 1.0

            # Mean reversion z-score: how far price is from 20-period SMA
            window = prices[max(0, i-20):i+1]
            sma = np.mean(window)
            std = np.std(window) if len(window) > 1 else 1
            mean_rev_z = (prices[i] - sma) / std if std > 0 else 0

            # Trend sign: sign of last 7 periods return
            if i >= 7:
                trend_sign = np.sign(prices[i] - prices[i-7])
            else:
                trend_sign = 0

            # Streak: consecutive up/down periods
            streak = 0
            for j in range(i, max(i-10, 0), -1):
                if j == 0:
                    break
                if prices[j] > prices[j-1]:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                elif prices[j] < prices[j-1]:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
                else:
                    break

            # Hour of day
            if timestamps is not None and len(timestamps) > i:
                import pandas as pd
                hour = pd.to_datetime(timestamps[i]).hour
            else:
                hour = 12

            # RSI momentum
            if i >= 5:
                rsi_mom = rsi - float(rsi_values[i-5]) if not np.isnan(rsi_values[i-5]) else 0
            else:
                rsi_mom = 0

            bb_pos = float(bb_positions[i]) if not np.isnan(bb_positions[i]) else 0.5

            # Price acceleration: second derivative of returns
            if i >= 2:
                ret_1 = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] > 0 else 0
                ret_2 = (prices[i-1] - prices[i-2]) / prices[i-2] if prices[i-2] > 0 else 0
                accel = ret_1 - ret_2
            else:
                accel = 0

            features_list.append({
                "rsi": rsi,
                "macd_hist": macd_hist,
                "volatility_regime": vol_regime,
                "mean_reversion_z": mean_rev_z,
                "trend_sign": trend_sign,
                "streak": streak,
                "hour_of_day": hour / 24.0,  # normalize to 0-1
                "rsi_momentum": rsi_mom,
                "bb_position": bb_pos,
                "price_acceleration": accel * 100,
            })

        return features_list
