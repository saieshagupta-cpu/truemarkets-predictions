"""
Direction LSTM: predicts next-period price direction (up/down).
Uses price-only sequential features — no RSI/MACD overlap with XGBoost.
Small architecture to avoid overfitting on limited data.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class DirectionLSTM(nn.Module):
    """Compact LSTM for directional prediction."""
    def __init__(self, input_size: int = 6, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.dropout(hn[-1])
        return self.fc(out).squeeze(-1)


class DirectionLSTMPredictor:
    """Wrapper that loads trained weights and produces direction probabilities."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_lstm.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_lstm_norm.json")

        if os.path.exists(model_path):
            try:
                self.model = DirectionLSTM(input_size=6, hidden_size=32, num_layers=1)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

    def predict_direction(self, price_series: np.ndarray, timestamps: np.ndarray = None) -> float:
        """
        Predict probability of next-period UP.
        price_series: array of recent prices (at least SEQUENCE_LENGTH points)
        timestamps: optional array of timestamps for hour-of-day encoding
        Returns: float 0-1 (probability of UP)
        """
        if len(price_series) < 3:
            return 0.5

        features = self._build_features(price_series, timestamps)

        if self.trained and self.model is not None:
            # Normalize
            if self.norm_params:
                means = np.array(self.norm_params["means"])
                stds = np.array(self.norm_params["stds"])
                stds[stds == 0] = 1
                features = (features - means) / stds

            seq_len = min(SEQUENCE_LENGTH, len(features))
            seq = features[-seq_len:]

            # Pad if needed
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]))
                seq = np.vstack([pad, seq])

            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(seq).unsqueeze(0)
                prob = self.model(x).item()
            return float(np.clip(prob, 0.05, 0.95))

        # Heuristic fallback
        return self._heuristic(price_series)

    def _build_features(self, prices: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
        """Build 6 features: log_return, vol_5, vol_20, price_position, hour_sin, hour_cos."""
        n = len(prices)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        log_returns = np.concatenate([[0], log_returns])

        # Rolling volatilities
        vol_5 = np.array([np.std(log_returns[max(0, i-5):i+1]) if i >= 1 else 0 for i in range(n)])
        vol_20 = np.array([np.std(log_returns[max(0, i-20):i+1]) if i >= 1 else 0 for i in range(n)])

        # Price position in rolling window
        price_pos = np.zeros(n)
        for i in range(n):
            window = prices[max(0, i-20):i+1]
            hi, lo = window.max(), window.min()
            if hi > lo:
                price_pos[i] = (prices[i] - lo) / (hi - lo)
            else:
                price_pos[i] = 0.5

        # Hour encoding
        if timestamps is not None and len(timestamps) == n:
            import pandas as pd
            hours = pd.to_datetime(timestamps).hour
            hour_sin = np.sin(2 * np.pi * hours / 24)
            hour_cos = np.cos(2 * np.pi * hours / 24)
        else:
            hour_sin = np.zeros(n)
            hour_cos = np.zeros(n)

        features = np.column_stack([log_returns, vol_5, vol_20, price_pos, hour_sin, hour_cos])
        return features

    def _heuristic(self, prices: np.ndarray) -> float:
        """Simple momentum + mean-reversion heuristic."""
        if len(prices) < 5:
            return 0.5

        # Short-term momentum (last 3 periods)
        ret_3 = (prices[-1] - prices[-4]) / prices[-4] if prices[-4] > 0 else 0

        # Medium momentum (last 12 periods if available)
        lookback = min(12, len(prices) - 1)
        ret_m = (prices[-1] - prices[-lookback-1]) / prices[-lookback-1] if prices[-lookback-1] > 0 else 0

        # Combine: slight momentum bias
        signal = ret_3 * 10 + ret_m * 5
        prob = 1 / (1 + np.exp(-signal))
        return float(np.clip(prob, 0.2, 0.8))
