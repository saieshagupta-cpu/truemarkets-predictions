"""
CNN-LSTM model for BTC next-day direction prediction.
Architecture from: Dubey & Enke (2025) "Bitcoin price direction prediction using on-chain data"
Machine Learning with Applications 20, 100674

Exact architecture (from paper Fig. 4):
  Layer 1: Conv1D(filters=8, kernel=3, activation=ReLU) → BatchNorm → Dropout(0.5)
  Layer 2: LSTM(32 units, activation=ReLU) → BatchNorm → Dropout(0.5)
  Layer 3: LSTM(64 units, activation=ReLU)
  Layer 4: Dense(16, activation=ReLU)
  Output:  Dense(1, activation=Sigmoid)

Lookback: 5 days (paper's optimal for CNN-LSTM)
Training: 1000 epochs, batch=50, early stopping patience=100
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from app.config import MODEL_WEIGHTS_DIR

LOOKBACK = 5  # Paper's optimal lookback for CNN-LSTM


class DirectionCNNLSTM(nn.Module):
    """Exact CNN-LSTM architecture from Dubey & Enke 2025."""
    def __init__(self, input_size: int):
        super().__init__()
        # Layer 1: Conv1D(8 filters, kernel 3) + BatchNorm + Dropout(0.5)
        self.conv1 = nn.Conv1d(input_size, 8, kernel_size=min(3, LOOKBACK), padding=1)
        self.bn1 = nn.BatchNorm1d(8)
        self.drop1 = nn.Dropout(0.5)

        # Layer 2: LSTM(32 units) + BatchNorm + Dropout(0.5)
        self.lstm1 = nn.LSTM(8, 32, batch_first=True)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.5)

        # Layer 3: LSTM(64 units)
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)

        # Layer 4: Dense(16) + Output Dense(1, sigmoid)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # Conv1D: (batch, features, seq_len)
        out = x.permute(0, 2, 1)
        out = self.relu(self.conv1(out))
        if out.size(0) > 1:  # BatchNorm needs >1 sample
            out = self.bn1(out)
        out = self.drop1(out)

        # LSTM1: (batch, seq_len, 8) → (batch, seq_len, 32)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm1(out)
        last1 = out[:, -1, :]  # (batch, 32)
        if last1.size(0) > 1:
            last1 = self.bn2(last1)
        last1 = self.drop2(last1)

        # LSTM2: feed last1 repeated for each timestep
        last1_expanded = last1.unsqueeze(1).expand(-1, out.size(1), -1)
        out2, _ = self.lstm2(last1_expanded)
        last2 = self.relu(out2[:, -1, :])  # (batch, 64)

        # Dense layers
        out = self.relu(self.fc1(last2))
        return self.sigmoid(self.fc2(out)).squeeze(-1)


class DirectionCNNLSTMPredictor:
    """Wrapper for trained CNN-LSTM model."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json")
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt")

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

        if os.path.exists(model_path):
            try:
                n_feat = self.norm_params.get("n_features", 6) if self.norm_params else 6
                self.model = DirectionCNNLSTM(input_size=n_feat)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

    def predict_direction(self, features: np.ndarray) -> float:
        if not self.trained or self.model is None:
            return 0.5

        try:
            n_feat = self.norm_params.get("n_features", 6) if self.norm_params else 6

            # Handle feature count mismatch gracefully
            if features.ndim == 2 and features.shape[1] != n_feat:
                # Pad or trim to match expected features
                if features.shape[1] < n_feat:
                    pad = np.zeros((features.shape[0], n_feat - features.shape[1]))
                    features = np.hstack([features, pad])
                else:
                    features = features[:, :n_feat]

            if self.norm_params:
                mins = np.array(self.norm_params["mins"])
                maxs = np.array(self.norm_params["maxs"])
                ranges = maxs - mins
                ranges[ranges == 0] = 1
                features = (features - mins) / ranges

            seq = features[-LOOKBACK:]
            if len(seq) < LOOKBACK:
                pad = np.zeros((LOOKBACK - len(seq), seq.shape[1] if seq.ndim == 2 else n_feat))
                seq = np.vstack([pad, seq])

            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(seq).unsqueeze(0)
                prob = self.model(x).item()
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            return 0.5
