"""
CNN-LSTM model for BTC next-day direction prediction.
Architecture from: Omole & Enke (2024) "Deep learning for Bitcoin price direction
prediction: models and trading strategies empirically compared"
Financial Innovation 10:117

Architecture:
  Conv1D(n_feat→128, k=3) → BN → ReLU → Dropout(0.3)
  Conv1D(128→64, k=3)     → BN → ReLU → Dropout(0.3)
  LSTM(64→256, 2 layers)  → Dropout(0.5)
  Dense(256→128→64→1)     → Sigmoid

Window size: 5 days (paper's optimal from parameter study)
Feature selection: Boruta
Normalization: Min-Max [0,1]
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

WINDOW_SIZE = 5  # Paper's optimal


class CNNLSTM(nn.Module):
    """CNN-LSTM from Omole & Enke (2024)."""
    def __init__(self, n_features: int):
        super().__init__()
        # CNN block 1
        self.conv1 = nn.Conv1d(n_features, 128, kernel_size=min(3, WINDOW_SIZE), padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)

        # CNN block 2
        self.conv2 = nn.Conv1d(128, 64, kernel_size=min(3, WINDOW_SIZE), padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)

        # LSTM
        self.lstm = nn.LSTM(64, 256, num_layers=2, batch_first=True, dropout=0.5)

        # Dense
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size = x.size(0)

        # CNN: (batch, features, seq_len)
        out = x.permute(0, 2, 1)
        out = self.drop1(self.relu(self.bn1(self.conv1(out)) if batch_size > 1 else self.conv1(out)))
        out = self.drop2(self.relu(self.bn2(self.conv2(out)) if batch_size > 1 else self.conv2(out)))

        # LSTM: (batch, seq_len, 64)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = out[:, -1, :]  # last timestep: (batch, 256)

        # Dense
        out = self.drop3(self.relu(self.fc1(out)))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out.squeeze(-1)


class CNNLSTMPredictor:
    """Wrapper for trained CNN-LSTM model."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self.selected_features = None
        self.metrics = None
        self._load()

    def _load(self):
        from app.config import SAVED_DIR
        model_path = os.path.join(SAVED_DIR, "cnn_lstm.pt")
        norm_path = os.path.join(SAVED_DIR, "cnn_lstm_norm.json")
        metrics_path = os.path.join(SAVED_DIR, "cnn_lstm_metrics.json")

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)
                self.selected_features = self.norm_params.get("selected_features", [])

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.metrics = json.load(f)

        if os.path.exists(model_path) and self.norm_params:
            try:
                n_feat = self.norm_params.get("n_features", 10)
                self.model = CNNLSTM(n_features=n_feat)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception as e:
                print(f"[cnn_lstm] Failed to load: {e}")
                self.trained = False

    def predict(self, features: dict) -> float:
        """
        Predict probability of BTC UP tomorrow.
        features: dict of {feature_name: list of last WINDOW_SIZE values}
        """
        if not self.trained or not self.selected_features:
            return 0.5

        try:
            mins = np.array(self.norm_params["mins"])
            maxs = np.array(self.norm_params["maxs"])
            ranges = maxs - mins
            ranges[ranges == 0] = 1

            # Build sequence: (WINDOW_SIZE, n_features)
            seq = []
            for feat in self.selected_features:
                vals = features.get(feat, [0.0] * WINDOW_SIZE)
                if len(vals) < WINDOW_SIZE:
                    vals = [0.0] * (WINDOW_SIZE - len(vals)) + list(vals)
                seq.append(vals[-WINDOW_SIZE:])

            seq = np.array(seq).T  # (WINDOW_SIZE, n_features)
            seq = (seq - mins) / ranges

            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(seq).unsqueeze(0)
                prob = self.model(x).item()
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            return 0.5

    def get_accuracy(self) -> float:
        if self.metrics:
            return self.metrics.get("test_accuracy", 0.5)
        return 0.5
