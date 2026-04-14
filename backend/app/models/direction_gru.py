"""
GRU model for BTC next-day direction prediction.
Based on: PMC11935774 — "Cryptocurrency Price Prediction Using GRU"

Architecture (from paper):
  - 2-layer GRU, 100 units each
  - Dropout 0.2
  - Adam optimizer
  - MinMaxScaler on input
  - Daily closing prices as input

Adapted for direction prediction (paper predicts price, we predict direction).
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH


class DirectionGRU(nn.Module):
    """2-layer GRU exactly as described in the paper."""
    def __init__(self, input_size: int = 1, hidden_size: int = 100, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])  # take last timestep
        return self.sigmoid(self.fc(out)).squeeze(-1)


class DirectionGRUPredictor:
    """Wrapper that loads trained GRU weights and predicts direction."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_gru.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_gru_norm.json")

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

        if os.path.exists(model_path):
            try:
                n_feat = self.norm_params.get("n_features", 1) if self.norm_params else 1
                self.model = DirectionGRU(input_size=n_feat, hidden_size=100, num_layers=2, dropout=0.2)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

    def predict_direction(self, features: np.ndarray) -> float:
        """features: (seq_len, n_features) array. Returns probability of UP."""
        if not self.trained or self.model is None:
            return 0.5

        if self.norm_params:
            mins = np.array(self.norm_params["mins"])
            maxs = np.array(self.norm_params["maxs"])
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            features = (features - mins) / ranges

        seq_len = min(SEQUENCE_LENGTH, len(features))
        seq = features[-seq_len:]
        if len(seq) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]))
            seq = np.vstack([pad, seq])

        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            prob = self.model(x).item()
        return float(np.clip(prob, 0.05, 0.95))
