"""
CNN-LSTM Hybrid: CNN extracts local patterns, LSTM captures sequence dependencies.
Best of both worlds — local feature extraction + temporal memory.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class DirectionCNNLSTM(nn.Module):
    def __init__(self, input_size=10, cnn_channels=32, lstm_hidden=24, dropout=0.2):
        super().__init__()
        # CNN feature extractor (local patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(),
        )
        # LSTM over CNN features (temporal dependencies)
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        cnn_in = x.permute(0, 2, 1)       # (batch, features, seq_len)
        cnn_out = self.cnn(cnn_in)          # (batch, channels, seq_len)
        lstm_in = cnn_out.permute(0, 2, 1) # (batch, seq_len, channels)
        _, (hn, _) = self.lstm(lstm_in)
        return self.head(hn[-1]).squeeze(-1)


class DirectionCNNLSTMPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_lstm_norm.json")
        if os.path.exists(path):
            try:
                self.model = DirectionCNNLSTM(input_size=10, cnn_channels=32, lstm_hidden=24)
                self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

    def predict_direction(self, features):
        if not self.trained or self.model is None:
            return 0.5
        if self.norm_params:
            means = np.array(self.norm_params["means"])
            stds = np.array(self.norm_params["stds"])
            stds[stds == 0] = 1
            features = (features - means) / stds
        seq = features[-SEQUENCE_LENGTH:]
        if len(seq) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(seq), seq.shape[1]))
            seq = np.vstack([pad, seq])
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            return float(np.clip(self.model(x).item(), 0.05, 0.95))
