"""
1D-CNN for direction prediction.
Captures local patterns (candle formations, micro-structures) that LSTMs and TCNs may miss.
Uses multi-scale kernels (like Inception) to detect patterns at different widths.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class MultiScaleConv(nn.Module):
    """Parallel convolutions with different kernel sizes (Inception-style)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 3 different kernel sizes to capture short/medium/long patterns
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels - 2 * (out_channels // 3), kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        out = torch.cat([c3, c5, c7], dim=1)
        return self.relu(self.bn(out))


class DirectionCNN(nn.Module):
    """Multi-scale 1D CNN for binary direction prediction."""
    def __init__(self, input_size=10, num_channels=30, dropout=0.25):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, num_channels, 1)

        self.layer1 = MultiScaleConv(num_channels, num_channels)
        self.layer2 = MultiScaleConv(num_channels, num_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.input_proj(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


class DirectionCNNPredictor:
    """Wrapper for trained CNN."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_cnn_norm.json")

        if os.path.exists(model_path):
            try:
                self.model = DirectionCNN(input_size=10, num_channels=30)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

    def predict_direction(self, features: np.ndarray) -> float:
        """features: (seq_len, 10) array."""
        if not self.trained or self.model is None:
            return 0.5

        if self.norm_params:
            means = np.array(self.norm_params["means"])
            stds = np.array(self.norm_params["stds"])
            stds[stds == 0] = 1
            features = (features - means) / stds

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
