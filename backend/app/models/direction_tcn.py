"""
Temporal Convolutional Network (TCN) for direction prediction.
TCNs use dilated causal convolutions — they see patterns at multiple time scales
without the vanishing gradient issues of RNNs. Often outperform LSTMs on time series.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class CausalConv1d(nn.Module):
    """Causal convolution: output at time t only depends on inputs at time <= t."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # remove future padding
        return out


class TCNBlock(nn.Module):
    """Residual block with two causal convolutions + skip connection."""
    def __init__(self, channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return out + residual  # skip connection


class DirectionTCN(nn.Module):
    """TCN for binary direction prediction."""
    def __init__(self, input_size=10, num_channels=32, num_layers=3, kernel_size=3, dropout=0.2):
        super().__init__()
        # Project input features to channel dimension
        self.input_proj = nn.Conv1d(input_size, num_channels, 1)

        # Stack of TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(num_channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, features) → (batch, features, seq_len) for conv1d
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(-1)


class DirectionTCNPredictor:
    """Wrapper for trained TCN."""
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        model_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_norm.json")

        if os.path.exists(model_path):
            try:
                self.model = DirectionTCN(input_size=10, num_channels=48, num_layers=4)
                self.model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
            except Exception:
                self.trained = False

        if os.path.exists(norm_path):
            with open(norm_path) as f:
                self.norm_params = json.load(f)

    def predict_direction(self, features: np.ndarray) -> float:
        """features: (seq_len, 10) array of TCN features."""
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
