"""
WaveNet-style model with gated dilated causal convolutions.
Originally designed for audio generation — excellent at capturing
multi-scale temporal patterns in sequential data.
"""

import os, json
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class GatedDilatedConv(nn.Module):
    """Gated activation: tanh(conv_filter) * sigmoid(conv_gate)"""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        f = self.conv_filter(x)[:, :, :x.size(2)]
        g = self.conv_gate(x)[:, :, :x.size(2)]
        activated = torch.tanh(f) * torch.sigmoid(g)
        residual = self.residual(activated) + x
        skip = self.skip(activated)
        return residual, skip


class DirectionWaveNet(nn.Module):
    def __init__(self, input_size=10, channels=32, num_layers=4, kernel_size=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_size, channels, 1)
        self.layers = nn.ModuleList([
            GatedDilatedConv(channels, kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, 16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        skip_sum = 0
        for layer in self.layers:
            x, skip = layer(x)
            skip_sum = skip_sum + skip
        x = self.dropout(skip_sum)
        return self.head(x).squeeze(-1)


class DirectionWaveNetPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "direction_wavenet.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_wavenet_norm.json")
        if os.path.exists(path):
            try:
                self.model = DirectionWaveNet(input_size=10, channels=32, num_layers=4)
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
