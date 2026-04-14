"""
Transformer encoder for direction prediction.
Self-attention learns which timesteps matter most for the next move.
Positional encoding preserves time ordering.
"""

import os, json, math
import numpy as np
import torch
import torch.nn as nn
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DirectionTransformer(nn.Module):
    def __init__(self, input_size=10, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 16), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]  # take last timestep
        return self.head(x).squeeze(-1)


class DirectionTransformerPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        self.norm_params = None
        self._load()

    def _load(self):
        path = os.path.join(MODEL_WEIGHTS_DIR, "direction_transformer.pt")
        norm_path = os.path.join(MODEL_WEIGHTS_DIR, "direction_transformer_norm.json")
        if os.path.exists(path):
            try:
                self.model = DirectionTransformer(input_size=10, d_model=32, nhead=4, num_layers=2)
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
