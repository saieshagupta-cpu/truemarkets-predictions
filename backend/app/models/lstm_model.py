import numpy as np
import torch
import torch.nn as nn
import os
from app.config import SEQUENCE_LENGTH, MODEL_WEIGHTS_DIR


# ─── Legacy LSTM (backward compat) ──────────────────────

class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 64, num_layers: int = 2, num_thresholds: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_thresholds), nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


# ─── Attention LSTM (new) ────────────────────────────────

class MultiHeadAttention(nn.Module):
    """Self-attention over LSTM timestep outputs."""
    def __init__(self, hidden_size: int, num_heads: int = 2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_out)  # residual connection


class AttentionLSTMPredictor(nn.Module):
    """LSTM + self-attention: learns which days in the window matter most."""
    def __init__(self, input_size: int = 6, hidden_size: int = 64,
                 num_layers: int = 2, num_thresholds: int = 6, num_heads: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_thresholds), nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        attn_out = self.attention(lstm_out)    # (batch, seq_len, hidden)
        pooled = attn_out.mean(dim=1)          # (batch, hidden) — weighted average
        return self.fc(pooled)


# ─── Predictor wrapper ───────────────────────────────────

class LSTMPredictor:
    def __init__(self, num_thresholds: int = 6):
        self.num_thresholds = num_thresholds
        self.model = None
        self.trained = False
        self._load_weights()

    def _load_weights(self):
        # Try attention model first, then legacy, then heuristic
        attn_path = os.path.join(MODEL_WEIGHTS_DIR, "lstm_attention_model.pt")
        legacy_path = os.path.join(MODEL_WEIGHTS_DIR, "lstm_model.pt")

        if os.path.exists(attn_path):
            self.model = AttentionLSTMPredictor(num_thresholds=self.num_thresholds)
            try:
                self.model.load_state_dict(torch.load(attn_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
                return
            except Exception:
                pass

        if os.path.exists(legacy_path):
            self.model = LSTMPricePredictor(num_thresholds=self.num_thresholds)
            try:
                self.model.load_state_dict(torch.load(legacy_path, map_location="cpu", weights_only=True))
                self.model.eval()
                self.trained = True
                return
            except Exception:
                pass

        self.trained = False

    def predict(self, price_data: np.ndarray, thresholds: list[float], current_price: float, volatility: float) -> dict[str, float]:
        if self.trained and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(price_data).unsqueeze(0)
                probs = self.model(x).squeeze().numpy()
            results = {}
            for i, threshold in enumerate(thresholds):
                if i < len(probs):
                    results[str(threshold)] = float(np.clip(probs[i], 0.01, 0.99))
            return results

        return self._heuristic_predict(current_price, volatility, thresholds)

    def _heuristic_predict(self, current_price: float, volatility: float, thresholds: list[float]) -> dict[str, float]:
        if current_price <= 0:
            return {str(t): 0.5 for t in thresholds}

        daily_vol = max(volatility, 0.01)
        horizon_vol = daily_vol * np.sqrt(30)

        results = {}
        for threshold in thresholds:
            pct_move = (threshold - current_price) / current_price
            if threshold > current_price:
                z = pct_move / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))
            else:
                z = abs(pct_move) / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))
            results[str(threshold)] = float(np.clip(prob, 0.01, 0.99))
        return results


def _norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
