"""
Training script for the True Markets prediction models.
Pulls historical data, engineers features, and trains LSTM + XGBoost models.

Usage: python -m train.train_models
"""

import asyncio
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data.coingecko import fetch_historical_prices
from app.models.lstm_model import LSTMPricePredictor
from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH

THRESHOLDS_BTC = [45000, 50000, 55000, 80000, 90000, 100000]
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


def create_sequences(data: np.ndarray, prices: np.ndarray, thresholds: list[float], seq_len: int = SEQUENCE_LENGTH):
    """Create training sequences with binary labels for each threshold."""
    X, Y = [], []
    for i in range(seq_len, len(data) - 30):
        X.append(data[i - seq_len : i])
        # Label: will price reach each threshold in next 30 days?
        future_max = prices[i : i + 30].max()
        future_min = prices[i : i + 30].min()
        labels = []
        current_price = prices[i]
        for t in thresholds:
            if t > current_price:
                labels.append(1.0 if future_max >= t else 0.0)
            else:
                labels.append(1.0 if future_min <= t else 0.0)
        Y.append(labels)
    return np.array(X), np.array(Y)


def train_lstm(df: pd.DataFrame, thresholds: list[float]):
    print("Training LSTM model...")
    feature_cols = ["price", "volume", "rsi", "macd", "bb_width", "volatility_20d"]
    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values
    prices = df["price"].values

    # Normalize
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds == 0] = 1
    data_norm = (data - means) / stds

    X, Y = create_sequences(data_norm, prices, thresholds)
    if len(X) == 0:
        print("Not enough data for LSTM training")
        return

    # Train/val split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMPricePredictor(input_size=len(available), num_thresholds=len(thresholds))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        train_loss /= len(train_loader)
        val_loss /= max(len(val_loader), 1)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "lstm_model.pt"))

    print(f"  LSTM trained. Best val_loss: {best_val_loss:.4f}")


def train_xgboost(df: pd.DataFrame, thresholds: list[float]):
    """Train simple logistic regression weights (XGBoost-style) for each threshold."""
    print("Training XGBoost feature model...")

    feature_cols = ["rsi", "macd", "bb_width", "volatility_20d", "volume_change"]
    available = [c for c in feature_cols if c in df.columns]
    prices = df["price"].values

    models = {}
    for threshold in thresholds:
        labels = []
        features = []
        for i in range(50, len(df) - 30):
            future_max = prices[i : i + 30].max()
            future_min = prices[i : i + 30].min()
            current = prices[i]

            if threshold > current:
                label = 1.0 if future_max >= threshold else 0.0
            else:
                label = 1.0 if future_min <= threshold else 0.0

            feat = {col: float(df[col].iloc[i]) for col in available}
            feat["price"] = current
            features.append(feat)
            labels.append(label)

        if not features:
            continue

        # Simple logistic regression via gradient descent
        feature_keys = list(features[0].keys())
        X = np.array([[f[k] for k in feature_keys] for f in features])
        y = np.array(labels)

        # Normalize features
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        stds[stds == 0] = 1
        X_norm = (X - means) / stds

        # Logistic regression
        weights = np.zeros(X_norm.shape[1])
        intercept = 0.0
        lr = 0.01

        for _ in range(200):
            z = X_norm @ weights + intercept
            pred = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
            error = pred - y
            weights -= lr * (X_norm.T @ error) / len(y)
            intercept -= lr * error.mean()

        weight_dict = {"intercept": float(intercept)}
        for i, key in enumerate(feature_keys):
            weight_dict[key] = float(weights[i])
        models[str(threshold)] = weight_dict

    os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
    with open(os.path.join(MODEL_WEIGHTS_DIR, "xgboost_models.json"), "w") as f:
        json.dump(models, f, indent=2)

    print(f"  XGBoost models trained for {len(models)} thresholds")


async def main():
    print("=" * 60)
    print("True Markets — Model Training Pipeline")
    print("=" * 60)

    print("\nFetching historical BTC data (365 days)...")
    df = await fetch_historical_prices("bitcoin", days=365)
    print(f"  Got {len(df)} data points with columns: {list(df.columns)}")

    train_lstm(df, THRESHOLDS_BTC)
    train_xgboost(df, THRESHOLDS_BTC)

    print("\nTraining complete! Models saved to:", MODEL_WEIGHTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
