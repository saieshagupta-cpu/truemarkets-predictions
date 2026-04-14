import asyncio
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router, _cache, _chart_cache, _tm_data
from app.data.truemarkets_mcp import fetch_current_price, _fetch_price_data
from app.config import FRONTEND_URL

logger = logging.getLogger("truemarkets")

_refresh_task = None


async def _retrain_tcn():
    """Retrain TCN on latest data. Runs on startup and can be triggered."""
    try:
        logger.info("Retraining TCN on latest data...")
        from app.data.truemarkets_mcp import fetch_historical_prices
        from app.config import MODEL_WEIGHTS_DIR, SEQUENCE_LENGTH
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from app.models.direction_tcn import DirectionTCN
        import json, os

        df = await fetch_historical_prices("BTC", days=7)
        if len(df) < 50:
            logger.warning(f"Not enough data for retraining ({len(df)} points)")
            return

        prices = df["price"].values
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None
        n = len(prices)

        # Build features
        log_ret = np.concatenate([[0], np.diff(np.log(np.maximum(prices, 1e-10)))])
        vol_5 = pd.Series(log_ret).rolling(5, min_periods=1).std().fillna(0).values
        vol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().fillna(0.01).values
        price_pos = np.array([(lambda w: (prices[i]-w.min())/(w.max()-w.min()) if w.max()>w.min() else 0.5)(prices[max(0,i-20):i+1]) for i in range(n)])
        mom_5 = np.concatenate([np.zeros(5), [(prices[i]-prices[i-5])/prices[i-5] for i in range(5,n)]])
        mean_rev = np.array([(lambda w: (prices[i]-np.mean(w))/np.std(w) if np.std(w)>0 else 0)(prices[max(0,i-20):i+1]) for i in range(n)])
        accel = np.concatenate([np.zeros(2), [log_ret[i]-log_ret[i-1] for i in range(2,n)]])
        vol_ratio = np.where(vol_20 > 1e-10, vol_5/vol_20, 1.0)

        if timestamps is not None:
            hours = pd.to_datetime(timestamps).hour
            h_sin = np.sin(2*np.pi*hours/24).values
            h_cos = np.cos(2*np.pi*hours/24).values
        else:
            h_sin, h_cos = np.zeros(n), np.zeros(n)

        features = np.column_stack([log_ret, vol_5, vol_20, price_pos, mom_5, mean_rev, accel, vol_ratio, h_sin, h_cos])

        # Consensus sequences
        X, Y = [], []
        for i in range(SEQUENCE_LENGTH, len(features) - 6):
            dirs = [prices[min(i+h, n-1)] > prices[i] for h in [1, 2, 3, 4, 6]]
            up = sum(dirs)
            if all(dirs): label = 0.95
            elif not any(dirs): label = 0.05
            elif up >= 4: label = 0.85
            elif up <= 1: label = 0.15
            else: continue
            X.append(features[i-SEQUENCE_LENGTH:i])
            Y.append(label)

        X, Y = np.array(X), np.array(Y)
        if len(X) < 30:
            logger.warning(f"Not enough consensus sequences ({len(X)})")
            return

        split = int(len(X) * 0.8)
        X_aug = np.vstack([X[:split], X[:split] + np.random.randn(*X[:split].shape) * 0.002])
        Y_aug = np.hstack([Y[:split], Y[:split]])

        flat = X_aug.reshape(-1, X_aug.shape[-1])
        means, stds = flat.mean(0), flat.std(0)
        stds[stds == 0] = 1

        X_t = torch.FloatTensor((X_aug - means) / stds)
        Y_t = torch.FloatTensor(Y_aug)
        X_v = torch.FloatTensor((X[split:] - means) / stds)
        Y_v_hard = (torch.FloatTensor(Y[split:]) > 0.5).float()

        loader = DataLoader(TensorDataset(X_t, Y_t), batch_size=16, shuffle=True)
        model = DirectionTCN(input_size=10, num_channels=32, num_layers=3, kernel_size=3, dropout=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
        criterion = nn.BCELoss()

        best_acc, patience = 0, 0
        for epoch in range(200):
            model.train()
            for xb, yb in loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                va = ((model(X_v) > 0.5).float() == Y_v_hard).float().mean().item()

            if va > best_acc:
                best_acc = va; patience = 0
                os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn.pt"))
            else:
                patience += 1
                if patience >= 40: break

        with open(os.path.join(MODEL_WEIGHTS_DIR, "direction_tcn_norm.json"), "w") as f:
            json.dump({"means": means.tolist(), "stds": stds.tolist()}, f)

        logger.info(f"TCN retrained: {len(X)} sequences, val_acc={best_acc:.1%}")

    except Exception as e:
        logger.error(f"TCN retraining failed: {e}")


async def _refresh_loop():
    """Background loop: refresh BTC price and clear caches every 30 seconds."""
    while True:
        try:
            price_data = await fetch_current_price("BTC")
            if price_data and price_data.get("price", 0) > 0:
                _tm_data["price"] = price_data["price"]
                _tm_data["updated"] = time.time()
                try:
                    await _fetch_price_data("BTC", "1d", "1h")
                except Exception:
                    pass
                _cache.clear()
                _chart_cache.clear()
        except Exception:
            pass
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: retrain TCN on fresh data, then start refresh loop."""
    global _refresh_task
    # Retrain TCN with latest data every launch
    await _retrain_tcn()
    # Start background price refresh
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="True Markets Prediction Engine",
    description="TCN direction prediction + backtested signal ensemble for BTC",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "name": "True Markets Prediction Engine",
        "model": "TCN (retrained on every launch)",
        "signals": ["Technical (40%)", "TCN (30%)", "Order Flow (20%)", "Sentiment (10%)"],
        "refresh": "every 30 seconds",
        "docs": "/docs",
    }
