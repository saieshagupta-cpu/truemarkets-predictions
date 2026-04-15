"""
Feature engineering for LightGBM BTC direction model.
35 features across 7 categories, all computable from OHLCV data.

Categories:
  1. Price & Returns (6)
  2. Volatility (5)
  3. Volume (5)
  4. Technical Indicators (6)
  5. Order Book / Microstructure (5)
  6. Market Structure (5)
  7. Temporal (3)
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 35 features from OHLCV DataFrame.
    Input columns required: price (or close), open, high, low, volume.
    Returns DataFrame with feature columns added.
    """
    p = df["price"].values if "price" in df.columns else df["close"].values
    o = df["open"].values if "open" in df.columns else p
    h = df["high"].values if "high" in df.columns else p
    lo = df["low"].values if "low" in df.columns else p
    v = df["volume"].values if "volume" in df.columns else np.ones(len(p))
    n = len(p)

    feats = pd.DataFrame(index=df.index)

    # ═══════════════════════════════════════════════════════
    # 1. PRICE & RETURNS (6)
    # ═══════════════════════════════════════════════════════
    feats["returns"] = pd.Series(p).pct_change().values
    feats["log_returns"] = np.concatenate([[0], np.diff(np.log(np.maximum(p, 1e-10)))])
    feats["price_momentum_1d"] = feats["returns"]
    feats["price_momentum_4d"] = pd.Series(p).pct_change(4).values
    feats["price_momentum_24d"] = pd.Series(p).pct_change(24).values
    feats["price_acceleration"] = feats["returns"].diff()

    # ═══════════════════════════════════════════════════════
    # 2. VOLATILITY (5)
    # ═══════════════════════════════════════════════════════
    feats["volatility_5d"] = pd.Series(feats["returns"]).rolling(5, min_periods=1).std().values
    feats["volatility_20d"] = pd.Series(feats["returns"]).rolling(20, min_periods=1).std().values
    feats["realized_volatility"] = feats["volatility_20d"] * np.sqrt(252)

    # Garman-Klass volatility: 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2
    log_hl = np.log(np.maximum(h, 1e-10) / np.maximum(lo, 1e-10))
    log_co = np.log(np.maximum(p, 1e-10) / np.maximum(o, 1e-10))
    gk_daily = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    feats["garman_klass_volatility"] = pd.Series(gk_daily).rolling(20, min_periods=1).mean().values

    # Parkinson volatility: ln(H/L)^2 / (4*ln2)
    park_daily = log_hl**2 / (4 * np.log(2))
    feats["parkinson_volatility"] = pd.Series(park_daily).rolling(20, min_periods=1).mean().values

    # ═══════════════════════════════════════════════════════
    # 3. VOLUME (5)
    # ═══════════════════════════════════════════════════════
    feats["volume_change"] = pd.Series(v).pct_change().values
    vol_5 = pd.Series(v).rolling(5, min_periods=1).mean().values
    vol_20 = pd.Series(v).rolling(20, min_periods=1).mean().values
    feats["volume_momentum"] = np.where(vol_20 > 0, vol_5 / vol_20, 1.0)
    feats["relative_volume"] = np.where(vol_20 > 0, v / vol_20, 1.0)

    # Volume-price trend
    vpt = np.cumsum(v * feats["returns"].fillna(0).values)
    feats["volume_price_trend"] = vpt / (np.abs(vpt).max() + 1e-10)  # normalize

    # OBV momentum
    obv = np.cumsum(np.where(np.diff(p, prepend=p[0]) > 0, v, -v))
    feats["obv_momentum"] = pd.Series(obv).diff(5).fillna(0).values
    feats["obv_momentum"] = feats["obv_momentum"] / (np.abs(feats["obv_momentum"]).max() + 1e-10)

    # ═══════════════════════════════════════════════════════
    # 4. TECHNICAL INDICATORS (6)
    # ═══════════════════════════════════════════════════════
    # RSI (14)
    delta = pd.Series(p).diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    feats["rsi"] = (100 - (100 / (1 + gain / loss_s.replace(0, np.nan)))).fillna(50).values

    # MACD histogram (12, 26, 9)
    ema12 = pd.Series(p).ewm(span=12).mean().values
    ema26 = pd.Series(p).ewm(span=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9).mean().values
    feats["macd"] = macd_line - signal_line  # histogram
    feats["macd"] = feats["macd"] / (np.abs(feats["macd"]).max() + 1e-10)

    # Bollinger position: (price - lower) / (upper - lower)
    sma20 = pd.Series(p).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(p).rolling(20, min_periods=1).std().fillna(1).values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    feats["bollinger_position"] = np.where(bb_range > 0, (p - bb_lower) / bb_range, 0.5)

    # Stochastic %K (14, 3)
    low14 = pd.Series(lo).rolling(14, min_periods=1).min().values
    high14 = pd.Series(h).rolling(14, min_periods=1).max().values
    k_range = high14 - low14
    feats["stochastic_k"] = np.where(k_range > 0, (p - low14) / k_range * 100, 50)

    # ATR ratio: ATR(14) / price
    tr = np.maximum(h - lo, np.maximum(np.abs(h - np.roll(p, 1)), np.abs(lo - np.roll(p, 1))))
    tr[0] = h[0] - lo[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    feats["atr_ratio"] = atr14 / np.maximum(p, 1e-10)

    # CCI (20): (price - SMA) / (0.015 * mean_deviation)
    sma20_cci = pd.Series(p).rolling(20, min_periods=1).mean().values
    mean_dev = pd.Series(np.abs(p - sma20_cci)).rolling(20, min_periods=1).mean().values
    feats["cci"] = (p - sma20_cci) / (0.015 * np.maximum(mean_dev, 1e-10))

    # ═══════════════════════════════════════════════════════
    # 5. ORDER BOOK / MICROSTRUCTURE (5)
    # ═══════════════════════════════════════════════════════
    # Historical proxies (live data from Binance at inference time)

    # Order book imbalance proxy: volume direction bias
    up_vol = np.where(np.diff(p, prepend=p[0]) > 0, v, 0)
    down_vol = np.where(np.diff(p, prepend=p[0]) <= 0, v, 0)
    up_sum = pd.Series(up_vol).rolling(10, min_periods=1).sum().values
    down_sum = pd.Series(down_vol).rolling(10, min_periods=1).sum().values
    total_sum = up_sum + down_sum
    feats["order_book_imbalance"] = np.where(total_sum > 0, (up_sum - down_sum) / total_sum, 0)

    # Trade flow imbalance: volume-weighted return sign
    signed_vol = v * np.sign(np.diff(p, prepend=p[0]))
    feats["trade_flow_imbalance"] = pd.Series(signed_vol).rolling(10, min_periods=1).sum().values
    feats["trade_flow_imbalance"] = feats["trade_flow_imbalance"] / (np.abs(feats["trade_flow_imbalance"]).max() + 1e-10)

    # Kyle's lambda: price impact = |return| / volume
    feats["kyle_lambda"] = np.abs(feats["returns"].fillna(0).values) / np.maximum(v, 1e-10)
    feats["kyle_lambda"] = pd.Series(feats["kyle_lambda"]).rolling(10, min_periods=1).mean().values
    feats["kyle_lambda"] = feats["kyle_lambda"] / (feats["kyle_lambda"].max() + 1e-10)

    # Amihud illiquidity: |return| / dollar_volume
    dollar_vol = v * p
    feats["amihud_illiquidity"] = np.abs(feats["returns"].fillna(0).values) / np.maximum(dollar_vol, 1e-10)
    feats["amihud_illiquidity"] = pd.Series(feats["amihud_illiquidity"]).rolling(10, min_periods=1).mean().values
    feats["amihud_illiquidity"] = feats["amihud_illiquidity"] / (feats["amihud_illiquidity"].max() + 1e-10)

    # Roll spread estimate: 2 * sqrt(-cov(delta_p_t, delta_p_t-1))
    dp = np.diff(p, prepend=p[0])
    roll_vals = np.zeros(n)
    for i in range(20, n):
        cov = np.cov(dp[i-20:i], dp[i-19:i+1] if i+1 <= n else dp[i-19:i])[0, 1] if i+1 <= n else 0
        roll_vals[i] = 2 * np.sqrt(max(-cov, 0))
    feats["roll_spread"] = roll_vals / (roll_vals.max() + 1e-10)

    # ═══════════════════════════════════════════════════════
    # 6. MARKET STRUCTURE (5)
    # ═══════════════════════════════════════════════════════
    feats["high_low_ratio"] = (h - lo) / np.maximum(p, 1e-10)
    feats["close_position"] = np.where((h - lo) > 0, (p - lo) / (h - lo), 0.5)
    feats["gap"] = o / np.roll(p, 1) - 1
    feats.loc[feats.index[0], "gap"] = 0

    feats["mean_reversion_z"] = np.array([
        (lambda w: (p[i] - np.mean(w)) / np.std(w) if np.std(w) > 0 else 0)(p[max(0, i-20):i+1])
        for i in range(n)
    ])

    trend = np.zeros(n)
    for i in range(20, n):
        x = np.arange(20)
        slope = np.polyfit(x, p[i-20:i], 1)[0]
        trend[i] = slope / max(np.mean(p[i-20:i]), 1e-10) * 20
    feats["trend_strength"] = np.clip(trend, -3, 3)

    # ═══════════════════════════════════════════════════════
    # 7. TEMPORAL (3)
    # ═══════════════════════════════════════════════════════
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        dow = ts.dt.dayofweek.values
        month = ts.dt.month.values
    else:
        dow = np.zeros(n)
        month = np.ones(n)
    feats["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)

    # Clean up
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats


def build_labels(prices: np.ndarray, horizon: int = 3) -> np.ndarray:
    """
    3-day direction labels.
    Label = 1 if average of next `horizon` prices > current price, else 0.
    """
    n = len(prices)
    labels = np.zeros(n)
    for i in range(n - horizon):
        future_avg = np.mean(prices[i+1:i+1+horizon])
        labels[i] = 1.0 if future_avg > prices[i] else 0.0
    return labels


FEATURE_NAMES = [
    # Price & Returns
    "returns", "log_returns", "price_momentum_1d", "price_momentum_4d",
    "price_momentum_24d", "price_acceleration",
    # Volatility
    "volatility_5d", "volatility_20d", "realized_volatility",
    "garman_klass_volatility", "parkinson_volatility",
    # Volume
    "volume_change", "volume_momentum", "relative_volume",
    "volume_price_trend", "obv_momentum",
    # Technical
    "rsi", "macd", "bollinger_position", "stochastic_k", "atr_ratio", "cci",
    # Microstructure
    "order_book_imbalance", "trade_flow_imbalance", "kyle_lambda",
    "amihud_illiquidity", "roll_spread",
    # Market Structure
    "high_low_ratio", "close_position", "gap", "mean_reversion_z", "trend_strength",
    # Temporal
    "day_of_week_sin", "day_of_week_cos", "month_sin",
]
