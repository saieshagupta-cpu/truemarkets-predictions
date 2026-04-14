import numpy as np


class SentimentPredictor:
    def predict(self, sentiment_data: dict, fear_greed_data: dict, thresholds: list[float], current_price: float, volatility: float) -> dict[str, float]:
        """
        Sentiment-adjusted probability using social + fear/greed signals.
        """
        if current_price <= 0:
            return {str(t): 0.5 for t in thresholds}

        sentiment_score = sentiment_data.get("sentiment_score", 0)
        bullish_ratio = sentiment_data.get("bullish_ratio", 0.5)
        fg_value = fear_greed_data.get("current", {}).get("value", 50)
        fg_avg = fear_greed_data.get("average_30d", 50)

        # Combined directional signal: -1 to +1
        signal = 0.0
        signal += np.clip(sentiment_score * 2, -1, 1) * 0.30
        signal += (bullish_ratio - 0.5) * 2 * 0.25
        signal += (fg_value - 50) / 50 * 0.30
        signal += (fg_value - fg_avg) / 50 * 0.15

        daily_vol = max(volatility, 0.01)
        horizon_vol = daily_vol * np.sqrt(30)

        results = {}
        for threshold in thresholds:
            pct_move = (threshold - current_price) / current_price

            if threshold > current_price:
                z = (pct_move - signal * 0.04) / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))
            else:
                z = (abs(pct_move) + signal * 0.04) / horizon_vol
                prob = 2 * (1 - _norm_cdf(z))

            results[str(threshold)] = float(np.clip(prob, 0.01, 0.99))
        return results

    def predict_direction(self, fg_value: float = 50, fg_avg: float = 50,
                          sentiment_score: float = 0, order_flow: float = 0) -> float:
        """
        Contrarian at extremes, momentum in the middle.
        Returns probability of next-period UP (0-1).
        """
        # Extreme Fear → contrarian bullish (crowd is wrong at extremes)
        if fg_value < 20:
            base = 0.65 + (20 - fg_value) * 0.005  # up to 0.75
        # Extreme Greed → contrarian bearish
        elif fg_value > 80:
            base = 0.35 - (fg_value - 80) * 0.005  # down to 0.25
        else:
            # Middle zone: momentum-following blend
            signal = (
                sentiment_score * 0.35 +
                (fg_value - 50) / 100 * 0.35 +
                order_flow * 0.15 +
                (fg_value - fg_avg) / 100 * 0.15
            )
            base = 0.5 + np.clip(signal, -0.2, 0.2)

        return float(np.clip(base, 0.15, 0.85))

    def get_signal_breakdown(self, sentiment_data: dict, fear_greed_data: dict) -> dict:
        fg_value = fear_greed_data.get("current", {}).get("value", 50)
        sentiment_score = sentiment_data.get("sentiment_score", 0)

        # Score: positive = bullish, negative = bearish
        score = 0.0
        score += np.clip(sentiment_score * 5, -1, 1) * 0.4  # Reddit sentiment

        # Fear & Greed: fear = bearish, greed = bullish
        fg_signal = (fg_value - 50) / 50  # -1 at FG=0 (fear=bearish), +1 at FG=100 (greed=bullish)
        score += fg_signal * 0.6

        if score > 0.4:
            overall = "Strongly Bullish"
        elif score > 0.15:
            overall = "Bullish"
        elif score < -0.4:
            overall = "Strongly Bearish"
        elif score < -0.15:
            overall = "Bearish"
        else:
            overall = "Neutral"

        return {
            "overall_signal": overall,
            "social_sentiment": sentiment_data.get("classification", "Neutral"),
            "fear_greed": fear_greed_data.get("current", {}).get("classification", "Neutral"),
            "fear_greed_value": fg_value,
            "sentiment_score": sentiment_score,
            "bullish_ratio": sentiment_data.get("bullish_ratio", 0.5),
        }


def _norm_cdf(x):
    return 0.5 * (1 + np.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))
