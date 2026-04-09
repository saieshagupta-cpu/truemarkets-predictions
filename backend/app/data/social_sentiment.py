import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REDDIT_SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.json"
SUBREDDITS = ["Bitcoin", "CryptoCurrency"]
analyzer = SentimentIntensityAnalyzer()


async def fetch_social_sentiment(query: str = "bitcoin", subreddits: list[str] | None = None) -> dict:
    all_scores = []
    post_count = 0
    bullish_count = 0
    subs = subreddits or SUBREDDITS

    async with httpx.AsyncClient(
        headers={"User-Agent": "TrueMarkets-Prediction-Bot/1.0"}, timeout=15, follow_redirects=True
    ) as client:
        for sub in subs:
            try:
                resp = await client.get(
                    REDDIT_SEARCH_URL.format(subreddit=sub),
                    params={"q": query, "sort": "new", "limit": 25, "t": "week", "restrict_sr": "on"},
                )
                if resp.status_code != 200:
                    continue
                posts = resp.json().get("data", {}).get("children", [])
                for post in posts:
                    title = post["data"].get("title", "")
                    selftext = post["data"].get("selftext", "")[:200]
                    text = f"{title} {selftext}"
                    scores = analyzer.polarity_scores(text)
                    compound = scores["compound"]
                    all_scores.append(compound)
                    post_count += 1
                    if compound > 0.05:
                        bullish_count += 1
            except Exception:
                continue

    if not all_scores:
        return {
            "sentiment_score": 0.0,
            "post_volume": 0,
            "bullish_ratio": 0.5,
            "classification": "Neutral",
        }

    avg_sentiment = sum(all_scores) / len(all_scores)
    bullish_ratio = bullish_count / max(post_count, 1)

    if avg_sentiment > 0.2:
        classification = "Very Bullish"
    elif avg_sentiment > 0.05:
        classification = "Bullish"
    elif avg_sentiment > -0.05:
        classification = "Neutral"
    elif avg_sentiment > -0.2:
        classification = "Bearish"
    else:
        classification = "Very Bearish"

    return {
        "sentiment_score": round(avg_sentiment, 4),
        "post_volume": post_count,
        "bullish_ratio": round(bullish_ratio, 4),
        "classification": classification,
    }
