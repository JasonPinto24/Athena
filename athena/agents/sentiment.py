import pandas as pd
from transformers import pipeline

from athena.tools.news import get_news

_finbert = None


def load_finbert():
    global _finbert
    if _finbert is None:
        _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    return _finbert


def score_headlines(titles):
    model = load_finbert()
    results = model(list(titles), truncation=True)
    scores = []
    for result in results:
        probs = {item["label"].lower(): item["score"] for item in result}
        scores.append(probs.get("positive", 0) - probs.get("negative", 0))
    return scores


def sentiment_score(company_name, as_of, half_life_days=3, shrink=5):
    news = get_news(company_name, as_of)
    news = news[news["title"].str.contains(company_name, case=False, na=False)]
    n = len(news)
    if n == 0:
        return {"S": 0.0, "n": 0, "note": "No news found"}

    news = news.copy()
    news["score"] = score_headlines(news["title"])

    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    age_days = (end - news["date"]).dt.total_seconds() / 86400
    news["weight"] = 0.5 ** (age_days.clip(lower=0) / half_life_days)

    raw = (news["score"] * news["weight"]).sum() / news["weight"].sum()
    S = raw * n / (n + shrink)

    news = news.sort_values("score")
    return {
        "S": round(float(S), 3),
        "raw": round(float(raw), 3),
        "n": n,
        "most_negative": news.iloc[0]["title"],
        "most_positive": news.iloc[-1]["title"],
    }