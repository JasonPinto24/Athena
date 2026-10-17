import os

import pandas as pd
from transformers import pipeline

from athena.tools.company import news_aliases, title_matches
from athena.tools.news import get_news

LOCAL_MODEL = "models/finbert-athena"
FALLBACK_MODEL = "ProsusAI/finbert"
_finbert = None


def load_finbert():
    global _finbert
    if _finbert is None:
        model_name = LOCAL_MODEL if os.path.exists(LOCAL_MODEL) else FALLBACK_MODEL
        print("Sentiment model:", model_name)
        _finbert = pipeline("text-classification", model=model_name, top_k=None)
    return _finbert


def score_headlines(titles):
    model = load_finbert()
    results = model(list(titles), truncation=True)
    scores = []
    for result in results:
        probs = {item["label"].lower(): item["score"] for item in result}
        scores.append(probs.get("positive", 0) - probs.get("negative", 0))
    return scores


def sentiment_score(company_name, as_of, ticker=None, half_life_days=3, shrink=5):
    aliases = news_aliases(ticker) if ticker else [company_name]
    news = get_news(aliases, as_of)
    news = news[news["title"].apply(lambda t: title_matches(t, aliases))]
    n = len(news)
    if n == 0:
        return {"S": 0.0, "n": 0, "has_data": False, "note": "No news found"}

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
        "has_data": True,
        "most_negative": news.iloc[0]["title"],
        "most_positive": news.iloc[-1]["title"],
    }