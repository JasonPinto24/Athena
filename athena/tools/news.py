import time
import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COLUMNS = ["date", "title", "source", "url"]


def get_news(company_name, as_of, days=30, max_records=100, retries=3):
    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    now = pd.Timestamp.utcnow().tz_localize(None)
    end = min(end, now)
    start = end - pd.Timedelta(days=days)

    params = {
        "query": f'"{company_name}" sourcelang:english',
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    articles = []
    for attempt in range(retries):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(6)
                continue
            articles = response.json().get("articles", [])
            break
        except Exception:
            time.sleep(6)

    rows = []
    for article in articles:
        rows.append({
            "date": pd.to_datetime(article["seendate"], format="%Y%m%dT%H%M%SZ"),
            "title": article["title"],
            "source": article.get("domain", ""),
            "url": article["url"],
        })

    news = pd.DataFrame(rows, columns=COLUMNS)
    news = news.drop_duplicates(subset="title")
    return news