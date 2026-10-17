import time
import duckdb
import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COLUMNS = ["date", "title", "source", "url"]
DB_PATH = "data/athena.duckdb"


def _read_cache(company_name, as_of):
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            company VARCHAR, as_of VARCHAR, date TIMESTAMP,
            title VARCHAR, source VARCHAR, url VARCHAR
        )
    """)
    cached = con.execute(
        "SELECT date, title, source, url FROM news_cache WHERE company = ? AND as_of = ?",
        [company_name, str(as_of)],
    ).df()
    con.close()
    return cached


def _save_cache(company_name, as_of, news):
    to_save = news.copy()
    to_save.insert(0, "as_of", str(as_of))
    to_save.insert(0, "company", company_name)
    con = duckdb.connect(DB_PATH)
    con.execute("INSERT INTO news_cache SELECT * FROM to_save")
    con.close()


def _fetch_gdelt(company_name, as_of, days, max_records, retries):
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

    for attempt in range(retries):
        try:
            response = requests.get(GDELT_URL, params=params, timeout=30)
            if response.status_code == 429:
                print(f"GDELT rate limit, waiting 10s (attempt {attempt + 1}/{retries})")
                time.sleep(10)
                continue
            return response.json().get("articles", [])
        except Exception as error:
            print(f"GDELT error: {error}, waiting 10s (attempt {attempt + 1}/{retries})")
            time.sleep(10)

    print("GDELT: gave up after all attempts")
    return []


def get_news(company_name, as_of, days=30, max_records=100, retries=5):
    cached = _read_cache(company_name, as_of)
    if len(cached) > 0:
        return cached

    articles = _fetch_gdelt(company_name, as_of, days, max_records, retries)

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

    if len(news) > 0:
        _save_cache(company_name, as_of, news)
    return news