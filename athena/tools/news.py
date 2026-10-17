import re
import time

import duckdb
import pandas as pd
import requests

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COLUMNS = ["date", "title", "source", "url"]
DB_PATH = "data/athena.duckdb"
MIN_GAP_SECONDS = 5        # GDELT rule: at most 1 request every 5 seconds
REMEMBER_SECONDS = 600     # remember empty/failed searches for 10 minutes

_session = {}              # (key, as_of) -> (time, news table) for empty or failed searches
_last_call = 0.0


def _read_cache(key, as_of):
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS news_cache (
            company VARCHAR, as_of VARCHAR, date TIMESTAMP,
            title VARCHAR, source VARCHAR, url VARCHAR
        )
    """)
    cached = con.execute(
        "SELECT date, title, source, url FROM news_cache WHERE company = ? AND as_of = ?",
        [key, str(as_of)],
    ).df()
    con.close()
    return cached


def _save_cache(key, as_of, news):
    to_save = news.copy()
    to_save.insert(0, "as_of", str(as_of))
    to_save.insert(0, "company", key)
    con = duckdb.connect(DB_PATH)
    con.execute("INSERT INTO news_cache SELECT * FROM to_save")
    con.close()


def _build_query(names):
    terms = []
    for name in names:
        term = " ".join(re.sub(r"[^\w\s]", " ", name).split())
        if len(term.replace(" ", "")) >= 3:
            terms.append(f'"{term}"')
    if not terms:
        terms = [f'"{names[0]}"']
    joined = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
    return f"{joined} sourcelang:english"


def _wait_your_turn():
    """Keep at least MIN_GAP_SECONDS between two GDELT requests."""
    global _last_call
    gap = time.time() - _last_call
    if gap < MIN_GAP_SECONDS:
        time.sleep(MIN_GAP_SECONDS - gap)
    _last_call = time.time()


def _fetch_gdelt(names, as_of, days, max_records, retries):
    """Returns a list of articles, or None if GDELT kept refusing."""
    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    now = pd.Timestamp.utcnow().tz_localize(None)
    end = min(end, now)
    start = end - pd.Timedelta(days=days)

    params = {
        "query": _build_query(names),
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }

    for attempt in range(retries):
        wait = 10 * (attempt + 1)            # backoff: 10s, 20s, 30s
        _wait_your_turn()
        try:
            response = requests.get(GDELT_URL, params=params, timeout=30)
            if response.status_code == 429:
                print(f"GDELT rate limit, waiting {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            try:
                return response.json().get("articles", [])
            except ValueError:               # plain-text reply = GDELT asking us to slow down
                print(f"GDELT sent a non-JSON reply, waiting {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
        except Exception as error:
            print(f"GDELT error: {error}, waiting {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)

    print("GDELT: gave up after all attempts (will not retry this search for 10 minutes)")
    return None


def get_news(names, as_of, days=30, max_records=100, retries=3):
    if isinstance(names, str):
        names = [names]
    key = "|".join(names)
    memo_key = (key, str(as_of))

    # 1. Saved headlines from an earlier run
    cached = _read_cache(key, as_of)
    if len(cached) > 0:
        return cached

    # 2. Empty or failed search in the last 10 minutes -> don't ask again
    if memo_key in _session:
        saved_at, news = _session[memo_key]
        if time.time() - saved_at < REMEMBER_SECONDS:
            return news.copy()

    # 3. Ask GDELT
    articles = _fetch_gdelt(names, as_of, days, max_records, retries)

    rows = []
    for article in articles or []:
        rows.append({
            "date": pd.to_datetime(article["seendate"], format="%Y%m%dT%H%M%SZ"),
            "title": article["title"],
            "source": article.get("domain", ""),
            "url": article["url"],
        })

    news = pd.DataFrame(rows, columns=COLUMNS)
    news["title"] = news["title"].str.split(" | ", regex=False).str[0].str.strip()
    news = news.drop_duplicates(subset="title")

    if len(news) > 0:
        _save_cache(key, as_of, news)            # real headlines: save forever
    else:
        _session[memo_key] = (time.time(), news)  # empty or failed: remember for 10 minutes
    return news
