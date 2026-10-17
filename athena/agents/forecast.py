import joblib
import pandas as pd

from athena.tools.features import build_features, FEATURES, MARKET_FEATURES
from athena.tools.prices import download_prices, get_prices

MODEL_PATH = "models/forecast_lgbm_v2.pkl"
UNIVERSE = [
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS",
    "NESTLEIND.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "MARUTI.NS", "M&M.NS", "TATASTEEL.NS", "LT.NS",
]
_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def forecast_score(ticker, as_of):
    bundle = load_model()

    if len(get_prices(ticker, as_of)) == 0:
        download_prices(ticker, start="2018-01-01")

    tickers = UNIVERSE if ticker in UNIVERSE else UNIVERSE + [ticker]

    latest_rows = []
    for t in tickers:
        df = build_features(t, as_of).dropna(subset=FEATURES + MARKET_FEATURES)
        if len(df) > 0:
            latest_rows.append(df.iloc[[-1]])
    day = pd.concat(latest_rows)

    for f in FEATURES:
        day[f + "_rank"] = day[f].rank(pct=True)

    day["p"] = bundle["model"].predict_proba(day[bundle["features"]])[:, 1]
    day["p_rank"] = day["p"].rank(pct=True)

    row = day[day["ticker"] == ticker]
    if len(row) == 0:
        return {"M": 0.0, "note": "Not enough price history"}

    p = float(row["p"].iloc[0])
    return {
        "M": round(2 * (p - 0.5), 3),
        "p_top": round(p, 3),
        "rank_among_peers": round(float(row["p_rank"].iloc[0]), 2),
        "date_used": str(row.index[0].date()),
    }