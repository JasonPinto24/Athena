import joblib
import pandas as pd

from athena.tools.features import build_features, FEATURES, MARKET_FEATURES
from athena.tools.prices import download_prices, get_prices
from athena.universe import UNIVERSE

MODEL_PATH = "models/forecast_lgbm_v2.pkl"
_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def forecast_all(as_of, extra_ticker=None):
    """Score every stock in the race at once (features built only once per stock)."""
    bundle = load_model()
    tickers = list(UNIVERSE)
    if extra_ticker and extra_ticker not in tickers:
        if len(get_prices(extra_ticker, as_of)) == 0:
            download_prices(extra_ticker, start="2018-01-01")
        tickers.append(extra_ticker)

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
    day["M"] = 2 * (day["p"] - 0.5)
    return day.set_index("ticker")[["M", "p", "p_rank"]]


def forecast_score(ticker, as_of):
    table = forecast_all(as_of, extra_ticker=ticker)
    if ticker not in table.index:
        return {"M": 0.0, "note": "Not enough price history"}
    row = table.loc[ticker]
    return {
        "M": round(float(row["M"]), 3),
        "p_top": round(float(row["p"]), 3),
        "rank_among_peers": round(float(row["p_rank"]), 2),
    }