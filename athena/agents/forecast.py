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
        try:
            if len(get_prices(extra_ticker, as_of)) == 0:
                download_prices(extra_ticker, start="2018-01-01")
            tickers.append(extra_ticker)
        except Exception as e:
            print(f"Could not get prices for {extra_ticker}: {e}")

    latest_rows = []
    for t in tickers:
        try:
            df = build_features(t, as_of).dropna(subset=FEATURES + MARKET_FEATURES)
        except Exception as e:
            print(f"Skipping {t}: {e}")
            continue
        if len(df) > 0:
            latest_rows.append(df.iloc[[-1]])
    day = pd.concat(latest_rows)

    for f in FEATURES:
        day[f + "_rank"] = day[f].rank(pct=True)

    day["p"] = bundle["model"].predict_proba(day[bundle["features"]])[:, 1]
    day["M"] = 2 * (day["p"] - 0.5)

    # Rank ONLY against the Nifty 50 peers, so an outside stock never shifts anyone's rank.
    peers_p = day.loc[day["ticker"].isin(UNIVERSE), "p"]
    day["p_rank"] = day["p"].apply(lambda v: (peers_p <= v).mean())   # no rounding here
    day["n_peers"] = len(peers_p)
    day["in_universe"] = day["ticker"].isin(UNIVERSE)
    return day.set_index("ticker")[["M", "p", "p_rank", "n_peers", "in_universe"]]


def forecast_score(ticker, as_of):
    table = forecast_all(as_of, extra_ticker=ticker)
    if ticker not in table.index:
        # Bad ticker or not enough history: neutral answer so nothing downstream crashes
        return {"M": 0.0, "p_top": 0.5, "rank_among_peers": 0.5, "n_peers": 0,
                "in_universe": ticker in UNIVERSE, "note": "No data or not enough price history"}
    row = table.loc[ticker]
    return {
        "M": float(row["M"]),
        "p_top": float(row["p"]),
        "rank_among_peers": float(row["p_rank"]),   # full precision; round only when printing
        "n_peers": int(row["n_peers"]),
        "in_universe": bool(row["in_universe"]),
    }