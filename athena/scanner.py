import pandas as pd
import yaml

from athena.agents.forecast import forecast_all
from athena.agents.fundamentals import fundamentals_score
from athena.agents.risk import risk_check
from athena.agents.sentiment import sentiment_score
from athena.tools.company import company_name
from athena.universe import UNIVERSE

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
W = CONFIG["weights"]


def scan(as_of, tickers=UNIVERSE, top_fraction=0.2):
    print("Forecasting all stocks at once...")
    forecasts = forecast_all(as_of)

    rows = []
    for t in tickers:
        if t not in forecasts.index:
            print(f"{t:16s} skipped (no price data)")
            continue
        name = company_name(t)
        fund = fundamentals_score(t, as_of)
        sent = sentiment_score(name, as_of, ticker=t)
        m = float(forecasts.loc[t, "M"])
        r = risk_check(t, as_of, decision="Buy")

        rows.append({
            "ticker": t, "company": name,
            "F": fund["F"] if fund.get("grades") else None,
            "S": sent["S"] if sent.get("has_data") else None,
            "M": m,
            "news_n": sent.get("n", 0),
            "veto": r["veto"], "size": r["position_size"],
        })
        print(f"{t:16s} F={rows[-1]['F']}  S={rows[-1]['S']}  M={m:+.2f}  news={sent.get('n', 0)}")

    df = pd.DataFrame(rows)

    # Missing data = neutral: z-score using only stocks that HAVE data
    for col in ["F", "S", "M"]:
        known = df[col].dropna()
        std = known.std()
        if len(known) > 1 and std > 0:
            df[col + "_z"] = ((df[col] - known.mean()) / std).fillna(0.0)
        else:
            df[col + "_z"] = 0.0

    total = W["fundamentals"] + W["sentiment"] + W["forecast"]
    df["score"] = (W["fundamentals"] * df["F_z"] + W["sentiment"] * df["S_z"]
                   + W["forecast"] * df["M_z"]) / total

    top_n = max(1, int(len(df) * top_fraction))
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["decision"] = "Hold"
    df.loc[df["rank"] <= top_n, "decision"] = "Buy"
    df.loc[df["rank"] > len(df) - top_n, "decision"] = "Sell"
    df.loc[(df["decision"] == "Buy") & df["veto"], "decision"] = "Hold (risk veto)"
    df.loc[df["decision"] != "Buy", "size"] = 0.0
    return df