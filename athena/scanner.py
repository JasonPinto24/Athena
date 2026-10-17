import time

import pandas as pd
import yaml

from athena.agents.forecast import UNIVERSE, forecast_score
from athena.agents.fundamentals import fundamentals_score
from athena.agents.risk import risk_check
from athena.agents.sentiment import sentiment_score
from athena.tools.company import company_name

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
W = CONFIG["weights"]


def scan(as_of, tickers=UNIVERSE, top_n=4):
    rows = []
    for t in tickers:
        name = company_name(t)
        f = fundamentals_score(t, as_of)["F"]
        s = sentiment_score(name, as_of)["S"]
        m = forecast_score(t, as_of)["M"]
        r = risk_check(t, as_of, decision="Buy")
        rows.append({"ticker": t, "company": name, "F": f, "S": s, "M": m,
                     "veto": r["veto"], "size": r["position_size"]})
        print(f"{t:14s} F={f:+.2f}  S={s:+.2f}  M={m:+.2f}  veto={r['veto']}")
        time.sleep(5)  # be polite to GDELT's 1-request-per-5-seconds limit

    df = pd.DataFrame(rows)

    # Put every agent on the same scale: z-score across today's stocks
    for col in ["F", "S", "M"]:
        std = df[col].std()
        df[col + "_z"] = (df[col] - df[col].mean()) / std if std > 0 else 0.0

    # Combine with the config weights, re-scaled because the debate isn't used here
    total = W["fundamentals"] + W["sentiment"] + W["forecast"]
    df["score"] = (W["fundamentals"] * df["F_z"] + W["sentiment"] * df["S_z"]
                   + W["forecast"] * df["M_z"]) / total

    # Rank: top N = Buy, bottom N = Sell (avoid), the rest Hold
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["decision"] = "Hold"
    df.loc[df["rank"] <= top_n, "decision"] = "Buy"
    df.loc[df["rank"] > len(df) - top_n, "decision"] = "Sell"
    df.loc[(df["decision"] == "Buy") & df["veto"], "decision"] = "Hold (risk veto)"
    df.loc[df["decision"] != "Buy", "size"] = 0.0
    return df