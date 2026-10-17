import pandas as pd

from athena.agents.fundamentals import fundamentals_score
from athena.agents.risk import risk_check
from athena.agents.sentiment import sentiment_score
from athena.tools.company import company_name

START, END = "2023-01-01", "2025-12-31"   # 2026 stays hidden for the final exam
FETCH_NEWS = False                         # True = also fetch historical news (slow: hours)

# Walk-forward forecasts: each was made by a model trained ONLY on earlier data
wf = pd.read_csv("data/forecast_walkforward_v2.csv", parse_dates=["date"])
wf = wf[(wf["date"] >= START) & (wf["date"] <= END)]

# First trading day of each month
month_starts = wf.groupby(wf["date"].dt.to_period("M"))["date"].min().tolist()
print("Monthly dates:", len(month_starts))

rows = []
for d in month_starts:
    as_of = str(d.date())
    day = wf[wf["date"] == d]
    for _, w in day.iterrows():
        t = w["ticker"]
        fund = fundamentals_score(t, as_of)
        risk = risk_check(t, as_of, decision="Buy")

        S, S_has = 0.0, False
        if FETCH_NEWS:
            sent = sentiment_score(company_name(t), as_of, ticker=t)
            S, S_has = sent["S"], sent.get("has_data", False)

        rows.append({
            "date": as_of,
            "ticker": t,
            "F": fund["F"] if fund.get("grades") else None,
            "is_financial": fund.get("is_financial"),
            "M": w["p"],
            "S": S if S_has else None,
            "vol": risk.get("volatility"),
            "mdd": risk.get("max_drawdown"),
            "beta": risk.get("beta"),
            "veto": risk["veto"],
            "future_excess": w["future_excess"],
            "target": w["target"],
        })
    print(f"{as_of}: {len(day)} stocks done")

history = pd.DataFrame(rows)
history.to_csv("data/history.csv", index=False)

print("\nSaved data/history.csv:", len(history), "rows")
print("Missing F:", history["F"].isna().mean().round(2),
      "| Missing S:", history["S"].isna().mean().round(2))
print(history.head())