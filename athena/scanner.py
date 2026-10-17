import pandas as pd

from athena.agents.forecast import forecast_all
from athena.agents.fundamentals import fundamentals_score
from athena.agents.risk import risk_check
from athena.agents.sentiment import sentiment_score
from athena.tools.company import company_name
from athena.universe import UNIVERSE

BUY_FRACTION = 0.25           # top 25% of the forecast ranking -> Buy
SELL_FRACTION = 0.25          # bottom 25% -> Sell
CLOSE_CALL = 0.05 + 1e-9      # within 5 percentile points of a cut-off (+ a hair for decimal noise)


def label_from_rank(pct, buy_fraction=BUY_FRACTION, sell_fraction=SELL_FRACTION):
    """pct = share of peers this stock beats or ties (1.0 = best)."""
    if pct >= 1 - buy_fraction:
        return "Buy"
    if pct <= sell_fraction:
        return "Sell"
    return "Hold"


def is_close_call(pct, buy_fraction=BUY_FRACTION, sell_fraction=SELL_FRACTION):
    near_buy = abs(pct - (1 - buy_fraction)) <= CLOSE_CALL
    near_sell = abs(pct - sell_fraction) <= CLOSE_CALL
    return near_buy or near_sell


def scan(as_of, tickers=UNIVERSE, with_news=False):
    # 1. One forecast for all stocks on this date (same ranking as the single-stock view)
    fc = forecast_all(as_of)
    fc = fc[fc.index.isin(tickers)].copy()
    fc["pct"] = fc["p_rank"]
    fc["rank"] = fc["p"].rank(ascending=False, method="min").astype(int)
    fc = fc.sort_values("p", ascending=False)

    n = len(fc)
    rows = []
    for i, (t, r) in enumerate(fc.iterrows(), start=1):
        print(f"[{i}/{n}] {t}")

        # 2. Signal from the ranking
        signal = label_from_rank(r["pct"])

        # 3. Risk brake (only blocks Buys)
        risk = risk_check(t, as_of, decision="Buy")
        decision = signal
        if signal == "Buy" and risk["veto"]:
            decision = "Hold (risk veto)"
        reasons = risk.get("reasons") or risk.get("veto_reasons") or []

        # 4. Context: fundamentals (fast) and news (slow, optional)
        fund = fundamentals_score(t, as_of)
        F = fund["F"] if fund.get("grades") else None

        S, news_n = None, 0
        if with_news:
            sent = sentiment_score(company_name(t), as_of, ticker=t)
            news_n = sent.get("n", 0)
            if sent.get("has_data"):
                S = sent["S"]

        rows.append({
            "ticker": t,
            "rank": int(r["rank"]),
            "p_top": round(float(r["p"]), 3),
            "signal": signal,
            "decision": decision,
            "close_call": is_close_call(r["pct"]),
            "F": None if F is None else round(F, 2),
            "S": None if S is None else round(S, 2),
            "news_n": news_n,
            "vol": risk.get("volatility"),
            "veto": risk["veto"],
            "veto_reasons": "; ".join(reasons),
            "position_size": risk.get("position_size", 0) if decision == "Buy" else 0,
        })

    return pd.DataFrame(rows)