import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from athena.tools.prices import get_prices

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
RISK = CONFIG["risk"]


def next_earnings_date(ticker):
    try:
        calendar = yf.Ticker(ticker).calendar
        dates = calendar.get("Earnings Date", [])
        return pd.Timestamp(dates[0]) if dates else None
    except Exception:
        return None


def risk_check(ticker, as_of, decision="Buy", index_ticker="^NSEI"):
    prices = get_prices(ticker, as_of).set_index("date")["close"]
    index = get_prices(index_ticker, as_of).set_index("date")["close"]

    if len(prices) < 252:
        return {"veto": decision == "Buy", "reasons": ["Less than 1 year of price history"],
                "position_size": 0.0}

    # 1. Volatility: spread of the last 60 daily returns, converted to yearly
    returns = prices.pct_change().dropna()
    volatility = returns.tail(60).std() * np.sqrt(252)

    # 2. Max drawdown: worst fall from a peak over the last year
    last_year = prices.tail(252)
    drawdown = last_year / last_year.cummax() - 1
    max_drawdown = drawdown.min()

    # 3. Beta: how much the stock moves with the index (last year)
    both = pd.concat([returns, index.pct_change()], axis=1, join="inner").dropna().tail(252)
    both.columns = ["stock", "index"]
    beta = both["stock"].cov(both["index"]) / both["index"].var()

    # Veto checks
    reasons = []
    if volatility > RISK["max_volatility"]:
        reasons.append(f"Volatility {volatility:.0%} is above the {RISK['max_volatility']:.0%} limit")
    if max_drawdown < RISK["max_drawdown"]:
        reasons.append(f"Fell {max_drawdown:.0%} from its peak (limit {RISK['max_drawdown']:.0%})")

    as_of_date = pd.Timestamp(as_of)
    if abs((pd.Timestamp.today().normalize() - as_of_date).days) <= 3:
        earnings = next_earnings_date(ticker)
        if earnings is not None and 0 <= (earnings - as_of_date).days <= 7:
            reasons.append(f"Earnings announcement on {earnings.date()} (event risk)")

    veto = decision == "Buy" and len(reasons) > 0

    # Position size: riskier stock = smaller position, never above the max
    size = min(RISK["max_weight"], RISK["max_weight"] * RISK["target_volatility"] / volatility)

    return {
        "veto": veto,
        "reasons": reasons,
        "position_size": 0.0 if veto else round(float(size), 4),
        "volatility": round(float(volatility), 3),
        "max_drawdown": round(float(max_drawdown), 3),
        "beta": round(float(beta), 2),
    }