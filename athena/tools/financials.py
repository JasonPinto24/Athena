import pandas as pd
import yfinance as yf
from joblib import Memory

memory = Memory("data/cache", verbose=0)


def _row(table, names):
    for name in names:
        if name in table.index:
            return table.loc[name]
    return None


def _usable_dates(table, as_of, lag_days):
    dates = [d for d in table.columns if d + pd.Timedelta(days=lag_days) <= as_of]
    return sorted(dates, reverse=True)


def _info(stock):
    try:
        return stock.info
    except Exception:
        return {}


def _fx_rate(from_currency, to_currency, as_of):
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0
    pair = f"{from_currency}{to_currency}=X"
    fx = yf.download(pair, start=as_of - pd.Timedelta(days=10), end=as_of + pd.Timedelta(days=1),
                     auto_adjust=True, multi_level_index=False, progress=False)
    if len(fx) == 0:
        return 1.0
    return float(fx["Close"].iloc[-1])


@memory.cache
def get_financials(ticker, as_of, lag_days=45, annual_lag_days=60):
    as_of = pd.Timestamp(as_of)
    stock = yf.Ticker(ticker)
    info = _info(stock)

    income_q = stock.quarterly_income_stmt
    quarters = _usable_dates(income_q, as_of, lag_days)

    revenue_q = _row(income_q, ["Total Revenue"])
    profit_q = _row(income_q, ["Net Income"])

    if len(quarters) >= 4 and revenue_q is not None and profit_q is not None:
        # Normal case: add up the last 4 public quarters (TTM)
        source = "quarterly"
        last4 = quarters[:4]
        revenue_ttm = float(revenue_q[last4].sum())
        profit_ttm = float(profit_q[last4].sum())
        latest_period = quarters[0]
        growth = None
        if len(quarters) >= 5:
            growth = float(revenue_q[quarters[0]] / revenue_q[quarters[4]] - 1)
        balance = stock.quarterly_balance_sheet
        bal_dates = _usable_dates(balance, as_of, lag_days)
    else:
        # Fallback: use the latest public ANNUAL report (already covers 12 months)
        income_a = stock.income_stmt
        years = _usable_dates(income_a, as_of, annual_lag_days)
        revenue_a = _row(income_a, ["Total Revenue"])
        profit_a = _row(income_a, ["Net Income"])
        if not years or revenue_a is None or profit_a is None:
            return None
        source = "annual"
        revenue_ttm = float(revenue_a[years[0]])
        profit_ttm = float(profit_a[years[0]])
        latest_period = years[0]
        growth = None
        if len(years) >= 2 and revenue_a[years[1]]:
            growth = float(revenue_a[years[0]] / revenue_a[years[1]] - 1)
        balance = stock.balance_sheet
        bal_dates = _usable_dates(balance, as_of, annual_lag_days)

    # Convert financial figures into the same currency as the share price
    rate = _fx_rate(info.get("financialCurrency"), info.get("currency"), as_of)

    result = {
        "source": source,
        "latest_period": str(latest_period.date()),
        "revenue_ttm": revenue_ttm * rate,
        "profit_ttm": profit_ttm * rate,
        "currency": info.get("currency"),
        "fx_rate_used": round(rate, 4),
        "sector": info.get("sector"),
        "is_financial": info.get("sector") == "Financial Services",
    }
    if growth is not None:
        result["revenue_growth"] = growth

    if bal_dates:
        latest = bal_dates[0]
        debt = _row(balance, ["Total Debt"])
        equity = _row(balance, ["Stockholders Equity", "Common Stock Equity"])
        shares = _row(balance, ["Ordinary Shares Number", "Share Issued"])
        if debt is not None:
            result["total_debt"] = float(debt[latest]) * rate
        if equity is not None:
            result["equity"] = float(equity[latest]) * rate
        if shares is not None:
            result["shares"] = float(shares[latest])

    return result