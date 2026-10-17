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


@memory.cache
def _statements(ticker):
    """Download a company's statements ONCE; every past date is then computed locally."""
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
    except Exception:
        info = {}
    return {
        "info": {k: info.get(k) for k in ["financialCurrency", "currency", "sector"]},
        "income_q": stock.quarterly_income_stmt,
        "balance_q": stock.quarterly_balance_sheet,
        "income_a": stock.income_stmt,
        "balance_a": stock.balance_sheet,
    }


@memory.cache
def _fx_history(pair):
    """Download a full exchange-rate history ONCE (e.g. USDINR=X)."""
    fx = yf.download(pair, start="2017-01-01", auto_adjust=True,
                     multi_level_index=False, progress=False)
    return fx["Close"] if len(fx) else pd.Series(dtype=float)


def _fx_rate(from_currency, to_currency, as_of):
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0
    series = _fx_history(f"{from_currency}{to_currency}=X")
    series = series[series.index <= as_of]
    return float(series.iloc[-1]) if len(series) else 1.0


def get_financials(ticker, as_of, lag_days=45, annual_lag_days=60):
    as_of = pd.Timestamp(as_of)
    s = _statements(ticker)
    info = s["info"]

    income_q = s["income_q"]
    quarters = _usable_dates(income_q, as_of, lag_days)
    revenue_q = _row(income_q, ["Total Revenue"])
    profit_q = _row(income_q, ["Net Income"])

    if len(quarters) >= 4 and revenue_q is not None and profit_q is not None:
        source = "quarterly"
        last4 = quarters[:4]
        revenue_ttm = float(revenue_q[last4].sum())
        profit_ttm = float(profit_q[last4].sum())
        latest_period = quarters[0]
        growth = None
        if len(quarters) >= 5:
            growth = float(revenue_q[quarters[0]] / revenue_q[quarters[4]] - 1)
        balance = s["balance_q"]
        bal_dates = _usable_dates(balance, as_of, lag_days)
    else:
        income_a = s["income_a"]
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
        balance = s["balance_a"]
        bal_dates = _usable_dates(balance, as_of, annual_lag_days)

    if pd.isna(revenue_ttm) or pd.isna(profit_ttm):
        return None

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
    if growth is not None and not pd.isna(growth):
        result["revenue_growth"] = growth

    if bal_dates:
        latest = bal_dates[0]
        debt = _row(balance, ["Total Debt"])
        equity = _row(balance, ["Stockholders Equity", "Common Stock Equity"])
        shares = _row(balance, ["Ordinary Shares Number", "Share Issued"])
        if debt is not None and not pd.isna(debt[latest]):
            result["total_debt"] = float(debt[latest]) * rate
        if equity is not None and not pd.isna(equity[latest]):
            result["equity"] = float(equity[latest]) * rate
        if shares is not None and not pd.isna(shares[latest]):
            result["shares"] = float(shares[latest])

    return result