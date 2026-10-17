import pandas as pd
import yfinance as yf


def _row(table, names):
    for name in names:
        if name in table.index:
            return table.loc[name]
    return None


def _usable_dates(table, as_of, lag_days):
    dates = [d for d in table.columns if d + pd.Timedelta(days=lag_days) <= as_of]
    return sorted(dates, reverse=True)


def get_financials(ticker, as_of, lag_days=45):
    as_of = pd.Timestamp(as_of)
    stock = yf.Ticker(ticker)
    income = stock.quarterly_income_stmt
    balance = stock.quarterly_balance_sheet

    quarters = _usable_dates(income, as_of, lag_days)
    if len(quarters) < 4:
        return None

    revenue = _row(income, ["Total Revenue"])
    profit = _row(income, ["Net Income"])
    last4 = quarters[:4]

    result = {
        "latest_quarter": str(quarters[0].date()),
        "revenue_ttm": float(revenue[last4].sum()),
        "profit_ttm": float(profit[last4].sum()),
    }

    if len(quarters) >= 5:
        result["revenue_growth"] = float(revenue[quarters[0]] / revenue[quarters[4]] - 1)

    bal_dates = _usable_dates(balance, as_of, lag_days)
    if bal_dates:
        latest = bal_dates[0]
        debt = _row(balance, ["Total Debt"])
        equity = _row(balance, ["Stockholders Equity", "Common Stock Equity"])
        shares = _row(balance, ["Ordinary Shares Number", "Share Issued"])
        if debt is not None:
            result["total_debt"] = float(debt[latest])
        if equity is not None:
            result["equity"] = float(equity[latest])
        if shares is not None:
            result["shares"] = float(shares[latest])

    return result