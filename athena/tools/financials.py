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


def _currencies(stock):
    try:
        info = stock.info
        return info.get("financialCurrency"), info.get("currency")
    except Exception:
        return None, None


def _fx_rate(from_currency, to_currency, as_of):
    if not from_currency or not to_currency or from_currency == to_currency:
        return 1.0
    pair = f"{from_currency}{to_currency}=X"
    fx = yf.download(pair, start=as_of - pd.Timedelta(days=10), end=as_of + pd.Timedelta(days=1),
                     auto_adjust=True, multi_level_index=False, progress=False)
    if len(fx) == 0:
        return 1.0
    return float(fx["Close"].iloc[-1])


def get_financials(ticker, as_of, lag_days=45):
    as_of = pd.Timestamp(as_of)
    stock = yf.Ticker(ticker)
    income = stock.quarterly_income_stmt
    balance = stock.quarterly_balance_sheet

    quarters = _usable_dates(income, as_of, lag_days)
    if len(quarters) < 4:
        return None

    # Convert financial figures into the same currency as the share price
    fin_currency, price_currency = _currencies(stock)
    rate = _fx_rate(fin_currency, price_currency, as_of)

    revenue = _row(income, ["Total Revenue"])
    profit = _row(income, ["Net Income"])
    last4 = quarters[:4]

    result = {
        "latest_quarter": str(quarters[0].date()),
        "revenue_ttm": float(revenue[last4].sum()) * rate,
        "profit_ttm": float(profit[last4].sum()) * rate,
        "currency": price_currency,
        "fx_rate_used": round(rate, 4),
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
            result["total_debt"] = float(debt[latest]) * rate
        if equity is not None:
            result["equity"] = float(equity[latest]) * rate
        if shares is not None:
            result["shares"] = float(shares[latest])

    return result