from athena.tools.prices import get_prices
from athena.tools.financials import get_financials


def clip(x, low=-1.0, high=1.0):
    return max(low, min(high, x))


def fundamentals_score(ticker, as_of, sector_pe=25.0, sector_margin=0.15, is_bank=False):
    fin = get_financials(ticker, as_of)
    if fin is None:
        return {"F": 0.0, "grades": {}, "note": "No financial data"}

    prices = get_prices(ticker, as_of)
    price = float(prices["close"].iloc[-1])

    grades = {}

    # 1. Cheap? (P/E)
    if fin.get("shares", 0) > 0:
        eps = fin["profit_ttm"] / fin["shares"]
        if eps <= 0:
            grades["cheap"] = -1.0
        else:
            pe = price / eps
            grades["cheap"] = clip((sector_pe - pe) / sector_pe)

    # 2. Safe? (Debt / Equity)
    if not is_bank and "total_debt" in fin and fin.get("equity", 0) > 0:
        de = fin["total_debt"] / fin["equity"]
        grades["safe"] = clip(1 - de)

    # 3. Growing? (Revenue growth)
    if "revenue_growth" in fin:
        grades["growing"] = clip(fin["revenue_growth"] / 0.20)

    # 4. Efficient? (ROE)
    if fin.get("equity", 0) > 0:
        roe = fin["profit_ttm"] / fin["equity"]
        grades["efficient"] = clip((roe - 0.10) / 0.10)

    # 5. Profitable? (Net margin)
    if fin["revenue_ttm"] > 0:
        margin = fin["profit_ttm"] / fin["revenue_ttm"]
        grades["profitable"] = clip((margin - sector_margin) / 0.10)

    if grades:
        F = sum(grades.values()) / len(grades)
    else:
        F = 0.0

    return {
        "F": round(F, 3),
        "grades": {name: round(g, 3) for name, g in grades.items()},
        "price": price,
    }