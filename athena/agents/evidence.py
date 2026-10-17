import pandas as pd
from pydantic import BaseModel

from athena.tools.company import news_aliases, title_matches
from athena.tools.financials import get_financials
from athena.tools.news import get_news
from athena.tools.prices import get_prices


class Evidence(BaseModel):
    id: str
    category: str
    text: str


class EvidencePack(BaseModel):
    ticker: str
    company: str
    as_of: str
    items: list[Evidence]

    def ids(self):
        return {item.id for item in self.items}

    def to_text(self):
        lines = [f"Evidence for {self.company} as of {self.as_of}:"]
        for item in self.items:
            lines.append(f"[{item.id}] ({item.category}) {item.text}")
        return "\n".join(lines)


def _money(value, ticker):
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return f"Rs {value / 1e7:,.0f} crore"
    return f"${value / 1e9:,.2f} billion"


def _relevant_news(ticker, as_of, max_news):
    """Headlines that really mention the company, newest first. Never crashes on 'no news'."""
    aliases = news_aliases(ticker)
    news = get_news(aliases, as_of)
    if news is None or len(news) == 0 or "title" not in news.columns:
        return pd.DataFrame(columns=["date", "title"]), aliases

    mask = news["title"].apply(lambda t: title_matches(t, aliases)).astype(bool)
    news = news.loc[mask]                       # .loc + True/False -> filters rows, never columns
    if "date" in news.columns:
        news = news.sort_values("date", ascending=False)
    return news.head(max_news), aliases


def build_evidence_pack(ticker, company_name, as_of, agent_results=None,
                        index_ticker="^NSEI", anonymize=False, max_news=8):
    name = "Company A" if anonymize else company_name
    items = []

    # --- P: price facts ---
    prices = get_prices(ticker, as_of).set_index("date")["close"]
    index = get_prices(index_ticker, as_of).set_index("date")["close"]
    if len(prices) > 252:
        last = prices.iloc[-1]
        ret_1m = prices.iloc[-1] / prices.iloc[-21] - 1
        ret_3m = prices.iloc[-1] / prices.iloc[-63] - 1
        high_1y = prices.tail(252).max()
        idx_1m = index.iloc[-1] / index.iloc[-21] - 1
        price_facts = [
            f"Latest closing price: {last:,.2f}",
            f"Return over the last month: {ret_1m:+.1%}",
            f"Return over the last 3 months: {ret_3m:+.1%}",
            f"Price is {last / high_1y - 1:+.1%} versus its 1-year high",
            f"Last-month return versus the market index: {ret_1m - idx_1m:+.1%}",
        ]
        for i, text in enumerate(price_facts, start=1):
            items.append(Evidence(id=f"P{i}", category="price", text=text))

    # --- F: financial facts ---
    fin = get_financials(ticker, as_of)
    if fin:
        fin_facts = [
            f"Revenue over the last 12 months: {_money(fin['revenue_ttm'], ticker)}",
            f"Net profit over the last 12 months: {_money(fin['profit_ttm'], ticker)}",
            f"Net profit margin: {fin['profit_ttm'] / fin['revenue_ttm']:.1%}",
        ]
        if "revenue_growth" in fin:
            fin_facts.append(f"Revenue growth vs same quarter last year: {fin['revenue_growth']:+.1%}")
        if fin.get("equity", 0) > 0:
            fin_facts.append(f"Return on equity (ROE): {fin['profit_ttm'] / fin['equity']:.1%}")
            if "total_debt" in fin:
                fin_facts.append(f"Debt-to-equity ratio: {fin['total_debt'] / fin['equity']:.2f}")
        if fin.get("shares", 0) > 0 and fin["profit_ttm"] > 0 and len(prices) > 0:
            pe = prices.iloc[-1] / (fin["profit_ttm"] / fin["shares"])
            fin_facts.append(f"Price-to-earnings (P/E) ratio: {pe:.1f}")
        for i, text in enumerate(fin_facts, start=1):
            items.append(Evidence(id=f"F{i}", category="financials", text=text))

    # --- N: news headlines (using headline aliases) ---
    news, aliases = _relevant_news(ticker, as_of, max_news)
    for i, (_, row) in enumerate(news.iterrows(), start=1):
        title = row["title"]
        if anonymize:
            for alias in aliases + [company_name]:
                title = title.replace(alias, name)
        day = pd.to_datetime(row["date"]).date()
        items.append(Evidence(id=f"N{i}", category="news", text=f"{day}: {title}"))

    # --- A: other agents' results ---
    for i, text in enumerate(agent_results or [], start=1):
        items.append(Evidence(id=f"A{i}", category="agent", text=text))

    return EvidencePack(ticker="HIDDEN" if anonymize else ticker,
                        company=name, as_of=str(as_of), items=items)