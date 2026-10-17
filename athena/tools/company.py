import re

import yfinance as yf
from joblib import Memory

memory = Memory("data/cache", verbose=0)
SUFFIXES = r"\b(limited|ltd|inc|incorporated|corporation|corp|plc|holdings)\b\.?"

# How each company is usually written in headlines (add more anytime)
NEWS_ALIASES = {
    "INFY.NS": ["Infosys"],
    "TCS.NS": ["TCS", "Tata Consultancy"],
    "WIPRO.NS": ["Wipro"],
    "HCLTECH.NS": ["HCLTech", "HCL Tech", "HCL Technologies"],
    "RELIANCE.NS": ["Reliance Industries", "RIL"],
    "HDFCBANK.NS": ["HDFC Bank"],
    "ICICIBANK.NS": ["ICICI Bank"],
    "KOTAKBANK.NS": ["Kotak Mahindra", "Kotak Bank"],
    "SBIN.NS": ["SBI", "State Bank of India"],
    "AXISBANK.NS": ["Axis Bank"],
    "ITC.NS": ["ITC"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "NESTLEIND.NS": ["Nestle India", "Nestlé India"],
    "SUNPHARMA.NS": ["Sun Pharma"],
    "DRREDDY.NS": ["Dr Reddy", "Dr. Reddy"],
    "CIPLA.NS": ["Cipla"],
    "MARUTI.NS": ["Maruti Suzuki", "Maruti"],
    "M&M.NS": ["Mahindra & Mahindra", "M&M"],
    "TATASTEEL.NS": ["Tata Steel"],
    "LT.NS": ["Larsen & Toubro", "Larsen", "L&T"],
}


@memory.cache
def company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        name = ticker
    name = name.replace("’", "'")
    name = re.sub(SUFFIXES, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s&']", " ", name)
    return " ".join(word.capitalize() for word in name.split())


def news_aliases(ticker):
    return NEWS_ALIASES.get(ticker) or [company_name(ticker)]


def title_matches(title, aliases):
    for alias in aliases:
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, str(title), flags=re.IGNORECASE):
            return True
    return False