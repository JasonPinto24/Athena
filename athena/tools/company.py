import re

import yfinance as yf

SUFFIXES = r"\b(limited|ltd|inc|incorporated|corporation|corp|plc|holdings)\b\.?"


def company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
    except Exception:
        name = ticker
    name = re.sub(SUFFIXES, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s&]", " ", name)
    return " ".join(name.split()).title()