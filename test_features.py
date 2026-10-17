from athena.tools.prices import download_prices
from athena.tools.features import build_features, FEATURES

TICKERS = [
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS",
    "NESTLEIND.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "MARUTI.NS", "M&M.NS", "TATASTEEL.NS", "LT.NS",
]

for t in TICKERS + ["^NSEI"]:
    rows = download_prices(t, start="2018-01-01")
    print(t, rows)

df = build_features("INFY.NS", "2026-10-16")
print(df[FEATURES + ["label"]].dropna().tail())
print("Rows with features and label:", len(df.dropna()))