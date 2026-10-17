import yfinance as yf

df = yf.download("INFY.NS", start="2025-01-01", auto_adjust=True, multi_level_index=False)

df["return"] = df["Close"].pct_change()

print(df[["Close", "return"]].tail())