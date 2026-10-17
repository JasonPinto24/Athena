import yfinance as yf

for ticker in ["HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS"]:
    stock = yf.Ticker(ticker)
    income = stock.quarterly_income_stmt
    balance = stock.quarterly_balance_sheet
    print(f"\n===== {ticker} =====")
    print("Sector:", stock.info.get("sector"), "| Industry:", stock.info.get("industry"))
    print("Quarters available:", [str(c.date()) for c in income.columns])
    print("Income rows (first 15):", list(income.index[:15]))
    print("Has 'Total Revenue':", "Total Revenue" in income.index,
          "| Has 'Net Income':", "Net Income" in income.index)
    print("Has 'Total Debt':", "Total Debt" in balance.index,
          "| Has 'Stockholders Equity':", "Stockholders Equity" in balance.index)