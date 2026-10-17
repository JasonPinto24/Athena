from athena.tools.financials import get_financials

data = get_financials("INFY.NS", "2026-10-17")
for key, value in data.items():
    print(key, ":", value)