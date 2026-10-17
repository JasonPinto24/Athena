from athena.tools.prices import download_prices, get_prices

rows = download_prices("INFY.NS")
print("Saved rows:", rows)

df = get_prices("INFY.NS", "2024-06-30")
print(df.tail(3))