from athena.tools.prices import download_prices
from athena.universe import INDEX, UNIVERSE

ok, failed = [], []
for t in UNIVERSE + [INDEX]:
    try:
        rows = download_prices(t, start="2018-01-01")
        if rows > 0:
            ok.append(t)
            print(f"{t:16s} {rows} days")
        else:
            failed.append(t)
            print(f"{t:16s} NO DATA")
    except Exception as error:
        failed.append(t)
        print(f"{t:16s} FAILED: {error}")

print(f"\nDownloaded: {len(ok)} | Failed: {len(failed)}")
if failed:
    print("Failed tickers (check the symbol on Yahoo / NSE):", failed)