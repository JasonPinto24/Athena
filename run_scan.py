import sys
from datetime import date

import pandas as pd

from athena.scanner import scan

args = [a for a in sys.argv[1:] if not a.startswith("--")]
as_of = args[0] if args else str(date.today())
with_news = "--news" in sys.argv

print(f"Scanning Nifty 50 as of {as_of} | news: {'on' if with_news else 'off'}\n")
table = scan(as_of, with_news=with_news)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)
cols = ["rank", "ticker", "p_top", "decision", "close_call", "F", "S", "news_n", "vol", "veto_reasons"]
print("\n" + table[cols].to_string(index=False))

print("\nDecisions:")
print(table["decision"].value_counts().to_string())
print(f"Close calls (debate will decide): {table['close_call'].sum()}")

out = f"data/scan_{as_of}.csv"
table.to_csv(out, index=False)
print(f"\nSaved {out}")