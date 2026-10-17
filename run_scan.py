import pandas as pd

from athena.scanner import scan

pd.set_option("display.width", 200)
table = scan("2026-10-16")
print("\n========== ATHENA SCAN (20 stocks) ==========")
print(table[["rank", "ticker", "company", "F", "S", "M", "score", "decision", "size"]]
      .round(3).to_string(index=False))