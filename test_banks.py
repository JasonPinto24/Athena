from athena.agents.fundamentals import fundamentals_score

for t in ["HDFCBANK.NS", "ICICIBANK.NS", "BAJFINANCE.NS", "INFY.NS"]:
    r = fundamentals_score(t, "2026-10-16")
    print(f"{t:15s} F={r['F']:+.3f}  source={r.get('data_source')}  "
          f"financial={r.get('is_financial')}  grades={r['grades']}")