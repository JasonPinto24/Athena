from athena.agents.risk import risk_check

for ticker in ["INFY.NS", "TATASTEEL.NS"]:
    print(ticker)
    result = risk_check(ticker, "2026-10-16")
    for key, value in result.items():
        print("  ", key, ":", value)