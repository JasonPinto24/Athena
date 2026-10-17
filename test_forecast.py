from athena.agents.forecast import forecast_score

result = forecast_score("INFY.NS", "2026-10-16")
for key, value in result.items():
    print(key, ":", value)