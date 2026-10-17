from athena.agents.fundamentals import fundamentals_score

result = fundamentals_score("INFY.NS", "2026-10-17")
print("F score:", result["F"])
print("Grades:", result["grades"])
print("Price used:", result["price"])