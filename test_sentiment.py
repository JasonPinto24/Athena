from athena.agents.sentiment import sentiment_score

result = sentiment_score("Infosys", "2026-10-17")
for key, value in result.items():
    print(key, ":", value)