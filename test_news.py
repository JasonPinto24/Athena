from athena.tools.news import get_news

news = get_news("Infosys", "2026-10-17")
print("Headlines found:", len(news))
print(news[["date", "title"]].head(5))