import requests

params = {
    "query": '"Infosys" sourcelang:english',
    "mode": "artlist",
    "format": "json",
    "maxrecords": 10,
    "timespan": "7d",
}

r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
print("Status:", r.status_code)
print(r.text[:500])