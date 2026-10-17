from athena.agents.forecast import forecast_score
from athena.scanner import is_close_call, label_from_rank

AS_OF = "2026-10-17"
for t in ["MARUTI.NS", "HINDUNILVR.NS", "TECHM.NS", "IRCTC.NS", "ZOMATO.NS"]:
    f = forecast_score(t, AS_OF)
    pct = f["rank_among_peers"]
    print(f"{t:15} pct={pct:.3f}  signal={label_from_rank(pct):5}  close_call={is_close_call(pct)}  "
          f"in_nifty50={f['in_universe']}  peers={f['n_peers']}  {f.get('note', '')}")