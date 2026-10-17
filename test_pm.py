from athena.graph import analyze
from athena.tools.company import company_name

AS_OF = "2026-10-17"
# clear Buy, close call near the Buy line, clear Sell (from your scan)
for t in ["MARUTI.NS", "HINDUNILVR.NS", "TECHM.NS"]:
    print(f"\n===== {t} =====")
    out = analyze(t, company_name(t), AS_OF)
    r = out["result"]
    print(f"Signal: {r['signal']} -> Decision: {r['decision']}")
    print(f"p_top: {r['p_top']} | ranks above {r['percentile']:.0%} of peers | close call: {r['close_call']}")
    print(f"Debate ran: {'debate' in out['steps']} | tie-break: {r['tie_break']}")
    print(f"Confidence: {r['confidence']} | size: {r['position_size']} | veto: {r['vetoed']}")
    print(f"Scores: {r['scores']}")
    print(f"Headline: {out['report']['headline']}")