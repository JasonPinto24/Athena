from athena.agents.risk import risk_check
from athena.graph import analyze
from athena.tools.company import company_name

AS_OF = "2026-10-17"

# 1. Risk text: should show one decimal (e.g. -40.3%, -54.x%)
for t in ["INFY.NS", "TRENT.NS"]:
    print(t, risk_check(t, AS_OF)["reasons"])

# 2 + 3. HINDUNILVR is now a close call -> the debate runs; news is tried only once
t = "HINDUNILVR.NS"
print(f"\n===== {t} =====")
out = analyze(t, company_name(t), AS_OF)
r, d = out["result"], out.get("debate", {})
print(f"Signal: {r['signal']} -> Decision: {r['decision']} | tie-break: {r['tie_break']}")
print(f"Debate: winner={d.get('winner')} D={d.get('D')} rounds={d.get('rounds')} dropped={d.get('claims_dropped')}")
print("\nBULL claims:")
for c in d.get("bull_claims", []):
    print(f"  - {c.text} {c.citations}")
print("BEAR claims:")
for c in d.get("bear_claims", []):
    print(f"  - {c.text} {c.citations}")
print(f"\nJudge: {d.get('reasoning')}")
print(f"Headline: {out['report']['headline']}")