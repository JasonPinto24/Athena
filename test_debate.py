from athena.agents.evidence import build_evidence_pack
from athena.agents.debate import run_debate

notes = [
    "Fundamentals agent: score +0.52 (cheap +0.40, safe +0.90, growing +0.14, efficient +1.00, profitable +0.14)",
    "Sentiment agent: score +0.05 from 7 recent headlines (mildly positive)",
    "Forecast model: 47.5% chance of being a top performer; ranks 15th percentile among 20 peers",
    "Risk agent: max drawdown -40.4% in the last year; earnings due 2026-10-23",
]

pack = build_evidence_pack("INFY.NS", "Infosys", "2026-10-16", agent_results=notes)
result = run_debate(pack)

print("BULL CLAIMS:")
for c in result["bull_claims"]:
    print(" -", c.text, c.citations)
print("\nBEAR CLAIMS:")
for c in result["bear_claims"]:
    print(" -", c.text, c.citations)

print("\nD:", result["D"], "| winner:", result["winner"])
print("Convincing - bull:", result["bull_convincing"], "| bear:", result["bear_convincing"])
print("Rounds:", result["rounds"], "| claims dropped:", result["claims_dropped"])
print("Decisive evidence:", result["decisive_ids"])
print("Judge's reasoning:", result["reasoning"])