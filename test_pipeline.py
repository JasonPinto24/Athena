from athena.agents.fundamentals import fundamentals_score
from athena.agents.sentiment import sentiment_score
from athena.agents.forecast import forecast_score
from athena.agents.risk import risk_check
from athena.agents.evidence import build_evidence_pack
from athena.agents.debate import run_debate
from athena.agents.portfolio_manager import decide, write_report

TICKER, COMPANY, AS_OF = "INFY.NS", "Infosys", "2026-10-16"

print("1-3. Running analysts...")
fund = fundamentals_score(TICKER, AS_OF)
sent = sentiment_score(COMPANY, AS_OF)
fore = forecast_score(TICKER, AS_OF)
risk = risk_check(TICKER, AS_OF, decision="Buy")

notes = [
    f"Fundamentals agent: score {fund['F']:+.2f}, grades {fund['grades']}",
    f"Sentiment agent: score {sent['S']:+.2f} from {sent['n']} recent headlines",
    f"Forecast model: {fore.get('p_top', 0.5):.1%} chance of being a top performer; "
    f"ranks {fore.get('rank_among_peers', 0.5):.0%} among peers",
    f"Risk agent: volatility {risk.get('volatility', 0):.0%}, max drawdown {risk.get('max_drawdown', 0):.1%}, "
    f"concerns: {risk['reasons'] or 'none'}",
]

print("4. Building evidence pack...")
pack = build_evidence_pack(TICKER, COMPANY, AS_OF, agent_results=notes)

print("5. Running debate...")
debate = run_debate(pack)

print("6-7. Deciding and writing report...")
result = decide(fund["F"], sent["S"], fore["M"], debate["D"], risk)
report = write_report(pack, result, debate)

print("\n========== ATHENA REPORT ==========")
print(f"{COMPANY} ({TICKER}) as of {AS_OF}")
print(f"DECISION: {result['decision']}  |  Score C: {result['C']:+.3f}  |  Confidence: {result['confidence']:.0%}")
print(f"Position size: {result['position_size']:.1%}  |  Risk veto: {result['vetoed']}")
print("Scores:", result["scores"])
print("Contributions:", result["contributions"])
print("\n" + report["headline"])
print("\nReasons:")
for r in report["reasons"]:
    print(" -", r.text, r.citations)
print("\nRisks:")
for r in report["risks"]:
    print(" -", r.text, r.citations)