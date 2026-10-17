from athena.agents.evidence import build_evidence_pack

agent_notes = [
    "Forecast model: 47.5% chance of being a top performer; ranks 15th percentile among 20 peers",
    "Risk agent: max drawdown -40.4% in the last year; earnings due 2026-10-23",
]

pack = build_evidence_pack("INFY.NS", "Infosys", "2026-10-16", agent_results=agent_notes)
print(pack.to_text())
print("\nValid IDs:", sorted(pack.ids()))