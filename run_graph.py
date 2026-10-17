from athena.graph import analyze, app

print("GRAPH (paste into https://mermaid.live to see the picture):")
print(app.get_graph().draw_mermaid())

for force in [False, True]:
    print(f"\n========== Infosys | force_debate={force} ==========")
    state = analyze("INFY.NS", "Infosys", "2026-10-16", force_debate=force)
    r = state["result"]
    print("Steps run:", state["steps"])
    print(f"DECISION: {r['decision']} | C: {r['C']:+.3f} | confidence: {r['confidence']:.0%}")
    print("Debate:", state.get("debate", {}).get("winner", "skipped"),
          "| D:", state.get("debate", {}).get("D", 0.0))
    print("Headline:", state["report"]["headline"])