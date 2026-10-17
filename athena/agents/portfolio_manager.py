import yaml
from pydantic import BaseModel, Field

from athena.agents.debate import Claim, ask, validate

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
WEIGHTS = CONFIG["weights"]


class Report(BaseModel):
    headline: str = Field(description="One-sentence verdict in plain English")
    reasons: list[Claim] = Field(description="3-4 key reasons, each citing evidence IDs")
    risks: list[Claim] = Field(description="1-2 main risks, each citing evidence IDs")


def decide(F, S, M, D, risk):
    scores = {"fundamentals": F, "sentiment": S, "forecast": M, "debate": D}

    # 1. Weighted combination
    contributions = {name: WEIGHTS[name] * score for name, score in scores.items()}
    C = sum(contributions.values())

    # 2. Decision from thresholds
    if C >= CONFIG["buy_threshold"]:
        decision = "Buy"
    elif C <= CONFIG["sell_threshold"]:
        decision = "Sell"
    else:
        decision = "Hold"

    # 3. Risk veto (only blocks Buys)
    vetoed = decision == "Buy" and risk["veto"]
    if vetoed:
        decision = "Hold"

    # 4. Position size (Buy only)
    size = risk["position_size"] if decision == "Buy" else 0.0

    # 5. Confidence = agreement x strength
    if C == 0:
        confidence = 0.0
    else:
        agree = sum(1 for s in scores.values() if s * C > 0)
        confidence = (agree / 4) * min(1.0, abs(C) / 0.5)

    return {
        "decision": decision,
        "C": round(C, 3),
        "confidence": round(confidence, 2),
        "position_size": size,
        "vetoed": vetoed,
        "veto_reasons": risk["reasons"] if vetoed else [],
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "contributions": {k: round(v, 3) for k, v in contributions.items()},
    }


def write_report(pack, result, debate):
    system = (
        "You are the portfolio manager writing a short investment report. "
        f"The decision is FINAL and is: {result['decision']}. Do not change it. "
        "Explain it using ONLY the evidence provided. Every reason and risk must cite "
        "evidence IDs (like F5 or N2). Do not invent facts or numbers."
    )
    user = (
        f"{pack.to_text()}\n\n"
        f"Decision: {result['decision']} | Final score C: {result['C']} | "
        f"Confidence: {result['confidence']:.0%}\n"
        f"Agent scores: {result['scores']}\n"
        f"Risk veto: {result['vetoed']} {result['veto_reasons']}\n"
        f"Debate winner: {debate['winner']} (D = {debate['D']}). Judge said: {debate['reasoning']}"
    )
    report = ask(system, user, Report)

    valid = pack.ids()
    reasons, _ = validate(report.reasons, valid)
    risks, _ = validate(report.risks, valid)
    return {"headline": report.headline, "reasons": reasons, "risks": risks}