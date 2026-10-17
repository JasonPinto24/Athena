from pydantic import BaseModel, Field

from athena.agents.debate import Claim, ask, validate
from athena.scanner import BUY_FRACTION, CLOSE_CALL, SELL_FRACTION, label_from_rank

BUY_LINE = 1 - BUY_FRACTION    # 0.75
SELL_LINE = SELL_FRACTION      # 0.25


class Report(BaseModel):
    headline: str = Field(description="One full sentence explaining the verdict, not just the decision word")
    reasons: list[Claim] = Field(description="3-4 key reasons, each citing evidence IDs")
    risks: list[Claim] = Field(description="1-2 main risks, each citing evidence IDs")


def near_lines(pct):
    near_buy = abs(pct - BUY_LINE) <= CLOSE_CALL
    near_sell = abs(pct - SELL_LINE) <= CLOSE_CALL
    return near_buy, near_sell


def debate_can_change(pct, risk):
    near_buy, near_sell = near_lines(pct)
    if near_buy:
        return not risk["veto"]    # vetoed -> Hold either way, debate is pointless
    return near_sell


def _sign(x):
    return (x > 0) - (x < 0)


def decide(forecast, risk, F=None, S=None, D=None):
    p = forecast["p_top"]
    pct = forecast["rank_among_peers"]

    # 1. Signal from the forecast ranking
    signal = label_from_rank(pct)
    decision = signal

    # 2. Close call -> the debate breaks the tie
    near_buy, near_sell = near_lines(pct)
    close_call = near_buy or near_sell
    tie_break = None
    if close_call and D:    # D is None if skipped; 0 means a draw -> keep the signal
        if near_buy:
            decision = "Buy" if D > 0 else "Hold"
        else:
            decision = "Sell" if D < 0 else "Hold"
        verb = "confirmed" if decision == signal else f"moved {signal} ->"
        tie_break = f"Debate {verb} {decision} (D = {D:+.2f})"

    # 3. Confidence = half distance past the cut-off, half agreement of F, S, D
    if decision == "Buy":
        strength = (pct - BUY_LINE) / BUY_FRACTION
    elif decision == "Sell":
        strength = (SELL_LINE - pct) / SELL_FRACTION
    else:
        strength = min(pct - SELL_LINE, BUY_LINE - pct) / ((BUY_LINE - SELL_LINE) / 2)
    strength = max(0.0, min(1.0, strength))

    support = [x for x in (F, S, D) if x is not None]
    direction = {"Buy": 1, "Sell": -1}.get(decision)
    if direction is not None and support:
        agree = sum(1 for x in support if _sign(x) == direction) / len(support)
    else:
        agree = 0.5
    confidence = 0.5 * strength + 0.5 * agree

    # 4. Risk brake (only blocks Buys)
    vetoed = decision == "Buy" and risk["veto"]
    if vetoed:
        decision = "Hold"

    # 5. Position size (Buy only)
    size = risk["position_size"] if decision == "Buy" else 0.0

    return {
        "decision": decision,
        "signal": signal,
        "p_top": round(p, 3),
        "percentile": round(pct, 3),
        "close_call": close_call,
        "tie_break": tie_break,
        "confidence": round(confidence, 2),
        "position_size": size,
        "vetoed": vetoed,
        "veto_reasons": risk["reasons"] if vetoed else [],
        "scores": {
            "forecast_M": round(forecast["M"], 3),
            "fundamentals_F": None if F is None else round(F, 3),
            "sentiment_S": None if S is None else round(S, 3),
            "debate_D": None if D is None else round(D, 3),
        },
    }


def write_report(pack, result, debate):
    system = (
        "You are the portfolio manager writing a short investment report. "
        f"The decision is FINAL and is: {result['decision']}. Do not change it. "
        "The decision comes from a trained forecast model ranking this stock against its "
        "Nifty 50 peers, then a risk check. Explain it using ONLY the evidence provided. "
        "Every reason and risk must cite evidence IDs (like F5 or N2). Do not invent facts or numbers. "
        "The headline must be one full sentence explaining the verdict, not just the decision word."
    )
    user = (
        f"{pack.to_text()}\n\n"
        f"Decision: {result['decision']} | Signal from the forecast ranking: {result['signal']}\n"
        f"Forecast: {result['p_top']:.1%} chance of being a top-30% performer next month; "
        f"ranks above {result['percentile']:.0%} of peers\n"
        f"Close call: {result['close_call']} | Tie-break: {result['tie_break'] or 'none'}\n"
        f"Confidence: {result['confidence']:.0%}\n"
        f"Context scores (explain, not used for ranking): {result['scores']}\n"
        f"Risk veto: {result['vetoed']} {result['veto_reasons']}\n"
        f"Debate winner: {debate['winner']} (D = {debate['D']}). Judge said: {debate['reasoning']}"
    )
    report = ask(system, user, Report)

    valid = pack.ids()
    reasons, _ = validate(report.reasons, valid)
    risks, _ = validate(report.risks, valid)
    return {"headline": report.headline, "reasons": reasons, "risks": risks}