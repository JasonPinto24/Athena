import operator
from typing import Annotated, Any, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph

from athena.agents.debate import run_debate
from athena.agents.evidence import build_evidence_pack
from athena.agents.forecast import forecast_score
from athena.agents.fundamentals import fundamentals_score
from athena.agents.portfolio_manager import decide, write_report
from athena.agents.risk import risk_check
from athena.agents.sentiment import sentiment_score

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)
WEIGHTS = CONFIG["weights"]


# ---------- The shared notebook ----------
class AthenaState(TypedDict, total=False):
    ticker: str
    company: str
    as_of: str
    force_debate: bool
    fundamentals: dict
    sentiment: dict
    forecast: dict
    risk: dict
    pack: Any
    debate: dict
    result: dict
    report: dict
    steps: Annotated[list[str], operator.add]


# ---------- Nodes: the 4 analysts (run in parallel) ----------
def fundamentals_node(state):
    return {"fundamentals": fundamentals_score(state["ticker"], state["as_of"]),
            "steps": ["fundamentals"]}


def sentiment_node(state):
    return {"sentiment": sentiment_score(state["company"], state["as_of"]),
            "steps": ["sentiment"]}


def forecast_node(state):
    return {"forecast": forecast_score(state["ticker"], state["as_of"]),
            "steps": ["forecast"]}


def risk_node(state):
    return {"risk": risk_check(state["ticker"], state["as_of"], decision="Buy"),
            "steps": ["risk"]}


# ---------- Node: build the evidence pack ----------
def evidence_node(state):
    f, s, m, r = state["fundamentals"], state["sentiment"], state["forecast"], state["risk"]
    notes = [
        f"Fundamentals agent: score {f['F']:+.2f}, grades {f.get('grades', {})}",
        f"Sentiment agent: score {s['S']:+.2f} from {s.get('n', 0)} recent headlines",
        f"Forecast model: {m.get('p_top', 0.5):.1%} chance of being a top performer; "
        f"ranks {m.get('rank_among_peers', 0.5):.0%} among peers",
        f"Risk agent: volatility {r.get('volatility', 0):.0%}, "
        f"max drawdown {r.get('max_drawdown', 0):.1%}, concerns: {r['reasons'] or 'none'}",
    ]
    pack = build_evidence_pack(state["ticker"], state["company"], state["as_of"], agent_results=notes)
    return {"pack": pack, "steps": ["evidence"]}


# ---------- Conditional edge: is the debate worth running? ----------
def final_label(C, risk):
    if C >= CONFIG["buy_threshold"]:
        label = "Buy"
    elif C <= CONFIG["sell_threshold"]:
        label = "Sell"
    else:
        label = "Hold"
    if label == "Buy" and risk["veto"]:
        label = "Hold"
    return label


def route_after_evidence(state):
    if state.get("force_debate"):
        return "debate"
    partial = (WEIGHTS["fundamentals"] * state["fundamentals"]["F"]
               + WEIGHTS["sentiment"] * state["sentiment"]["S"]
               + WEIGHTS["forecast"] * state["forecast"]["M"])
    swing = WEIGHTS["debate"]
    if final_label(partial + swing, state["risk"]) != final_label(partial - swing, state["risk"]):
        return "debate"
    return "skip"


# ---------- Nodes: debate, decide, report ----------
def debate_node(state):
    return {"debate": run_debate(state["pack"]), "steps": ["debate"]}


def decide_node(state):
    D = state.get("debate", {}).get("D", 0.0)
    result = decide(state["fundamentals"]["F"], state["sentiment"]["S"],
                    state["forecast"]["M"], D, state["risk"])
    return {"result": result, "steps": ["decide"]}


def report_node(state):
    debate = state.get("debate") or {
        "winner": "skipped", "D": 0.0,
        "reasoning": "Debate skipped because it could not change the decision.",
    }
    return {"report": write_report(state["pack"], state["result"], debate), "steps": ["report"]}


# ---------- Build the graph ----------
def build_graph():
    g = StateGraph(AthenaState)

    g.add_node("fundamentals", fundamentals_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("forecast", forecast_node)
    g.add_node("risk", risk_node)
    g.add_node("evidence", evidence_node)
    g.add_node("debate", debate_node)
    g.add_node("decide", decide_node)
    g.add_node("report", report_node)

    for analyst in ["fundamentals", "sentiment", "forecast", "risk"]:
        g.add_edge(START, analyst)
        g.add_edge(analyst, "evidence")

    g.add_conditional_edges("evidence", route_after_evidence,
                            {"debate": "debate", "skip": "decide"})
    g.add_edge("debate", "decide")
    g.add_edge("decide", "report")
    g.add_edge("report", END)

    return g.compile()


app = build_graph()


def analyze(ticker, company, as_of, force_debate=False):
    return app.invoke({"ticker": ticker, "company": company, "as_of": as_of,
                       "force_debate": force_debate, "steps": []})