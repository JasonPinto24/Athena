import json
import re
import time

import yaml
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# ---------- The "forms" the AI must fill in ----------
class Claim(BaseModel):
    text: str = Field(description="One-sentence argument")
    citations: list[str] = Field(description="Evidence IDs that support it, e.g. ['F5', 'N2']")


class Argument(BaseModel):
    claims: list[Claim]


class Judgement(BaseModel):
    bull_convincing: int = Field(description="How many bull claims are convincing")
    bear_convincing: int = Field(description="How many bear claims are convincing")
    decisive_ids: list[str] = Field(description="The 2 evidence IDs that mattered most")
    reasoning: str = Field(description="2-3 sentence explanation of the verdict")


# ---------- Talking to the LLM ----------
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=CONFIG["groq_model"], temperature=0)
    return _llm


def parse_json(text, form):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in the reply")
    return form.model_validate_json(match.group(0))


def ask(system, user, form, attempts=3):
    last_error = None
    schema = json.dumps(form.model_json_schema())
    for attempt in range(attempts):
        # Try 1 and 2: let LangChain force the form
        for method in ("json_schema", "function_calling"):
            try:
                llm = get_llm().with_structured_output(form, method=method)
                return llm.invoke([("system", system), ("human", user)])
            except Exception as error:
                last_error = error
        # Try 3: ask for plain JSON text and read it ourselves
        try:
            json_system = f"{system}\nReply ONLY with JSON that matches this schema:\n{schema}"
            reply = get_llm().invoke([("system", json_system), ("human", user)])
            return parse_json(reply.content, form)
        except Exception as error:
            last_error = error
        print(f"LLM call failed (attempt {attempt + 1}/{attempts}), retrying...")
        time.sleep(2)
    raise last_error


# ---------- Citation checking ----------
def clean_id(raw):
    return raw.strip().strip("[]").upper()


def validate(claims, valid_ids):
    kept, dropped = [], 0
    for claim in claims:
        ids = [clean_id(i) for i in claim.citations]
        if ids and all(i in valid_ids for i in ids):
            kept.append(Claim(text=claim.text, citations=ids))
        else:
            dropped += 1
    return kept, dropped


def as_list(claims):
    if not claims:
        return "(no valid claims)"
    return "\n".join(f"{n}. {c.text} [{', '.join(c.citations)}]" for n, c in enumerate(claims, 1))


RULES = (
    "Use ONLY the evidence provided. Every claim must cite at least one evidence ID "
    "(like F5 or N2) in its citations. Do not invent facts or numbers. "
    "Each claim is one sentence about the stock's prospects over the next month."
)


# ---------- The three roles ----------
def opening(side, pack):
    stance = "SHOULD be bought now" if side == "bull" else "should NOT be bought now"
    system = f"You are the {side} analyst. Argue that {pack.company} {stance}. {RULES}"
    user = f"{pack.to_text()}\n\nMake exactly 3 claims."
    return ask(system, user, Argument).claims


def rebuttal(side, pack, opponent_claims, own_claims):
    stance = "SHOULD be bought now" if side == "bull" else "should NOT be bought now"
    system = f"You are the {side} analyst arguing that {pack.company} {stance}. {RULES}"
    user = (
        f"{pack.to_text()}\n\n"
        f"Your earlier claims (do NOT repeat these):\n{as_list(own_claims)}\n\n"
        f"Your opponent argued:\n{as_list(opponent_claims)}\n\n"
        "Make exactly 2 NEW claims that directly answer their strongest points. "
        "Each claim should name which opponent point it answers. "
        "Do not repeat arguments already made."
    )
    return ask(system, user, Argument).claims


def judge(pack, bull, bear):
    system = (
        "You are an impartial investment judge. A claim is convincing only if its cited "
        "evidence really supports it and it matters for the next month. "
        "A claim that only repeats an earlier point does not count twice. "
        "Count the convincing claims on each side."
    )
    user = f"{pack.to_text()}\n\nBULL claims:\n{as_list(bull)}\n\nBEAR claims:\n{as_list(bear)}"
    verdict = ask(system, user, Judgement)
    v_bull = max(0, min(verdict.bull_convincing, len(bull)))
    v_bear = max(0, min(verdict.bear_convincing, len(bear)))
    D = (v_bull - v_bear) / (v_bull + v_bear) if (v_bull + v_bear) > 0 else 0.0
    return verdict, v_bull, v_bear, D


# ---------- The full debate ----------
def run_debate(pack, max_rounds=3, close_call=0.2):
    valid = pack.ids()

    bull_open, d1 = validate(opening("bull", pack), valid)
    bear_open, d2 = validate(opening("bear", pack), valid)
    bull_reb, d3 = validate(rebuttal("bull", pack, bear_open, bull_open), valid)
    bear_reb, d4 = validate(rebuttal("bear", pack, bull_open, bear_open), valid)

    bull = bull_open + bull_reb
    bear = bear_open + bear_reb
    dropped = d1 + d2 + d3 + d4
    rounds = 2

    verdict, v_bull, v_bear, D = judge(pack, bull, bear)

    if abs(D) < close_call and max_rounds >= 3:
        bull3, d5 = validate(rebuttal("bull", pack, bear_reb, bull), valid)
        bear3, d6 = validate(rebuttal("bear", pack, bull_reb, bear), valid)
        bull += bull3
        bear += bear3
        dropped += d5 + d6
        rounds = 3
        verdict, v_bull, v_bear, D = judge(pack, bull, bear)

    return {
        "D": round(D, 3),
        "winner": "bull" if D > 0 else "bear" if D < 0 else "tie",
        "bull_convincing": v_bull,
        "bear_convincing": v_bear,
        "rounds": rounds,
        "claims_dropped": dropped,
        "decisive_ids": [clean_id(i) for i in verdict.decisive_ids],
        "reasoning": verdict.reasoning,
        "bull_claims": bull,
        "bear_claims": bear,
    }