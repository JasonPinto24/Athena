import datetime

import pandas as pd
import streamlit as st

from athena.graph import analyze
from athena.tools.company import company_name
from athena.tools.prices import get_prices

st.set_page_config(page_title="Athena AI", page_icon="🦉", layout="wide")


@st.cache_data(show_spinner=False)
def run_analysis(ticker, company, as_of, force_debate):
    state = analyze(ticker, company, as_of, force_debate=force_debate)
    state["evidence_text"] = state["pack"].to_text()
    state.pop("pack")
    return state


@st.cache_data(show_spinner=False)
def detect_name(ticker):
    return company_name(ticker)


# ---------- Sidebar: inputs ----------
st.sidebar.title("🦉 Athena AI")
st.sidebar.caption("Multi-agent equity research")
ticker = st.sidebar.text_input("Ticker", "INFY.NS").strip().upper()
as_of = st.sidebar.date_input("As-of date", datetime.date.today())
force_debate = st.sidebar.checkbox("Always run the debate", value=True)
go = st.sidebar.button("Run analysis", type="primary")

company = detect_name(ticker)
st.sidebar.caption(f"Company detected: **{company}**")

st.title("Athena AI: Multi-Agent Stock Research")

if not go:
    st.info("Enter a ticker in the sidebar, then click **Run analysis**.")
    st.stop()

with st.spinner(f"Agents are researching {company}... (about 1-2 minutes)"):
    state = run_analysis(ticker, company, str(as_of), force_debate)

result, report, risk = state["result"], state["report"], state["risk"]

# ---------- Decision banner ----------
banner = f"## {result['decision'].upper()}: {report['headline']}"
if result["decision"] == "Buy":
    st.success(banner)
elif result["decision"] == "Sell":
    st.error(banner)
else:
    st.warning(banner)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final score C", f"{result['C']:+.3f}")
c2.metric("Confidence", f"{result['confidence']:.0%}")
c3.metric("Position size", f"{result['position_size']:.1%}")
c4.metric("Risk veto", "Yes" if result["vetoed"] else "No")

# ---------- Agent scores ----------
st.subheader("Agent scores")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Fundamentals (F)", f"{result['scores']['fundamentals']:+.2f}")
s2.metric("Sentiment (S)", f"{result['scores']['sentiment']:+.2f}")
s3.metric("Forecast (M)", f"{result['scores']['forecast']:+.2f}")
s4.metric("Debate (D)", f"{result['scores']['debate']:+.2f}")

st.caption("Contribution of each agent to the final score (score x weight)")
contrib = pd.DataFrame.from_dict(result["contributions"], orient="index", columns=["contribution"])
st.bar_chart(contrib)

# ---------- Report ----------
st.subheader("Report")
left, right = st.columns(2)
with left:
    st.markdown("**Reasons**")
    for r in report["reasons"]:
        st.markdown(f"- {r.text} `{', '.join(r.citations)}`")
with right:
    st.markdown("**Risks**")
    for r in report["risks"]:
        st.markdown(f"- {r.text} `{', '.join(r.citations)}`")

# ---------- Price chart ----------
st.subheader("Price vs market (last year, both start at 100)")
stock = get_prices(ticker, str(as_of)).set_index("date")["close"].tail(252)
index = get_prices("^NSEI", str(as_of)).set_index("date")["close"].reindex(stock.index).ffill()
chart = pd.DataFrame({ticker: stock / stock.iloc[0] * 100, "Nifty 50": index / index.iloc[0] * 100})
st.line_chart(chart)

# ---------- Risk ----------
st.subheader("Risk check")
r1, r2, r3 = st.columns(3)
r1.metric("Volatility (yearly)", f"{risk.get('volatility', 0):.0%}")
r2.metric("Max drawdown (1 year)", f"{risk.get('max_drawdown', 0):.1%}")
r3.metric("Beta vs market", f"{risk.get('beta', 0):.2f}")
for reason in risk["reasons"]:
    st.markdown(f"⚠️ {reason}")

# ---------- Debate ----------
debate = state.get("debate")
with st.expander("⚔️ Bull vs bear debate"):
    if not debate:
        st.write("Debate skipped: it could not change the decision.")
    else:
        st.write(f"**Winner: {debate['winner']}** (D = {debate['D']:+.2f}, "
                 f"{debate['rounds']} rounds, {debate['claims_dropped']} claims dropped)")
        b, br = st.columns(2)
        with b:
            st.markdown("**🐂 Bull**")
            for c in debate["bull_claims"]:
                st.markdown(f"- {c.text} `{', '.join(c.citations)}`")
        with br:
            st.markdown("**🐻 Bear**")
            for c in debate["bear_claims"]:
                st.markdown(f"- {c.text} `{', '.join(c.citations)}`")
        st.markdown(f"**Judge:** {debate['reasoning']}")

# ---------- Evidence and workflow ----------
with st.expander("🗂️ Evidence pack (what every claim cites)"):
    st.text(state["evidence_text"])

with st.expander("🕸️ Agent workflow steps"):
    st.write(" → ".join(state["steps"]))

st.caption("For learning purposes only. Not financial advice.")