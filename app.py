import datetime
import html

import pandas as pd
import streamlit as st

from athena.agents.forecast import forecast_all
from athena.graph import analyze
from athena.scanner import label_from_rank
from athena.tools.company import company_name
from athena.tools.prices import download_prices, get_prices

st.set_page_config(page_title="Athena AI", page_icon="🦉", layout="centered",
                   initial_sidebar_state="collapsed")

# ---------- Look and feel ----------
st.markdown("""
<style>
.block-container {padding-top: 2.5rem; max-width: 920px;}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
.hero {text-align: center; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2.7rem; margin-bottom: 0;}
.hero p {color: #6b7280; font-size: 1.05rem; margin-top: .3rem;}
.verdict {border-radius: 18px; padding: 1.4rem 1.7rem; margin: 1.2rem 0 1rem 0; color: white;}
.verdict .label {font-size: .82rem; letter-spacing: .08em; text-transform: uppercase; opacity: .9;}
.verdict .decision {font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin-top: .2rem;}
.verdict .headline {font-size: 1.05rem; margin-top: .55rem; opacity: .97;}
.buy {background: linear-gradient(135deg, #047857, #10b981);}
.sell {background: linear-gradient(135deg, #b91c1c, #ef4444);}
.hold {background: linear-gradient(135deg, #b45309, #f59e0b);}
.pill {display: inline-block; padding: .3rem .75rem; border-radius: 999px; font-size: .86rem;
       margin: .15rem .35rem .35rem 0; background: rgba(120,120,140,.14);}
.pill.warn {background: rgba(245,158,11,.18);}
.step {border: 1px solid rgba(120,120,140,.25); border-radius: 14px; padding: 1rem; height: 100%;}
.step b {font-size: 1.02rem;}
</style>
""", unsafe_allow_html=True)


# ---------- Cached helpers (so the same question is never computed twice) ----------
@st.cache_data(show_spinner=False)
def price_rows(ticker, as_of):
    try:
        if len(get_prices(ticker, as_of)) == 0:
            download_prices(ticker, start="2018-01-01")
        return len(get_prices(ticker, as_of))
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def detect_name(ticker):
    try:
        return company_name(ticker)
    except Exception:
        return ticker


@st.cache_data(show_spinner=False)
def run_analysis(ticker, company, as_of, force_debate):
    state = analyze(ticker, company, as_of, force_debate=force_debate)
    state["evidence_text"] = state["pack"].to_text()
    state.pop("pack")
    return state


@st.cache_data(show_spinner=False)
def peer_table(as_of, ticker):
    return forecast_all(as_of, extra_ticker=ticker).reset_index()


def fmt(v, pattern="{:+.2f}"):
    return "unknown" if v is None else pattern.format(v)


def pill(text, warn=False):
    return f'<span class="pill{" warn" if warn else ""}">{html.escape(text)}</span>'


# ---------- Front page: title + search ----------
st.markdown('<div class="hero"><h1>🦉 Athena AI</h1>'
            '<p>Type a stock ticker. A team of AI agents researches it and gives a verdict with evidence.</p></div>',
            unsafe_allow_html=True)

with st.form("search"):
    c1, c2, c3 = st.columns([3, 2, 1.3], vertical_alignment="bottom")
    ticker_in = c1.text_input("Stock ticker", value=st.session_state.get("ticker", ""),
                              placeholder="e.g. MARUTI, TCS, INFY.NS")
    date_in = c2.date_input("As-of date", datetime.date.today())
    submitted = c3.form_submit_button("Analyze", type="primary", use_container_width=True)
    with st.expander("Advanced"):
        force_in = st.checkbox("Always run the bull vs bear debate (slower)", value=False)
st.caption("Indian stocks: `.NS` is added for you if you leave it out. Best for Nifty 50 companies.")

if submitted:
    t = ticker_in.strip().upper()
    if not t:
        st.warning("Please type a ticker first.")
        st.stop()
    if "." not in t and not t.startswith("^"):
        t += ".NS"
    st.session_state["ticker"] = t
    st.session_state["query"] = (t, str(date_in), force_in)

# ---------- Before the first search: how it works ----------
if "query" not in st.session_state:
    st.write("")
    a, b, c = st.columns(3)
    a.markdown('<div class="step"><b>1. Four analysts</b><br>Fundamentals, news sentiment, '
               'an ML forecast and a risk check run at the same time.</div>', unsafe_allow_html=True)
    b.markdown('<div class="step"><b>2. Ranked vs peers</b><br>The forecast ranks the stock against '
               'the Nifty 50. Close calls go to a bull vs bear debate.</div>', unsafe_allow_html=True)
    c.markdown('<div class="step"><b>3. Cited verdict</b><br>Buy, Hold or Sell, with a report where '
               'every claim cites a numbered fact.</div>', unsafe_allow_html=True)
    st.stop()

ticker, as_of, force = st.session_state["query"]

# ---------- Check the ticker exists ----------
rows = price_rows(ticker, as_of)
if rows == 0:
    st.error(f"No price data found for **{ticker}**. Check the ticker on Yahoo Finance "
             "(Indian stocks end in `.NS`, e.g. `TCS.NS`).")
    st.stop()
if rows < 252:
    st.warning(f"{ticker} has less than one year of price history, so the forecast and risk checks are limited.")

company = detect_name(ticker)
with st.spinner(f"Agents are researching {company}... (about 1-2 minutes the first time)"):
    state = run_analysis(ticker, company, as_of, force)
    peers = peer_table(as_of, ticker)

result, report, risk, fc = state["result"], state["report"], state["risk"], state["forecast"]
decision = result["decision"]

# ---------- Verdict card ----------
st.markdown(
    f'<div class="verdict {decision.lower()}">'
    f'<div class="label">{html.escape(company)} · {ticker} · as of {as_of}</div>'
    f'<div class="decision">{decision.upper()}</div>'
    f'<div class="headline">{html.escape(report["headline"])}</div></div>',
    unsafe_allow_html=True,
)

pills = []
if not fc.get("in_universe", True):
    pills.append(pill("Outside the Nifty 50: ranked against Nifty 50 peers, lower confidence", warn=True))
if fc.get("note"):
    pills.append(pill(fc["note"], warn=True))
if result["close_call"]:
    pills.append(pill(f"Close call: {result['tie_break'] or 'near a cut-off'}"))
if result["vetoed"]:
    pills.append(pill("Risk veto: " + "; ".join(result["veto_reasons"]), warn=True))
if pills:
    st.markdown("".join(pills), unsafe_allow_html=True)

# ---------- Key numbers ----------
universe = peers[peers["in_universe"]].sort_values("p", ascending=False).reset_index(drop=True)
n_peers = len(universe)
rank = int((universe["p"] > fc.get("p_top", 0.5)).sum()) + 1
rank_label = f"#{rank} of {n_peers}" if fc.get("in_universe", True) else f"~#{rank} of {n_peers}"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rank vs Nifty 50", rank_label, help="Rank by the forecast model's probability (1 = best).")
m2.metric("Chance of top 30%", f"{result['p_top']:.0%}",
          help="Model's probability that the stock is a top-30% performer over the next month.")
m3.metric("Confidence", f"{result['confidence']:.0%}",
          help="Half: how far past the Buy/Sell cut-off. Half: do fundamentals, news and debate agree?")
m4.metric("Position size", f"{result['position_size']:.1%}" if result["position_size"] else "—",
          help="Suggested share of the portfolio (Buys only, max 10%, smaller for jumpy stocks).")

# ---------- Details in tabs ----------
tab_sum, tab_peers, tab_agents, tab_debate, tab_ev = st.tabs(
    ["Summary", "Peers", "Agents", "Debate", "Evidence"])

with tab_sum:
    left, right = st.columns(2)
    with left:
        st.markdown("**Why**")
        for r in report["reasons"]:
            st.markdown(f"- {r.text} `{', '.join(r.citations)}`")
    with right:
        st.markdown("**Risks**")
        for r in report["risks"]:
            st.markdown(f"- {r.text} `{', '.join(r.citations)}`")

    st.markdown("**Price vs Nifty 50** (last year, both start at 100)")
    stock = get_prices(ticker, as_of).set_index("date")["close"].tail(252)
    index = get_prices("^NSEI", as_of).set_index("date")["close"].reindex(stock.index).ffill()
    if len(stock) > 1:
        st.line_chart(pd.DataFrame({ticker: stock / stock.iloc[0] * 100,
                                    "Nifty 50": index / index.iloc[0] * 100}))

with tab_peers:
    st.caption("All Nifty 50 stocks ranked by the forecast model for this date. "
               "Top 25% = Buy, bottom 25% = Sell (before the risk check).")
    short = ticker.replace(".NS", "")
    table = pd.DataFrame({
        "Rank": range(1, n_peers + 1),
        "Stock": universe["ticker"].str.replace(".NS", "", regex=False),
        "Chance of top 30%": universe["p"] * 100,
        "Signal": universe["p_rank"].apply(label_from_rank),
    })
    if not fc.get("in_universe", True):
        st.info(f"{short} is not in the Nifty 50. It would sit at about **#{rank}** in this table.")

    def highlight(row):
        on = row["Stock"] == short
        return ["background-color: rgba(16,185,129,.22); font-weight: 700" if on else "" for _ in row]

    st.dataframe(table.style.apply(highlight, axis=1).format({"Chance of top 30%": "{:.1f}%"}),
                 hide_index=True, use_container_width=True, height=430)

with tab_agents:
    s = result["scores"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Forecast (M)", fmt(s["forecast_M"]), help="Ranks the stock. -1 to +1.")
    a2.metric("Fundamentals (F)", fmt(s["fundamentals_F"]), help="Context: cheap, safe, growing, efficient, profitable.")
    a3.metric("News sentiment (S)", fmt(s["sentiment_S"]), help="Context: FinBERT on recent headlines. 'unknown' = no news.")
    a4.metric("Debate (D)", fmt(s["debate_D"]), help="Tie-breaker for close calls. 'unknown' = skipped.")

    grades = state["fundamentals"].get("grades") or {}
    if grades:
        st.markdown("**Fundamentals grades** (-1 bad, +1 good)")
        st.bar_chart(pd.Series(grades, name="grade"))

    sent = state["sentiment"]
    st.markdown(f"**News:** {sent.get('n', 0)} relevant headlines in the last 30 days.")
    if sent.get("most_positive"):
        st.markdown(f"- Most positive: {sent['most_positive']}")
    if sent.get("most_negative"):
        st.markdown(f"- Most negative: {sent['most_negative']}")

    st.markdown("**Risk check**")
    r1, r2, r3 = st.columns(3)
    r1.metric("Volatility (yearly)", fmt(risk.get("volatility"), "{:.1%}"))
    r2.metric("Max fall from peak (1y)", fmt(risk.get("max_drawdown"), "{:.1%}"))
    r3.metric("Beta vs market", fmt(risk.get("beta"), "{:.2f}"))
    for reason in risk.get("reasons", []):
        st.markdown(f"- ⚠️ {reason}")

    st.caption(f"Confidence {result['confidence']:.0%} = half distance past cut-off "
               f"({result.get('strength', 0):.0%}) + half agreement of other agents ({result.get('agreement', 0):.0%}).")

with tab_debate:
    debate = state.get("debate")
    if not debate:
        if result["close_call"] and result["tie_break"]:
            st.info(result["tie_break"])
        else:
            st.info("Debate skipped: the stock is not near a Buy or Sell cut-off, so it could not change the verdict.")
    else:
        st.markdown(f"**Winner: {debate['winner']}** · D = {debate['D']:+.2f} · "
                    f"{debate['rounds']} rounds · {debate['claims_dropped']} uncited claims dropped")
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

with tab_ev:
    st.caption("Every claim above cites one of these numbered facts. P = price, F = financials, N = news, A = agents.")
    st.text(state["evidence_text"])
    st.caption("Workflow: " + " → ".join(state["steps"]))

st.divider()
st.caption("Athena AI · for learning purposes only, not financial advice.")
