# app.py: the screens of the Client Portfolio Builder (built with Streamlit).
# All the maths lives in engine.py. This file only collects inputs and shows results.
# Educational tool only. Not financial advice.

import streamlit as st
import matplotlib.pyplot as plt
import engine as e

st.set_page_config(page_title="Client Portfolio Builder", page_icon="📈", layout="wide")

st.title("Client Portfolio Builder")
st.caption("Answer the risk questionnaire and set a goal to get a suitable model portfolio, its risk and return, "
           "and your chance of reaching the goal. **Educational tool only, not financial advice.**")


def show(fig):
    """Display a chart, then close it so charts don't pile up in memory."""
    st.pyplot(fig)
    plt.close(fig)

# ---------- Sidebar: the client's money and goal ----------
st.sidebar.header("Client details")
name = st.sidebar.text_input("Client name", "Test Client")
starting_amount = st.sidebar.number_input("Starting amount (£)", min_value=0, value=5000, step=500)
monthly_contribution = st.sidebar.number_input("Monthly contribution (£)", min_value=0, value=200, step=25)
years_to_goal = st.sidebar.slider("Years to goal", min_value=1, max_value=40, value=10)
goal_amount = st.sidebar.number_input("Goal amount (£)", min_value=1000, value=50000, step=1000)

tab1, tab2, tab3, tab4 = st.tabs(["1. Client profile", "2. Recommended portfolio",
                                  "3. Goal projection", "4. Client summary"])


# ---------- Tab 1: questionnaire ----------
with tab1:
    st.subheader("Attitude to risk")
    st.write("How much risk are you **willing** to take? Choose how far you agree with each statement.")
    atr_answers = []
    for i, question in enumerate(e.ATR_QUESTIONS):
        choice = st.radio(f"{i + 1}. {question}", e.AGREE_SCALE, index=2, horizontal=True, key=f"atr{i}")
        atr_answers.append(e.AGREE_SCALE.index(choice) + 1)      # "Strongly disagree" = 1 ... "Strongly agree" = 5

    st.subheader("Capacity for loss")
    st.write("How much loss could you **afford**? This is about your finances, not your feelings.")
    cfl_answers = []
    for i, item in enumerate(e.CFL_QUESTIONS):
        choice = st.radio(f"{i + 1}. {item['question']}", item["options"], index=2, horizontal=True, key=f"cfl{i}")
        cfl_answers.append(item["options"].index(choice) + 1)
client = {
    "name": name,
    "starting_amount": starting_amount,
    "monthly_contribution": monthly_contribution,
    "years_to_goal": years_to_goal,
    "goal_amount": goal_amount,
    "atr_answers": atr_answers,
    "cfl_answers": cfl_answers,
}

atr_level = e.score_to_level(e.average(atr_answers))
cfl_level = e.score_to_level(e.average(cfl_answers))
level, note = e.assess_client(client)

with tab1:
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Attitude to risk", e.risk_labels[atr_level])
    col2.metric("Capacity for loss", e.risk_labels[cfl_level])
    col3.metric("Recommended risk level", e.risk_labels[level])
    st.info(note)
    st.caption("Suitability rule: the recommended level is the **lower** of attitude to risk and capacity for loss.")
# ---------- Tab 2: the recommended portfolio ----------
exp_ret, vol, sharpe = e.portfolio_stats(level)
weights = e.model_portfolios[level]
equities = weights["UK equity"] + weights["Global equity"] + weights["Emerging markets"]
bonds = weights["UK gilts"] + weights["Global bonds"]
cash = weights["Cash"]

with tab2:
    st.subheader(f"Recommended portfolio: {e.risk_labels[level]}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected return", f"{exp_ret:.1%} a year")
    col2.metric("Volatility (risk)", f"{vol:.1%} a year")
    col3.metric("Sharpe ratio", f"{sharpe:.2f}")

    left, right = st.columns(2)
    with left:
        show(e.allocation_donut(level))
    with right:
        show(e.risk_return_chart(level))

    st.write(f"**Asset mix:** {equities}% equities, {bonds}% bonds, {cash}% cash.")
    st.caption("Expected returns, volatilities and correlations: J.P. Morgan Asset Management, 2027 Long-Term "
               "Capital Market Assumptions (Sterling), October 2026. Model portfolios are illustrative.")

# ---------- Tab 3: Monte Carlo goal projection ----------
prob, p10, p50, p90, values = e.monte_carlo(client)

with tab3:
    st.subheader("Will you reach your goal?")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chance of reaching goal", f"{prob:.0%}")
    col2.metric("Poor outcome (10th pct)", f"£{p10:,.0f}")
    col3.metric("Middle outcome (median)", f"£{p50:,.0f}")
    col4.metric("Good outcome (90th pct)", f"£{p90:,.0f}")

    show(e.fan_chart(client))

    if prob < 0.5:
        st.warning("Your chance of reaching the goal is below 50%. The main levers are paying in more each month, "
                   "giving it longer, or adjusting the goal. Try changing them in the sidebar.")
    st.caption("Based on 10,000 simulated futures with monthly returns drawn from a normal distribution. "
               "Real markets have more extreme falls than this model assumes.")
# ---------- Tab 4: plain-English client summary ----------
paid_in = starting_amount + monthly_contribution * years_to_goal * 12

summary = f"""
**Your recommended portfolio: {e.risk_labels[level]} (level {level} of 5)**

{note}

**What it holds:** {equities}% in shares (equities), {bonds}% in bonds and {cash}% in cash.
Shares give the best chance of growth but can fall sharply; bonds and cash are steadier.

**What to expect:** an average return of about **{exp_ret:.1%} a year**, but not every year will be average.
In a typical year, the value could move up or down by around **{vol:.1%}**, and in a bad year it could fall by more.

**Your goal:** £{goal_amount:,} in {years_to_goal} years, paying in £{monthly_contribution:,} a month
on top of £{starting_amount:,} today (£{paid_in:,} in total).

- Across 10,000 simulated futures, you reach your goal in **{prob:.0%}** of them.
- In a middle outcome you'd have about **£{p50:,.0f}**. In a poor one (bottom 10%), about **£{p10:,.0f}**,
  and in a good one (top 10%), about **£{p90:,.0f}**.

If your chance of reaching the goal is low, the main levers are paying in more each month, giving it longer,
or adjusting the goal. Taking more risk is only an option if you can afford the losses that come with it.

*This is an educational tool, not financial advice. Projections use J.P. Morgan's 2027 long-term capital market
assumptions and assume returns follow a normal distribution, so real outcomes, especially in a crash, can be worse.*
"""

with tab4:
    st.markdown(summary)
    st.download_button("Download summary", summary, file_name=f"{name.replace(' ', '_')}_summary.md")
