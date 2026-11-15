# engine.py: all the calculations for the Client Portfolio Builder.
# Educational tool only. Not financial advice.

# ---------- 1. Imports (all at the top) ----------
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ---------- 2. Fixed data ----------
asset_classes = ["UK equity", "Global equity", "Emerging markets", "UK gilts", "Global bonds", "Cash"]

risk_labels = {1: "Cautious", 2: "Moderately cautious", 3: "Balanced", 4: "Moderately adventurous", 5: "Adventurous"}

model_portfolios = {
    1: {"UK equity": 5, "Global equity": 10, "Emerging markets": 0, "UK gilts": 35, "Global bonds": 35, "Cash": 15},
    2: {"UK equity": 10, "Global equity": 20, "Emerging markets": 5, "UK gilts": 25, "Global bonds": 30, "Cash": 10},
    3: {"UK equity": 15, "Global equity": 35, "Emerging markets": 5, "UK gilts": 20, "Global bonds": 20, "Cash": 5},
    4: {"UK equity": 15, "Global equity": 50, "Emerging markets": 10, "UK gilts": 10, "Global bonds": 12, "Cash": 3},
    5: {"UK equity": 20, "Global equity": 60, "Emerging markets": 15, "UK gilts": 0, "Global bonds": 5, "Cash": 0},
}

asset_colours = {
    "UK equity": "#1f3b73",
    "Global equity": "#3a6fb0",
    "Emerging markets": "#8fb3e0",
    "UK gilts": "#2e7d6b",
    "Global bonds": "#7cc4a8",
    "Cash": "#b8b8b8",
}

# ---------- 3. Questionnaire (NEW: you write these) ----------
AGREE_SCALE = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]   # scores 1 to 5

ATR_QUESTIONS = [
    # 8 statements where AGREEING = more willing to take risk, e.g.
    "If my investments fell 20% in a year, I would stay invested rather than sell.",
    "I'm comfortable with my investments rising and falling in value if it means higher potential growth.",
    "I would rather aim for higher long-term returns than keep my money stable.",
    "I have experience of investing in shares or funds.",
    "Compared with people I know, I'm someone who takes financial risks.",
    "I'd be happy to invest in emerging markets, even though they can swing sharply.",
    "A fall in my investments would not cause me to lose sleep.",
    "I'm investing to grow my money rather than to protect what I already have.",
]

CFL_QUESTIONS = [
    # 4 questions, each with 5 options ordered from 1 (can't afford losses) to 5 (can easily afford them), e.g.
    {
        "question": "How many months of essential spending could you cover from savings, without touching this money?",
        "options": ["None", "Less than 1 month", "1 to 3 months", "3 to 6 months", "More than 6 months"],
    },
    {
        "question": "If this investment lost a third of its value, how much would it affect your lifestyle or plans?",
        "options": ["Severely", "Significantly", "Moderately", "A little", "Not at all"],
    },
    {
        "question": "Roughly what share of your total savings and investments is this money?",
        "options": ["Over 75%", "50% to 75%", "25% to 50%", "10% to 25%", "Under 10%"],
    },
    {
        "question": "How likely are you to need this money before your goal date?",
        "options": ["Very likely", "Likely", "Possibly", "Unlikely", "Very unlikely"],
    },
]


# ---------- 4. Risk scoring ----------
def average(answers):
    return sum(answers) / len(answers)

def score_to_level(avg):
    if avg < 1.5:
       return 1
    elif avg < 2.5:
       return 2
    elif avg < 3.5:
       return 3
    elif avg < 4.5:
       return 4
    else:
       return 5

def assess_client(c):
    atr_level = score_to_level(average(c["atr_answers"]))
    cfl_level = score_to_level(average(c["cfl_answers"]))
    final_level = min(atr_level, cfl_level)

    if atr_level > cfl_level:
        note = (f"Your attitude to risk is {risk_labels[atr_level]}, but your capacity "
                f"for loss is {risk_labels[cfl_level]}, so we recommend a "
                f"{risk_labels[final_level]} portfolio to protect you from losses you cannot afford.")
    elif cfl_level > atr_level:
        note = (f"You could afford a {risk_labels[cfl_level]} portfolio, but you prefer "
                f"less risk, so we recommend a {risk_labels[final_level]} portfolio "
                f"in line with your own attitude to risk.")
    else:
        note = f"Your recommended risk level is {risk_labels[final_level]}."

    return final_level, note

# ---------- 5. Portfolio check ----------
def check_portfolios(portfolios):
    all_ok = True
    for level, weights in portfolios.items():
        total = sum(weights.values())
        if total == 100:
            print(f"Level {level} ({risk_labels[level]}): {total}% ✅")
        else:
            print(f"Level {level} ({risk_labels[level]}): {total}% ❌ fix this portfolio")
            all_ok = False
    return all_ok


# ---------- 6. Capital market assumptions ----------
# CHANGED from Task 4: load from the CSV file instead of typing the table in.
assumptions = pd.read_csv(Path(__file__).parent / "assumptions.csv")   # finds the file next to engine.py

corr_values = [
    [ 1.00,  0.85,  0.72,  0.10,  0.03, -0.09],
    [ 0.85,  1.00,  0.79,  0.21,  0.15, -0.05],
    [ 0.72,  0.79,  1.00,  0.11,  0.09,  0.00],
    [ 0.10,  0.21,  0.11,  1.00,  0.85,  0.16],
    [ 0.03,  0.15,  0.09,  0.85,  1.00,  0.24],
    [-0.09, -0.05,  0.00,  0.16,  0.24,  1.00],
]

correlations = pd.DataFrame(corr_values, index=asset_classes, columns=asset_classes)

exp_returns = assumptions["exp_return"].values / 100
vols = assumptions["volatility"].values / 100
cov_matrix = np.outer(vols, vols) * correlations.values
risk_free = exp_returns[-1]

# ---------- 7. Risk and return ----------
def portfolio_stats(level):
    w = np.array([model_portfolios[level][a] for a in asset_classes]) / 100
    exp_ret = w @ exp_returns
    vol = np.sqrt(w @ cov_matrix @ w)
    sharpe = (exp_ret - risk_free) / vol
    return exp_ret, vol, sharpe

# ---------- 8. Monte Carlo ----------
def monte_carlo(c, n_paths = 10_000, seed=42):
    level, note = assess_client(c)
    exp_ret, vol, sharpe = portfolio_stats(level)

    months = c["years_to_goal"] * 12
    monthly_mean = exp_ret / 12 
    monthly_vol = vol / np.sqrt(12)

    rng = np.random.default_rng(seed)
    returns = rng.normal(monthly_mean, monthly_vol, size=(n_paths, months))

    values = np.zeros((n_paths, months + 1))
    values[:, 0] = c["starting_amount"]
    for m in range(months):
        values[:, m + 1] = values[:, m] * (1 + returns[:, m]) + c["monthly_contribution"]

    final = values[:, -1]
    prob = (final >= c["goal_amount"]).mean()
    p10, p50, p90 = np.percentile(final, [10, 50, 90])
    return prob, p10, p50, p90, values


# ---------- 9. Charts ----------
def allocation_donut(level):
    weights = model_portfolios[level]
    labels = [a for a in asset_classes if weights[a] > 0]
    sizes = [weights[a] for a in labels]
    colours = [asset_colours[a] for a in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, pcts = ax.pie(
        sizes,
        colors=colours,
        startangle=90,
        counterclock=False,
        autopct="%1.0f%%",
        pctdistance=0.8,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2}
    )
    for p in pcts:
        p.set_color("white")
        p.set_fontsize(10)
        p.set_fontweight("bold")

    ax.text(0, 0, f"Level {level}\n{risk_labels[level]}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_title("Recommended portfolio allocation", fontsize=14, fontweight="bold")
    ax.axis("equal")
    plt.tight_layout()
    return fig


def risk_return_chart(client_level):
    fig, ax = plt.subplots(figsize=(8, 5))

    # collect every portfolio's risk and return so we can join the dots with a line
    vols, rets = [], []
    for level in model_portfolios:
        r, v, s = portfolio_stats(level)
        vols.append(v * 100)
        rets.append(r * 100)
    ax.plot(vols, rets, color="#b8b8b8", linewidth=1.5, zorder=1)

    for level in model_portfolios:
        r, v, s = portfolio_stats(level)
        x, y = v * 100, r * 100
        if level == client_level:
            ax.scatter(x, y, s=220, color="crimson", edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"Recommended: {risk_labels[level]}\n{y:.1f}% return, {x:.1f}% risk",
                        (x, y), xytext=(12, -28), textcoords="offset points",
                        fontsize=10, fontweight="bold", color="crimson")
        else:
            ax.scatter(x, y, s=90, color="#3a6fb0", edgecolor="white", linewidth=1.5, zorder=2)
            ax.annotate(risk_labels[level], (x, y), xytext=(-10, 10), textcoords="offset points",
                        fontsize=9, color="#1f3b73", ha="right")

    ax.set_xlim(0, max(vols) * 1.2)
    ax.set_ylim(min(rets) - 1, max(rets) + 1)
    ax.set_xlabel("Risk: expected volatility, % per year")
    ax.set_ylabel("Expected return, % per year")
    ax.set_title("Risk vs return of the 5 model portfolios", fontsize=14, fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig



def fan_chart(c):
    prob, p10, p50, p90, values = monte_carlo(c)
    years = np.arange(values.shape[1]) / 12
    low, mid, high = np.percentile(values, [10, 50, 90], axis=0)
    paid_in = c["starting_amount"] + c["monthly_contribution"] * np.arange(values.shape[1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(years, low, high, color="#3a6fb0", alpha=0.2, label="Likely range (10th-90th percentile)")
    ax.plot(years, mid, color="#1f3b73", linewidth=2.5, label="Median outcome")
    ax.plot(years, paid_in, color="#b8b8b8", linewidth=1.5, label="Total paid in")
    ax.axhline(c["goal_amount"], color="crimson", linestyle="--", linewidth=1.5, label=f"Goal: £{c['goal_amount']:,}")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"£{x:,.0f}"))
    ax.set_xlim(0, c["years_to_goal"])
    ax.set_ylim(0, max(c["goal_amount"], high.max()) * 1.15)
    ax.set_xlabel("Years from today")
    ax.set_title(f"{c['name']}: {prob:.0%} chance of reaching the goal", fontsize=14, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
