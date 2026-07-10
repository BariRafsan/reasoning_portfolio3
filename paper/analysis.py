"""Post-training analysis for the P3.2 paper: runs greedy evaluation on every
trained agent, exports strategy charts, computes state-action space stats,
diffs strategy charts across rulesets, and writes:
  - results/summary.json         (all numeric results used in the paper)
  - paper/figs/*.png             (learning curves, profit bars, state growth)
Run this AFTER run_all_scenarios.sh has finished (needs logs/*.agent.pkl).
"""
import csv
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blackjack_rl.evaluate import evaluate, export_strategy_chart
from blackjack_rl.rules import RULESETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

RUNS = [
    "basic_mc_base_seed0",
    "basic_qlearning_base_seed0",
    "counting_mc_base_seed0",
    "counting_qlearning_base_seed0",
    "basic_mc_dealer_stands_soft17_seed0",
    "counting_mc_blackjack_6to5_seed0",
    "improved_mc_base_seed0",
    "improved_qlearning_base_seed0",
]

EVAL_EPISODES = 2_000_000
EVAL_SEED = 999


def read_progress(run_name):
    path = os.path.join(LOGS, f"{run_name}.progress.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def theoretical_state_action_count(use_counting, actions_per_state=5):
    # hand kinds: hard totals 4-20 (17), soft totals 12-20 (9... practically 13-20 since min soft is A+2=13),
    # we just count what's representable; pairs: 10 ranks.
    hard_totals = list(range(4, 21))          # 4..20 (21 is terminal/blackjack, excluded from decisions)
    soft_totals = list(range(13, 21))         # A+2=13 .. 20
    pair_ranks = 10
    dealer_upcards = 10
    hand_kinds = len(hard_totals) + len(soft_totals) + pair_ranks
    tc_buckets = 13 if use_counting else 1    # [-6,6]
    states = hand_kinds * dealer_upcards * tc_buckets
    return states, states * actions_per_state


def main():
    summary = {"runs": {}, "space": {}, "chart_diffs": {}}

    # --- Per-run: eval stats + state/space info ---
    for run in RUNS:
        agent_path = os.path.join(LOGS, f"{run}.agent.pkl")
        if not os.path.exists(agent_path):
            print(f"[skip] {agent_path} not found")
            continue
        with open(agent_path, "rb") as f:
            blob = pickle.load(f)
        scenario = blob["scenario"]
        chart_path = os.path.join(RESULTS, f"{run}.chart.csv")
        mean, ci95 = evaluate(agent_path, EVAL_EPISODES, EVAL_SEED, export_chart=chart_path)

        agent = blob["agent"]
        use_counting = scenario in ("counting", "improved")
        theo_states, theo_sa = theoretical_state_action_count(use_counting)
        learned_states = len(agent.Q)
        learned_sa = agent.state_action_count()

        low_visit_states = 0
        total_states = 0
        if os.path.exists(chart_path):
            with open(chart_path) as f:
                for row in csv.DictReader(f):
                    total_states += 1
                    if int(row["visits"]) < 30:
                        low_visit_states += 1

        summary["runs"][run] = {
            "scenario": scenario,
            "algo": blob["algo"],
            "ruleset": blob["ruleset"],
            "trained_episodes": blob["episodes"],
            "mean_profit_per_hand": mean,
            "ci95": ci95,
            "mean_profit_per_100": 100 * mean,
            "learned_states": learned_states,
            "learned_state_actions": learned_sa,
            "theoretical_states": theo_states,
            "theoretical_state_actions": theo_sa,
            "coverage_frac": learned_states / theo_states,
            "low_visit_states": low_visit_states,
            "low_visit_frac": (low_visit_states / total_states) if total_states else 0.0,
        }
        print(f"{run}: mean={mean:+.5f} ci95={ci95:.5f} states={learned_states}/{theo_states}")

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote results/summary.json")

    # --- Figure 1: learning curves, MC vs Q-learning, basic scenario ---
    plot_learning_curve_pair(
        "basic_mc_base_seed0", "basic_qlearning_base_seed0",
        "Monte Carlo", "Q-learning",
        "Scenario 1 (Basic Strategy): learning curves",
        os.path.join(FIGS, "fig_basic_algo_compare.png"),
    )
    plot_learning_curve_pair(
        "counting_mc_base_seed0", "counting_qlearning_base_seed0",
        "Monte Carlo", "Q-learning",
        "Scenario 2 (Point-Count): learning curves",
        os.path.join(FIGS, "fig_counting_algo_compare.png"),
    )

    # --- Figure 2: state-table growth (stability of Q estimates) ---
    plot_state_growth(
        ["basic_mc_base_seed0", "counting_mc_base_seed0", "improved_mc_base_seed0"],
        ["Scenario 1 (basic)", "Scenario 2 (counting)", "Scenario 4 (improved)"],
        os.path.join(FIGS, "fig_state_growth.png"),
    )

    # --- Figure 3: profit-per-100-hands bar chart with 95% CI across all scenarios ---
    plot_profit_bars(summary, os.path.join(FIGS, "fig_profit_bars.png"))

    # --- Strategy-chart diffs: base vs each rule variant ---
    diff_a = diff_charts(
        os.path.join(RESULTS, "basic_mc_base_seed0.chart.csv"),
        os.path.join(RESULTS, "basic_mc_dealer_stands_soft17_seed0.chart.csv"),
    )
    diff_b = diff_charts(
        os.path.join(RESULTS, "counting_mc_base_seed0.chart.csv"),
        os.path.join(RESULTS, "counting_mc_blackjack_6to5_seed0.chart.csv"),
    )
    summary["chart_diffs"]["basic_vs_dealer_stands_soft17"] = diff_a
    summary["chart_diffs"]["counting_vs_blackjack_6to5"] = diff_b

    # --- Correctness check: learned basic-strategy hard totals vs Thorp ---
    basic_chart = os.path.join(RESULTS, "basic_mc_base_seed0.chart.csv")
    agree, total, mismatches = check_against_thorp(basic_chart)
    summary["thorp_check"] = {
        "agree": agree, "total": total,
        "agree_frac": agree / total if total else 0.0,
        "mismatches": mismatches,
    }
    print(f"Thorp hard-strategy agreement: {agree}/{total} ({100*agree/total:.1f}%)")
    for m in mismatches:
        print("  mismatch:", m)

    plot_strategy_heatmap(basic_chart, "Learned policy: Scenario 1 (Basic Strategy)",
                           os.path.join(FIGS, "fig_strategy_heatmap_basic.png"))
    counting_chart = os.path.join(RESULTS, "counting_mc_base_seed0.chart.csv")
    if os.path.exists(counting_chart):
        plot_strategy_heatmap(counting_chart, "Learned policy: Scenario 2 (Point-Count, true count = 0)",
                               os.path.join(FIGS, "fig_strategy_heatmap_counting.png"), tc_filter=0)

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"dealer_stands_soft17 diff: {len(diff_a)} changed cells")
    print(f"blackjack_6to5 diff: {len(diff_b)} changed cells")


def plot_learning_curve_pair(run_a, run_b, label_a, label_b, title, out_path):
    rows_a = read_progress(run_a)
    rows_b = read_progress(run_b)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([r["episode"] / 1e6 for r in rows_a], [r["avg_profit_per_hand"] for r in rows_a],
            label=label_a, linewidth=1.4)
    ax.plot([r["episode"] / 1e6 for r in rows_b], [r["avg_profit_per_hand"] for r in rows_b],
            label=label_b, linewidth=1.4)
    ax.set_xlabel("Training episodes (millions)")
    ax.set_ylabel("Avg. profit / hand (per log window)")
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_state_growth(runs, labels, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for run, label in zip(runs, labels):
        rows = read_progress(run)
        ax.plot([r["episode"] / 1e6 for r in rows], [r["num_states"] for r in rows],
                label=label, linewidth=1.4)
    ax.set_xlabel("Training episodes (millions)")
    ax.set_ylabel("Distinct states visited")
    ax.set_title("State-space discovery over training", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_profit_bars(summary, out_path):
    order = [
        ("basic_mc_base_seed0", "Basic\n(MC)"),
        ("basic_qlearning_base_seed0", "Basic\n(Q-learn)"),
        ("counting_mc_base_seed0", "Counting\n(MC)"),
        ("counting_qlearning_base_seed0", "Counting\n(Q-learn)"),
        ("basic_mc_dealer_stands_soft17_seed0", "Basic\n(soft17)"),
        ("counting_mc_blackjack_6to5_seed0", "Counting\n(6:5 BJ)"),
        ("improved_mc_base_seed0", "Improved\n(MC)"),
        ("improved_qlearning_base_seed0", "Improved\n(Q-learn)"),
    ]
    names, vals, errs = [], [], []
    for run, label in order:
        if run not in summary["runs"]:
            continue
        r = summary["runs"][run]
        names.append(label)
        vals.append(r["mean_profit_per_100"])
        errs.append(100 * r["ci95"])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = range(len(names))
    ax.bar(x, vals, yerr=errs, capsize=3, color="#4C72B0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Profit / 100 hands (95% CI)")
    ax.set_title("Greedy-evaluation profit across scenarios", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# Canonical published basic-strategy actions for HARD player totals only
# (multi-deck, dealer hits soft 17, no surrender, DAS) -- from Thorp/standard
# strategy tables -- used as an independent correctness check on the learned
# policy, not as training data.
THORP_HARD_STRATEGY = {
    # total: {dealer_upcard_value: action}, dealer 11 == Ace
    9:  {u: ("DOUBLE" if u in (3, 4, 5, 6) else "HIT") for u in range(2, 12)},
    10: {u: ("DOUBLE" if u in range(2, 10) else "HIT") for u in range(2, 12)},
    11: {u: ("DOUBLE" if u in range(2, 11) else "HIT") for u in range(2, 12)},
    12: {u: ("STAND" if u in (4, 5, 6) else "HIT") for u in range(2, 12)},
    13: {u: ("STAND" if u in range(2, 7) else "HIT") for u in range(2, 12)},
    14: {u: ("STAND" if u in range(2, 7) else "HIT") for u in range(2, 12)},
    15: {u: ("STAND" if u in range(2, 7) else "HIT") for u in range(2, 12)},
    16: {u: ("STAND" if u in range(2, 7) else "HIT") for u in range(2, 12)},
    17: {u: "STAND" for u in range(2, 12)},
    18: {u: "STAND" for u in range(2, 12)},
    19: {u: "STAND" for u in range(2, 12)},
    20: {u: "STAND" for u in range(2, 12)},
}
for _t in range(4, 9):
    THORP_HARD_STRATEGY[_t] = {u: "HIT" for u in range(2, 12)}


def check_against_thorp(chart_path):
    """Compares the learned greedy policy's HARD-total actions (dealer
    upcard as int, 11=Ace) against THORP_HARD_STRATEGY. Returns
    (agree, total, mismatches)."""
    agree, total, mismatches = 0, 0, []
    with open(chart_path) as f:
        for row in csv.DictReader(f):
            if row["hand_kind"] != "hard":
                continue
            total_val = int(float(row["hand_value"]))
            if total_val not in THORP_HARD_STRATEGY:
                continue
            dealer_raw = row["dealer_upcard"]
            dealer = 11 if dealer_raw in ("11", "A") else int(float(dealer_raw))
            ref = THORP_HARD_STRATEGY[total_val].get(dealer)
            if ref is None:
                continue
            learned = row["best_action"]
            # DOUBLE only meaningfully comparable when it was a legal first
            # action; if the agent never had DOUBLE available for this state
            # (post-hit revisits) it will show HIT/STAND instead -- treat
            # DOUBLE-vs-HIT/STAND on the *same side* of the decision loosely
            # by comparing directly (both scenario ref and learned chart are
            # first-action greedy choices, since that's what dominates visits).
            total += 1
            if learned == ref:
                agree += 1
            else:
                mismatches.append({
                    "hard_total": total_val, "dealer_upcard": dealer,
                    "thorp": ref, "learned": learned,
                    "visits": int(row["visits"]),
                })
    return agree, total, mismatches


def plot_strategy_heatmap(chart_path, title, out_path, tc_filter=None):
    action_code = {"STAND": 0, "HIT": 1, "DOUBLE": 2, "SURRENDER": 3}
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"]
    totals = list(range(4, 21))
    dealer_vals = list(range(2, 12))  # 11 = Ace
    grid = [[None] * len(dealer_vals) for _ in totals]
    with open(chart_path) as f:
        for row in csv.DictReader(f):
            if row["hand_kind"] != "hard":
                continue
            if tc_filter is not None and row["true_count_bucket"] != str(tc_filter):
                continue
            t = int(float(row["hand_value"]))
            if t not in totals:
                continue
            dealer_raw = row["dealer_upcard"]
            d = 11 if dealer_raw in ("11", "A") else int(float(dealer_raw))
            if d not in dealer_vals:
                continue
            grid[totals.index(t)][dealer_vals.index(d)] = row["best_action"]

    import numpy as np
    mat = np.full((len(totals), len(dealer_vals)), -1)
    for i, row in enumerate(grid):
        for j, act in enumerate(row):
            if act in action_code:
                mat[i, j] = action_code[act]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(5.5, 6.0))
    masked = np.ma.masked_where(mat < 0, mat)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(dealer_vals)))
    ax.set_xticklabels(["A" if v == 11 else str(v) for v in dealer_vals])
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels(totals)
    ax.set_xlabel("Dealer upcard")
    ax.set_ylabel("Player hard total")
    ax.set_title(title, fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, ["Stand", "Hit", "Double", "Surrender"],
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def diff_charts(path_a, path_b):
    def load(path):
        d = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["hand_kind"], row["hand_value"], row["dealer_upcard"], row["true_count_bucket"])
                d[key] = row["best_action"]
        return d

    a = load(path_a)
    b = load(path_b)
    diffs = []
    for key in sorted(set(a) & set(b)):
        if a[key] != b[key]:
            diffs.append({"state": key, "base_action": a[key], "variant_action": b[key]})
    return diffs


if __name__ == "__main__":
    main()
