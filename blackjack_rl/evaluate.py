import argparse
import csv
import math
import pickle
import random

from .cards import Shoe
from .env import play_round, bucket_true_count, Action
from .rules import RULESETS


class _GreedyWrapper:
    """Wraps a trained agent to always act greedily (epsilon=0) without
    mutating the trained agent's exploration schedule."""

    def __init__(self, agent):
        self.agent = agent
        self.Q = agent.Q

    def choose_action(self, state, avail_actions):
        return self.agent.choose_action(state, avail_actions, greedy=True)

    def finish_hand(self, trajectory, profit):
        pass  # no learning during evaluation


def evaluate(agent_path, episodes, seed, export_chart=None):
    with open(agent_path, "rb") as f:
        blob = pickle.load(f)

    scenario = blob["scenario"]
    ruleset_name = blob["ruleset"]
    rules = RULESETS[ruleset_name]
    agent = blob["agent"]
    bet_agent = blob["bet_agent"]
    use_counting = scenario in ("counting", "improved")

    rng = random.Random(seed)
    shoe = Shoe(num_decks=rules.num_decks, penetration=rules.penetration, rng=rng)
    greedy_agent = _GreedyWrapper(agent)

    profits = []
    for _ep in range(episodes):
        shoe.reshuffle_if_needed()
        tc_bucket = bucket_true_count(shoe.true_count()) if use_counting else None
        bet_size = bet_agent.choose_bet(tc_bucket, greedy=True) if bet_agent else 1

        # Always deal the round so the shoe (and true count) keeps moving
        # even on a wong-out; see train.py for why freezing it is wrong.
        _hand_results, profit_units = play_round(
            shoe,
            rules,
            greedy_agent,
            get_tc_bucket_fn=True if use_counting else None,
        )
        if bet_agent and bet_size == 0:
            profits.append(0.0)
        else:
            profits.append(profit_units * max(bet_size, 1))

    n = len(profits)
    mean = sum(profits) / n
    var = sum((x - mean) ** 2 for x in profits) / (n - 1) if n > 1 else 0.0
    stderr = math.sqrt(var / n) if n > 1 else 0.0
    ci95 = 1.96 * stderr

    print(f"Agent: {agent_path}")
    print(
        f"Scenario={scenario} ruleset={ruleset_name} algo={blob['algo']} "
        f"trained_episodes={blob['episodes']:,}"
    )
    print(f"Evaluation hands: {n:,}")
    print(
        f"Mean profit / hand: {mean:+.5f}  (95% CI: [{mean-ci95:+.5f}, {mean+ci95:+.5f}])"
    )
    print(f"Mean profit / 100 hands: {100*mean:+.3f}")
    print(
        f"Learned states: {len(agent.Q):,}  state-actions: {agent.state_action_count():,}"
    )

    if export_chart:
        export_strategy_chart(agent, export_chart, use_counting)
        print(f"Strategy chart exported -> {export_chart}")

    return mean, ci95


def export_strategy_chart(agent, path, use_counting):
    """Dumps the greedy policy as a CSV: hand_kind,hand_value,dealer_up,tc,best_action."""
    rows = []
    for state, qvals in agent.Q.items():
        if use_counting:
            (kind, val), dealer_up, tc = state
        else:
            (kind, val), dealer_up = state
            tc = ""
        best_action = max(qvals, key=qvals.get)
        rows.append(
            {
                "hand_kind": kind,
                "hand_value": val,
                "dealer_upcard": dealer_up,
                "true_count_bucket": tc,
                "best_action": Action(best_action).name,
                "best_action_value": qvals[best_action],
                "visits": sum(agent.N.get((state, a), 0) for a in qvals),
            }
        )
    rows.sort(
        key=lambda r: (
            str(r["hand_kind"]),
            r["hand_value"],
            r["dealer_upcard"],
            r["true_count_bucket"],
        )
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(
        description="Greedy-evaluate a trained Blackjack agent."
    )
    p.add_argument(
        "--agent", required=True, help="path to a .agent.pkl produced by train.py"
    )
    p.add_argument("--episodes", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--export-chart",
        default=None,
        help="optional CSV path for the learned strategy chart",
    )
    args = p.parse_args()
    evaluate(args.agent, args.episodes, args.seed, args.export_chart)


if __name__ == "__main__":
    main()
