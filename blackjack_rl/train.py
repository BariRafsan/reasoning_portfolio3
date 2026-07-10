import argparse
import csv
import os
import pickle
import random
import time

from .agents import MCAgent, QLearningAgent, KellyBetAgent
from .cards import Shoe
from .env import play_round, bucket_true_count
from .rules import RULESETS

BET_SIZES_SCENARIO2 = (1, 2, 3, 4, 6, 8)
BET_SIZES_IMPROVED = (0, 1, 2, 4, 8, 12)  # 0 = "wong out" / sit out the round

BET_BURN_IN_EPSILON = 0.02


def make_agent(algo, seed, epsilon_min=0.01):
    rng = random.Random(seed)
    if algo == "mc":
        return MCAgent(rng=rng, epsilon_min=epsilon_min)
    if algo == "qlearning":
        return QLearningAgent(rng=rng, epsilon_min=epsilon_min)
    raise ValueError(f"unknown algo {algo}")


def train(scenario, algo, ruleset_name, episodes, seed, log_every, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rules = RULESETS[ruleset_name]
    rng = random.Random(seed)
    shoe = Shoe(num_decks=rules.num_decks, penetration=rules.penetration, rng=rng)

    use_counting = scenario in ("counting", "improved")
    use_bet_agent = scenario in ("counting", "improved")
    agent = make_agent(algo, seed)
    bet_sizes = BET_SIZES_IMPROVED if scenario == "improved" else BET_SIZES_SCENARIO2
    bet_agent = (
        KellyBetAgent(bet_sizes=bet_sizes, rng=random.Random(seed + 1))
        if use_bet_agent
        else None
    )

    run_name = f"{scenario}_{algo}_{ruleset_name}_seed{seed}"
    progress_path = os.path.join(out_dir, f"{run_name}.progress.csv")
    agent_path = os.path.join(out_dir, f"{run_name}.agent.pkl")

    progress_rows = []
    window = []
    t0 = time.time()

    for ep in range(1, episodes + 1):
        shoe.reshuffle_if_needed()

        # The bet agent stays off until the playing policy has (mostly)
        # converged (agent.epsilon has decayed close to its floor).
        bet_ready = use_bet_agent and agent.epsilon <= BET_BURN_IN_EPSILON

        tc_bucket = bucket_true_count(shoe.true_count()) if use_counting else None
        bet_size = bet_agent.choose_bet(tc_bucket) if bet_ready else 1

        # The round is always dealt even when wonging (bet_size == 0)
        # because at a real table the shoe keeps moving whether or not this
        # player is betting (other players / the dealer still consume
        # cards).
        hand_results, profit_units = play_round(
            shoe,
            rules,
            agent,
            get_tc_bucket_fn=True if use_counting else None,
        )
        for h in hand_results:
            agent.finish_hand(h["trajectory"], h["profit"])

        if bet_ready:
            bet_agent.observe(tc_bucket, profit_units)

        wonged_out = bet_ready and bet_size == 0
        if not wonged_out:
            money_profit = profit_units * max(bet_size, 1)
            window.append(money_profit)

        agent.decay_epsilon()
        if use_bet_agent:
            bet_agent.decay_epsilon()

        if ep % log_every == 0:
            avg = sum(window) / len(window) if window else 0.0
            elapsed = time.time() - t0
            progress_rows.append(
                {
                    "episode": ep,
                    "avg_profit_per_hand": avg,
                    "epsilon": agent.epsilon,
                    "num_states": len(agent.Q),
                    "num_state_actions": agent.state_action_count(),
                    "elapsed_sec": round(elapsed, 1),
                }
            )
            print(
                f"[{run_name}] ep={ep:,} avg_profit={avg:+.4f} eps={agent.epsilon:.4f} "
                f"states={len(agent.Q):,} t={elapsed:.0f}s"
            )
            window = []

    with open(progress_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                list(progress_rows[0].keys())
                if progress_rows
                else [
                    "episode",
                    "avg_profit_per_hand",
                    "epsilon",
                    "num_states",
                    "num_state_actions",
                    "elapsed_sec",
                ]
            ),
        )
        writer.writeheader()
        writer.writerows(progress_rows)

    with open(agent_path, "wb") as f:
        pickle.dump(
            {
                "scenario": scenario,
                "algo": algo,
                "ruleset": ruleset_name,
                "episodes": episodes,
                "seed": seed,
                "agent": agent,
                "bet_agent": bet_agent,
            },
            f,
        )

    print(f"Saved progress log -> {progress_path}")
    print(f"Saved trained agent -> {agent_path}")
    return agent_path


def main():
    p = argparse.ArgumentParser(description="Train a from-scratch RL Blackjack agent.")
    p.add_argument(
        "--scenario", choices=["basic", "counting", "improved"], required=True
    )
    p.add_argument("--algo", choices=["mc", "qlearning"], default="mc")
    p.add_argument("--ruleset", choices=list(RULESETS.keys()), default="base")
    p.add_argument("--episodes", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100_000)
    p.add_argument("--out-dir", default="logs")
    args = p.parse_args()

    train(
        args.scenario,
        args.algo,
        args.ruleset,
        args.episodes,
        args.seed,
        args.log_every,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
