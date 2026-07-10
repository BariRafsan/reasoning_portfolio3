"""Training entry point.

Examples
--------
Scenario 1 (Basic Strategy), Monte Carlo, base rules:
    python -m blackjack_rl.train --scenario basic --algo mc --episodes 3000000

Scenario 1, Q-learning (for the algorithm comparison in the paper):
    python -m blackjack_rl.train --scenario basic --algo qlearning --episodes 3000000

Scenario 2 (Complete Point-Count System + bet sizing):
    python -m blackjack_rl.train --scenario counting --algo mc --episodes 6000000

Scenario 3 (rule variations) -- reuse scenario 1/2 under a different ruleset:
    python -m blackjack_rl.train --scenario basic --ruleset dealer_stands_soft17 --episodes 3000000
    python -m blackjack_rl.train --scenario counting --ruleset blackjack_6to5 --episodes 6000000

Scenario 4 (improved counting system: wider bet spread + wonging/sit-out):
    python -m blackjack_rl.train --scenario improved --algo mc --episodes 6000000

All runs write:
    logs/<run_name>.progress.csv   - training curve (episode, avg profit/hand, epsilon, table size)
    logs/<run_name>.agent.pkl      - pickled trained agent(s), consumed by evaluate.py
"""
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

# Scenario 2 does not include a sit-out action (no "0" rung): the Complete
# Point-Count System bets the table minimum at neutral/unfavorable counts.
# Scenario 4 adds a wider ladder *and* a 0 = "wong out" rung as the
# improvement. Both scenarios use KellyBetAgent (see agents.py) rather than
# the naive per-bucket bandit -- the naive bandit's greedy bet choice
# degenerates to "always bet minimum" (scenario 2) or "always sit out"
# (scenario 4) once you look at its trained Q-table, because per-bucket
# money-profit noise swamps the true edge at every true count; that failure
# mode (and the trained models exhibiting it) is preserved for the paper in
# logs/naive_bandit_baseline/.
BET_SIZES_SCENARIO2 = (1, 2, 3, 4, 6, 8)
BET_SIZES_IMPROVED = (0, 1, 2, 4, 8, 12)  # 0 = "wong out" / sit out the round

# Bet agent stays idle (flat bet=1, no data collected) until the playing
# agent's epsilon has decayed close to its floor -- see the burn-in comment
# in train() below. This alone (without also touching the playing policy's
# epsilon floor) is enough to keep the bulk of the edge-model's data
# reasonably clean: a first version of this also lowered the playing
# policy's own epsilon_min to 0.002 hoping for an even cleaner edge signal,
# but that measurably *hurt* the learned playing policy itself (checked via
# a flat-bet-1 comparison: -2.66%/hand vs. the original 0.01-floor policy's
# -1.92%/hand) -- the counting state space (4,680 states) still needs
# continued exploration late into a 25M-episode run to keep correcting
# rarely-visited states, and starving that to clean up the bet agent's
# input cost more than it saved. So the playing policy keeps the standard
# epsilon_min=0.01 from BaseAgent, and only the burn-in gate below protects
# the edge model.
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
        if use_bet_agent else None
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
        # converged (agent.epsilon has decayed close to its floor). Feeding
        # the edge-vs-true-count regression from hands played under a
        # still-random, still-learning policy would bias edge_hat toward
        # the much larger negative "exploration-era" house edge rather than
        # the converged basic/counting strategy's true edge -- the model
        # has no way to tell those samples apart from later, trustworthy
        # ones, so it's cheaper to just not collect them.
        bet_ready = use_bet_agent and agent.epsilon <= BET_BURN_IN_EPSILON

        tc_bucket = bucket_true_count(shoe.true_count()) if use_counting else None
        bet_size = bet_agent.choose_bet(tc_bucket) if bet_ready else 1

        # The round is always dealt -- even when wonging (bet_size == 0) --
        # because at a real table the shoe keeps moving whether or not this
        # player is betting (other players / the dealer still consume
        # cards). Freezing the shoe on a sit-out would let the running/true
        # count get stuck, which is both unrealistic and, since the agent
        # only ever observes counts it has already decided to sit out on,
        # self-reinforcing (it can never learn there was anything on the
        # other side of a count it never lets the shoe reach).
        hand_results, profit_units = play_round(
            shoe, rules, agent,
            get_tc_bucket_fn=True if use_counting else None,
        )
        for h in hand_results:
            agent.finish_hand(h["trajectory"], h["profit"])

        if bet_ready:
            # The edge model always observes this hand's profit-per-unit-bet,
            # regardless of the bet actually placed (including 0/wonged-out
            # rounds) -- see KellyBetAgent.observe(). Money profit (what we
            # log/report) is 0 whenever no money was at risk.
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
            progress_rows.append({
                "episode": ep,
                "avg_profit_per_hand": avg,
                "epsilon": agent.epsilon,
                "num_states": len(agent.Q),
                "num_state_actions": agent.state_action_count(),
                "elapsed_sec": round(elapsed, 1),
            })
            print(f"[{run_name}] ep={ep:,} avg_profit={avg:+.4f} eps={agent.epsilon:.4f} "
                  f"states={len(agent.Q):,} t={elapsed:.0f}s")
            window = []

    with open(progress_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(progress_rows[0].keys()) if progress_rows else
                                 ["episode", "avg_profit_per_hand", "epsilon", "num_states",
                                  "num_state_actions", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(progress_rows)

    with open(agent_path, "wb") as f:
        pickle.dump({
            "scenario": scenario,
            "algo": algo,
            "ruleset": ruleset_name,
            "episodes": episodes,
            "seed": seed,
            "agent": agent,
            "bet_agent": bet_agent,
        }, f)

    print(f"Saved progress log -> {progress_path}")
    print(f"Saved trained agent -> {agent_path}")
    return agent_path


def main():
    p = argparse.ArgumentParser(description="Train a from-scratch RL Blackjack agent.")
    p.add_argument("--scenario", choices=["basic", "counting", "improved"], required=True)
    p.add_argument("--algo", choices=["mc", "qlearning"], default="mc")
    p.add_argument("--ruleset", choices=list(RULESETS.keys()), default="base")
    p.add_argument("--episodes", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100_000)
    p.add_argument("--out-dir", default="logs")
    args = p.parse_args()

    train(args.scenario, args.algo, args.ruleset, args.episodes, args.seed,
          args.log_every, args.out_dir)


if __name__ == "__main__":
    main()
