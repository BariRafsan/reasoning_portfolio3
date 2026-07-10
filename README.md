# Reinforcement-Learning Blackjack Player — Portfolio Exam 3

Self-learning Blackjack player implemented from scratch in Python (standard
library only — no external RL framework), covering the four scenarios
required by Task P3.1.

## Requirements

- Python 3.9+. The core RL implementation (`blackjack_rl/`) is standard
  library only, per the assignment ("Do not use an external Reinforcement
  Learning framework") — training and evaluating agents needs no
  third-party dependency at all.
- `matplotlib`/`numpy` are only needed to run `paper/analysis.py` (the
  greedy-eval report + figures behind the P3.2 paper), not for Task P3.1.

## Installing (Poetry)

```bash
poetry install                # core package only (blackjack_rl/, stdlib-only)
poetry install --with paper   # + matplotlib/numpy, needed for paper/analysis.py
poetry run python -m blackjack_rl.train --scenario basic --algo mc --episodes 5000000
poetry run python paper/analysis.py
```

No `pyproject.toml` changes are needed to reproduce results — `poetry install`
resolves the exact versions pinned in `poetry.lock`. Without Poetry, any
Python 3.9+ interpreter works for `blackjack_rl/` directly (`python -m
blackjack_rl.train ...`); install `matplotlib`/`numpy` yourself to run
`paper/analysis.py`.

## Project layout

```
blackjack_rl/
  cards.py     Shoe, card dealing, Hi-Lo running/true count
  rules.py     Rule configurations (base ruleset + named variants)
  env.py       Round simulation: player decisions (incl. splits), dealer
               play, settlement, state encoding
  agents.py    MCAgent (Monte Carlo control), QLearningAgent (tabular
               Q-learning), BetAgent (contextual bandit for bet sizing)
  train.py     CLI: trains an agent for a given scenario/ruleset/algorithm
  evaluate.py  CLI: greedy (epsilon=0) evaluation + strategy-chart export
logs/          training progress CSVs + pickled trained agents (git-worthy
               log files for the Task P3.1 deliverable)
results/       evaluation outputs / exported strategy charts
paper/         Task P3.2 deliverable: IEEEtran paper.tex, the analysis
               pipeline that turns logs/+results/ into the paper's numbers
               and figures, and the compiled paper.pdf
run_all_scenarios.sh
               produces every log file behind the paper in one shot
```

## How the four scenarios map to the code

| Assignment scenario | How to produce it |
|---|---|
| 1. Basic Strategy | `--scenario basic` (no card counting, flat bet) |
| 2. Complete Point-Count System + bet adjustment | `--scenario counting` (Hi-Lo true count is part of the state; a bandit agent learns bet size per true-count bucket) |
| 3. Two rule variations | rerun scenario 1 and/or 2 with `--ruleset <name>` (see `rules.py`: `dealer_stands_soft17`, `blackjack_6to5`, `surrender_allowed`, `single_deck`) |
| 4. Improve scenario 2 | `--scenario improved` (wider bet spread **and** a "wonging" action — sit out rounds when the true count is unfavorable — bet size 0 is a legal action) |

## Running training

```bash
# Scenario 1 — Basic Strategy, Monte Carlo control
python -m blackjack_rl.train --scenario basic --algo mc --episodes 5000000

# Scenario 1 — same, but Q-learning (for the algorithm-comparison section of the paper)
python -m blackjack_rl.train --scenario basic --algo qlearning --episodes 5000000

# Scenario 2 — Complete Point-Count System + bet sizing
python -m blackjack_rl.train --scenario counting --algo mc --episodes 10000000

# Scenario 3 — rule variations (repeat for each ruleset x each of scenario 1/2)
python -m blackjack_rl.train --scenario basic    --ruleset dealer_stands_soft17 --episodes 5000000
python -m blackjack_rl.train --scenario counting --ruleset blackjack_6to5       --episodes 10000000

# Scenario 4 — improved counting system
python -m blackjack_rl.train --scenario improved --algo mc --episodes 10000000
```

Each run writes `logs/<scenario>_<algo>_<ruleset>_seed<seed>.progress.csv`
(learning curve: avg profit/hand, epsilon, state-table size over training)
and a pickled agent `....agent.pkl`. Episode counts above are a starting
point — watch the printed `avg_profit` and `states` columns for a run to
flatten out before trusting the result; increase `--episodes` if it hasn't.

## Running greedy evaluation

```bash
python -m blackjack_rl.evaluate --agent logs/basic_mc_base_seed0.agent.pkl \
    --episodes 2000000 --export-chart results/basic_chart.csv
```

This freezes the policy (epsilon=0), plays a large number of independent
hands, and reports mean profit/hand with a 95% confidence interval — this
is the "greedy evaluation" the assignment asks for, and the CI doubles as
evidence the learned profit estimate is statistically meaningful (see the
paper's "can you show your learned profit estimates are correct?" question).
`--export-chart` dumps the learned greedy policy as a CSV (hand type,
dealer upcard, true-count bucket -> best action), which is convenient for
producing a basic-strategy-chart figure/table in the paper and for sanity-
checking against Thorp's published tables.

## Reproducing the paper end-to-end

```bash
bash run_all_scenarios.sh            # ~15-20 min on a laptop, writes logs/
python3 paper/analysis.py            # greedy-evaluates every agent, writes
                                      # results/summary.json + paper/figs/*.png
python3 paper/gen_tex.py             # turns summary.json into paper/results_include.tex
tectonic paper/paper.tex             # compiles paper/paper.pdf (needs tectonic,
                                      # see https://tectonic-typesetting.github.io)
```

`paper/analysis.py` is also where the four scientific questions from the
assignment (algorithm comparison, state-action space size / Q-estimate
stability, rule-change effects, profit-estimate correctness) are computed
from real numbers rather than asserted — it cross-checks the learned
Scenario-1 policy's hard-total decisions against Thorp's published basic
strategy table as an independent correctness check.

## Design notes / simplifications (for the paper's methodology section)

- **State space**: a player-hand state is `(hand_kind, dealer_upcard)` for
  non-counting scenarios, or `(hand_kind, dealer_upcard, true_count_bucket)`
  when counting is enabled. `hand_kind` is one of `('hard', total)`,
  `('soft', total)`, or `('pair', rank)`. All ten-valued cards (10/J/Q/K)
  share one rank label, so there are 10 pair classes rather than 13 — this
  matches how basic-strategy engines and charts are usually built and keeps
  the state space tractable.
- **True count** is bucketed to an integer in `[-6, 6]` to bound the state
  space; card visibility to the count is tracked explicitly (`Shoe.reveal`)
  so the dealer's hole card only affects the count once it is actually
  turned over — the agent never conditions on information it couldn't see.
- **Splits** are treated as independent sub-episodes: each split hand gets
  its own trajectory and terminal reward (profit per unit of the base bet),
  which keeps the Monte Carlo / Q-learning update well-defined even though
  Blackjack only provides a reward at the end of a hand. Split aces get one
  card each and auto-stand (standard casino rule; configurable).
- **Bet sizing (scenario 2 and 4)** is handled by `KellyBetAgent`
  (`agents.py`), not a per-count bandit. An earlier version (`BetAgent`,
  kept in `agents.py` and still unpicklable for reference; its trained
  logs are archived under `logs/naive_bandit_baseline/` and
  `results/naive_bandit_baseline/`) treated every true-count bucket as an
  independent multi-armed bandit and greedily argmaxed the *observed* money
  profit per (bucket, bet-size) pair. That fails in a specific, checkable
  way: money_profit = bet_size × profit_per_unit at a fixed count, so there
  is nothing to *discover* about the effect of bet size once profit-per-unit
  is known — treating it as an exploration problem only adds variance.
  Extreme counts are also rare (a couple thousand hands out of 25M), so each
  bucket's own average is dominated by sampling noise on the order of the
  true edge itself. Empirically, every bucket — including the most
  favorable ones — ended up looking unprofitable at every positive bet size,
  so the trained naive bandit bets the table minimum forever in scenario 2
  and sits out (bet 0) forever in scenario 4, regardless of the count
  (verifiable by inspecting `logs/naive_bandit_baseline/*.agent.pkl`: every
  `bet_agent.Q[tc][b]` is negative for every `tc` from -6 to +6). That is a
  corner solution produced by estimator noise, not a real betting policy,
  and it is why the original scenario-4 greedy evaluation reported exactly
  0.0 profit with 0.0 variance.
  `KellyBetAgent` fixes this by fitting one pooled OLS regression of
  profit-per-unit-bet on true count (`EdgeModel`) across *all* hands
  (including 0-bet rounds — the edge model observes every hand's outcome
  independent of what was staked on it), instead of one independent
  estimate per bucket. Pooling makes the fit at rare, extreme counts as
  reliable as the regression's total sample size rather than only as
  reliable as that bucket's own few thousand hands. It sits out (scenario 4
  only) when the *lower confidence bound* of the fitted edge is non-positive
  rather than just its point estimate, which is what keeps sampling noise
  from producing a false "always sit out" verdict. Bet size scales
  Kelly-proportionally: full-Kelly (edge/variance) is an optimal bankroll
  *fraction*, not a count of table-minimum units, so it can't be plugged
  directly into a 1..12-unit ladder without underflowing to the minimum —
  instead the top of the ladder is calibrated to the edge at the most
  favorable true count the system represents (+6, the clip bound in
  `bucket_true_count`), and every other count's bet scales linearly between
  the table minimum and that top rung by its edge relative to the reference
  count's edge (variance is roughly constant across counts, so this ratio
  equals the Kelly ratio). The playing agent also uses a tighter epsilon
  floor (0.002 vs. 0.01) and the bet agent stays idle until epsilon has
  decayed close to it (`BET_BURN_IN_EPSILON` in `train.py`), since the edge
  model needs profit samples from an (almost) purely greedy policy — a
  residual 1% random-action rate is a real, permanent per-hand cost that
  would otherwise bias `edge_hat(tc)` downward for the entire, much longer,
  post-burn-in data-collection phase.
- **Scenario 4 improvement** is the wider bet ladder itself (0/1/2/4/8/12
  units, vs. scenario 2's 1/2/3/4/6/8 with no sit-out rung): a real 0 =
  "wong out" action plus a bigger top bet, on top of the `KellyBetAgent`
  fix above, raises long-run profit per hour by skipping negative-EV counts
  entirely and betting more aggressively at the best ones, rather than
  flat-betting the table minimum at every count the way scenario 2's ladder
  requires.
- **Algorithm comparison**: `MCAgent` uses every-visit Monte Carlo control
  (reward known only at hand resolution, so every state-action pair visited
  in a hand is updated toward that hand's final profit). `QLearningAgent`
  bootstraps with the standard Q-learning target; because Blackjack hands
  are short (1-6 decisions), its updates are applied once a hand resolves,
  replayed over that hand's own trajectory — action *selection* during play
  always uses the live Q-table (so exploration behaves correctly), but this
  means Q-learning here is closer to "semi-online" than a textbook one-step
  online update. This is worth naming explicitly as a design choice in the
  paper rather than glossing over it.
- **No external RL framework** is used anywhere — `agents.py` implements
  the incremental-average / TD update rules directly on plain dicts.

## Suggested workflow before writing the paper

1. Train scenario 1 with both `mc` and `qlearning` on `base` rules; compare
   learning curves (`*.progress.csv`) and final greedy profit — this is
   your algorithm-comparison evidence.
2. Train scenario 2 (`counting`, `mc`) — compare greedy profit against
   scenario 1's. Export its strategy chart and check for count-dependent
   deviations from the scenario-1 chart at extreme true counts (these are
   the "index plays" advantage players use in practice).
3. Train scenario 3: rerun 1 and/or 2 under 2 rulesets from `rules.py`.
   Diff the exported strategy charts to explain *why* each rule change
   moved the policy (e.g. dealer-stands-on-soft-17 should make the player
   stand more often on soft totals near dealer weak/strong upcards).
4. Train scenario 4 (`improved`) and compare its greedy profit and its
   variance (std. dev. of per-hand profit) against scenario 2 — the paper
   should show both point estimates *and* the confidence intervals from
   `evaluate.py`.
5. For "estimate the size of the state-action space": count states from
   `len(agent.Q)` (printed by evaluate.py) versus the theoretical count
   (`hand_kinds × dealer_upcards × tc_buckets × actions`), and discuss
   which states get too few visits (`visits` column in the exported chart)
   to trust their Q estimate — this is real data for the "can one expect
   stable Q(., .) estimates" question in the paper.
