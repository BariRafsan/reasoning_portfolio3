#!/usr/bin/env bash
# Produces the log files / trained agents for Task P3.1's four scenarios.
# Episode counts here are a reasonable starting point (fast enough to run
# on a laptop); bump them up (see README.md) for tighter convergence on
# marginal-EV states before finalizing paper numbers.
set -e
cd "$(dirname "$0")"

# Use `poetry run python3` when Poetry is available and this is a Poetry
# project (matches pyproject.toml); otherwise fall back to a plain python3
# on PATH. The core blackjack_rl package has no runtime dependencies either
# way, so both paths work identically for training.
if command -v poetry >/dev/null 2>&1 && [ -f pyproject.toml ]; then
  PY="poetry run python3"
else
  PY="python3"
fi

# Scenario 1: Basic Strategy (two algorithms, for the paper's algorithm comparison)
$PY -m blackjack_rl.train --scenario basic --algo mc        --ruleset base --episodes 15000000 --log-every 1000000 --seed 0
$PY -m blackjack_rl.train --scenario basic --algo qlearning --ruleset base --episodes 15000000 --log-every 1000000 --seed 0

# Scenario 2: Complete Point-Count System + bet sizing (two algorithms too,
# so the algorithm-comparison section of the paper isn't limited to scenario 1)
$PY -m blackjack_rl.train --scenario counting --algo mc        --ruleset base --episodes 25000000 --log-every 2000000 --seed 0
$PY -m blackjack_rl.train --scenario counting --algo qlearning --ruleset base --episodes 25000000 --log-every 2000000 --seed 0

# Scenario 3: two rule variations, applied to both scenario 1 and 2
$PY -m blackjack_rl.train --scenario basic    --algo mc --ruleset dealer_stands_soft17 --episodes 15000000 --log-every 1000000 --seed 0
$PY -m blackjack_rl.train --scenario counting --algo mc --ruleset blackjack_6to5        --episodes 25000000 --log-every 2000000 --seed 0

# Scenario 4: improved counting system (wider bet spread + wonging)
$PY -m blackjack_rl.train --scenario improved --algo mc        --ruleset base --episodes 25000000 --log-every 2000000 --seed 0
$PY -m blackjack_rl.train --scenario improved --algo qlearning --ruleset base --episodes 25000000 --log-every 2000000 --seed 0

echo "ALL TRAINING RUNS COMPLETE"
