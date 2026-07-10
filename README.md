# Reinforcement-Learning Blackjack Player — Portfolio Exam 3

Self-learning Blackjack player

## Requirements

- Python 3.9+ — `blackjack_rl/` is standard library only.
- `matplotlib`/`numpy` — only needed for `paper/analysis.py`.

## Install

```bash
poetry install                # core package (blackjack_rl/), stdlib-only
poetry install --with paper   # + matplotlib/numpy, for paper/analysis.py
```

Without Poetry: any Python 3.9+ works directly for `blackjack_rl/`; `pip
install matplotlib numpy` yourself if you want to run.

## Project layout

```
blackjack_rl/   cards.py, rules.py, env.py, agents.py, train.py, evaluate.py
logs/           training progress CSVs + pickled trained agents
results/        greedy-eval outputs / exported strategy charts
run_all_scenarios.sh   reproduces every log file behind the paper
```

## Scenario -> code

| Scenario | Command |
|---|---|
| 1. Basic Strategy | `--scenario basic` |
| 2. Complete Point-Count + bet sizing | `--scenario counting` |
| 3. Rule variations | add `--ruleset <name>` (see `rules.py`) |
| 4. Improved counting system | `--scenario improved` |

## How to run

Train an agent:
```bash
python -m blackjack_rl.train --scenario basic --algo mc --episodes 5000000
python -m blackjack_rl.train --scenario counting --algo mc --episodes 10000000
python -m blackjack_rl.train --scenario basic --ruleset dealer_stands_soft17 --episodes 5000000
python -m blackjack_rl.train --scenario improved --algo mc --episodes 10000000
```
Writes `logs/<scenario>_<algo>_<ruleset>_seed<seed>.progress.csv` + `....agent.pkl`.

Evaluate a trained agent (greedy, epsilon=0):
```bash
python -m blackjack_rl.evaluate --agent logs/basic_mc_base_seed0.agent.pkl \
    --episodes 2000000 --export-chart results/basic_chart.csv
```

Reproduce everything (training -> analysis):
```bash
bash run_all_scenarios.sh      # ~15-20 min, writes logs/
```
