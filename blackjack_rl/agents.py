"""Learning agents. No external RL framework is used — everything here is
plain Python/dict-based tabular RL.

MCAgent          - every-visit Monte Carlo control (primary algorithm).
QLearningAgent   - tabular Q-learning, used as a comparison point in the
                   paper ("how do different learning algorithms behave").
                   Updates are applied once a hand resolves, sequentially
                   over that hand's own trajectory (semi-online: action
                   *selection* during play always uses the live Q-table,
                   but the terminal reward of a hand is only known after
                   dealer settlement, so the very last step's update is
                   necessarily deferred to then). This is documented as a
                   design simplification in the README/paper.
BetAgent         - naive per-count contextual bandit (kept only as the
                   documented "before" baseline: see KellyBetAgent for why
                   it fails and what replaces it for scenario 2/4).
EdgeModel        - online OLS regression of profit-per-unit-bet on true
                   count; the statistical core of KellyBetAgent.
KellyBetAgent    - regression + fractional-Kelly bet sizer used for
                   scenario 2/4 bet sizing (replaces BetAgent).
"""
import random


class BaseAgent:
    def __init__(self, epsilon=0.15, epsilon_min=0.01, epsilon_decay=0.999999, rng=None):
        self.Q = {}
        self.N = {}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or random.Random()

    def _ensure(self, state, avail_actions):
        d = self.Q.setdefault(state, {})
        for a in avail_actions:
            d.setdefault(int(a), 0.0)
        return d

    def choose_action(self, state, avail_actions, greedy=False):
        qvals = self._ensure(state, avail_actions)
        if (not greedy) and self.rng.random() < self.epsilon:
            return self.rng.choice(list(avail_actions))
        best = max(avail_actions, key=lambda a: qvals[int(a)])
        return best

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def state_action_count(self):
        return sum(len(v) for v in self.Q.values())

    def greedy_policy_table(self):
        return {s: max(av, key=av.get) for s, av in self.Q.items()}


class MCAgent(BaseAgent):
    algo_name = "monte_carlo"

    def finish_hand(self, trajectory, profit):
        for s, a in trajectory:
            key = (s, int(a))
            self.N[key] = self.N.get(key, 0) + 1
            n = self.N[key]
            q = self.Q[s][int(a)]
            self.Q[s][int(a)] = q + (profit - q) / n


class QLearningAgent(BaseAgent):
    algo_name = "q_learning"

    def __init__(self, alpha=0.05, gamma=1.0, **kw):
        super().__init__(**kw)
        self.alpha = alpha
        self.gamma = gamma

    def finish_hand(self, trajectory, profit):
        n = len(trajectory)
        for i, (s, a) in enumerate(trajectory):
            if i < n - 1:
                r = 0.0
                s_next, _ = trajectory[i + 1]
                max_next = max(self.Q[s_next].values()) if self.Q.get(s_next) else 0.0
            else:
                r = profit
                max_next = 0.0
            q = self.Q[s][int(a)]
            target = r + self.gamma * max_next
            self.Q[s][int(a)] = q + self.alpha * (target - q)


class BetAgent:
    """Contextual bandit: state = true-count bucket, action = bet size
    (in units of the table minimum bet). Learns the expected profit-per-unit
    for each bucket via incremental averaging, then a greedy policy picks
    the bet size maximizing expected money profit for that bucket.

    Because profit-per-unit-bet is independent of the bet actually placed
    (the cards don't care how much money is riding on them), a risk-neutral
    EV-maximizing bettor should always bet the table max once EV(true count)
    turns positive, and the table min otherwise -- a corner solution. This
    is intentionally naive and is improved on in scenario 4 with a smoother
    Kelly-style bet spread (see train.py).
    """

    def __init__(self, bet_sizes=(1, 2, 3, 4, 6, 8), epsilon=0.2, epsilon_min=0.02,
                 epsilon_decay=0.999995, rng=None):
        self.bet_sizes = list(bet_sizes)
        self.Q = {}   # tc_bucket -> {bet_size: avg money profit per round}
        self.N = {}
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or random.Random()

    def _ensure(self, tc_bucket):
        return self.Q.setdefault(tc_bucket, {b: 0.0 for b in self.bet_sizes})

    def choose_bet(self, tc_bucket, greedy=False):
        qvals = self._ensure(tc_bucket)
        if (not greedy) and self.rng.random() < self.epsilon:
            return self.rng.choice(self.bet_sizes)
        return max(self.bet_sizes, key=lambda b: qvals[b])

    def update(self, tc_bucket, bet_size, money_profit):
        key = (tc_bucket, bet_size)
        self.N[key] = self.N.get(key, 0) + 1
        n = self.N[key]
        q = self.Q[tc_bucket][bet_size]
        self.Q[tc_bucket][bet_size] = q + (money_profit - q) / n

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def ev_per_unit_table(self):
        """Average profit-per-unit-bet observed for each true-count bucket,
        estimated from the bet_size=1 samples (unit-normalized), used to
        fit/report the count->edge relationship in the paper."""
        return {tc: qvals.get(1, 0.0) for tc, qvals in self.Q.items()}


class EdgeModel:
    """Online ordinary-least-squares regression of realized profit-per-unit-
    bet on the Hi-Lo true-count bucket: edge_hat(tc) = a + b*tc.

    Updated from *every* hand, regardless of the bet actually placed. A
    hand's profit-per-unit-bet doesn't depend on how much was staked, so
    there's nothing to explore about the effect of bet size once
    profit-per-unit is known for a given count -- estimating edge(tc) is a
    regression problem, not a bandit problem. Pooling every hand into one
    fit also means rare, extreme counts borrow statistical strength from
    the whole dataset instead of relying only on their own few thousand
    samples (see KellyBetAgent's docstring for why that distinction
    matters).
    """

    def __init__(self, min_n_for_ci=30):
        self.n = 0
        self.sx = 0.0
        self.sxx = 0.0
        self.sy = 0.0
        self.sxy = 0.0
        self.syy = 0.0
        self.min_n_for_ci = min_n_for_ci

    def update(self, tc, profit_unit):
        self.n += 1
        self.sx += tc
        self.sxx += tc * tc
        self.sy += profit_unit
        self.sxy += tc * profit_unit
        self.syy += profit_unit * profit_unit

    def _ols(self):
        n = self.n
        if n < 2:
            return 0.0, 0.0
        denom = n * self.sxx - self.sx * self.sx
        if abs(denom) < 1e-9:
            return self.sy / n, 0.0
        b = (n * self.sxy - self.sx * self.sy) / denom
        a = (self.sy - b * self.sx) / n
        return a, b

    def residual_variance(self):
        """Mean squared residual of the fit -- also used as the per-hand
        outcome variance for Kelly sizing."""
        n = self.n
        if n < 2:
            return 1.0
        a, b = self._ols()
        sse_over_n = (self.syy / n - 2 * a * self.sy / n - 2 * b * self.sxy / n
                      + a * a + 2 * a * b * self.sx / n + b * b * self.sxx / n)
        return max(sse_over_n, 1e-6)

    def predict(self, tc):
        a, b = self._ols()
        return a + b * tc

    def predict_with_se(self, tc):
        """Returns (edge_hat, standard_error_of_the_regression_line_at_tc).
        Infinite SE below min_n_for_ci so callers treat early predictions as
        untrustworthy rather than acting on a handful of noisy samples."""
        n = self.n
        if n < self.min_n_for_ci:
            return self.predict(tc), float("inf")
        a, b = self._ols()
        resid_var = self.residual_variance()
        mean_x = self.sx / n
        var_x = max(self.sxx / n - mean_x ** 2, 1e-9)
        se2 = resid_var * (1.0 / n + (tc - mean_x) ** 2 / (n * var_x))
        return a + b * tc, se2 ** 0.5


class KellyBetAgent:
    """Bet-sizing policy for scenario 2/4, replacing the naive `BetAgent`.

    Why `BetAgent` fails: it treats each true-count bucket as an independent
    multi-armed bandit and greedily argmaxes the *observed* money profit for
    each (bucket, bet-size) pair. But money_profit = bet_size * profit_unit
    at a fixed count -- there is nothing to discover about the effect of bet
    size once profit-per-unit is known, so modeling this as an exploration
    problem only adds variance. Extreme counts are also rare, so each
    bucket's own average is estimated from only a couple thousand hands and
    is dominated by sampling noise on the order of the true edge itself.
    Empirically (see logs/naive_bandit_baseline/), every bucket -- including
    the most favorable ones -- ends up looking unprofitable at every
    positive bet size, so the greedy bettor bets the table minimum (or, once
    a sit-out action exists, bets 0) forever, regardless of the count. That
    is a corner solution driven by estimator noise, not a real policy.

    Fix: fit a single pooled regression of profit-per-unit-bet on true count
    (`EdgeModel`) instead of one independent estimate per bucket. Pooling
    makes the fit at rare, extreme counts as reliable as the regression's
    total sample size, not just as reliable as that one bucket's own hands.
    Sit out (scenario 4 only) when the *lower confidence bound* of the
    fitted edge is non-positive, not just its point estimate -- using a
    confidence bound rather than the point estimate for that decision is
    what keeps noise from producing a false "always sit out" verdict the
    way it does for `BetAgent`.

    Bet size is Kelly-*proportional* rather than the raw Kelly formula taken
    literally: full-Kelly (edge / variance) is an optimal fraction of
    bankroll (order 1e-3 for a Blackjack-sized edge), not a count of
    table-minimum units, so plugging it directly into a 1..12-unit ladder
    always underflows to the table minimum. Since the per-hand outcome
    variance is essentially constant across true counts, the Kelly ratio
    between any two counts collapses to the ratio of their edges -- so we
    calibrate the top of the bet ladder to the edge at the most favorable
    count the system can represent (true count is clipped to +6 by
    `env.bucket_true_count`) and scale every other count's bet linearly
    in between, which preserves Kelly's edge-proportionality while mapping
    it onto a realistic, discrete bet spread instead of a raw bankroll
    fraction.
    """

    def __init__(self, bet_sizes=(0, 1, 2, 4, 8, 12), reference_tc=6,
                 kelly_fraction=1.0, wong_z=1.0, min_n_for_ci=30, rng=None):
        self.bet_sizes = sorted(bet_sizes)
        self.reference_tc = reference_tc
        self.kelly_fraction = kelly_fraction
        self.wong_z = wong_z
        self.model = EdgeModel(min_n_for_ci=min_n_for_ci)
        self.rng = rng or random.Random()

    def observe(self, tc_bucket, profit_unit):
        """Feed the realized profit-per-unit-bet for this hand into the
        edge model. Called for every hand, independent of the bet placed
        (including 0-bet/wonged-out rounds), so the model keeps learning
        edge(tc) even while the policy is sitting a count out."""
        self.model.update(tc_bucket, profit_unit)

    def choose_bet(self, tc_bucket, greedy=False):
        table_min = min(b for b in self.bet_sizes if b > 0)
        max_bet = max(self.bet_sizes)
        can_wong = 0 in self.bet_sizes  # only the scenario-4 ladder has a sit-out rung

        edge_hat, se = self.model.predict_with_se(tc_bucket)
        lower_bound = edge_hat - self.wong_z * se
        if lower_bound <= 0.0:
            return 0 if can_wong else table_min

        edge_now = max(edge_hat, 0.0)
        edge_reference = max(self.model.predict(self.reference_tc), 1e-9)
        ratio = min(self.kelly_fraction * edge_now / edge_reference, 1.0)
        target = table_min + (max_bet - table_min) * ratio
        feasible = [b for b in self.bet_sizes if 0 < b <= target]
        return max(feasible) if feasible else table_min

    def decay_epsilon(self):
        pass  # bet sizes are derived analytically -- nothing to explore/decay

    def edge_table(self, tc_range=range(-6, 7)):
        """Fitted edge(tc) for reporting/plotting (the paper's
        profit-estimate-correctness check compares this against the
        textbook Hi-Lo rule of thumb of ~+0.5% edge per +1 true count)."""
        return {tc: self.model.predict(tc) for tc in tc_range}
