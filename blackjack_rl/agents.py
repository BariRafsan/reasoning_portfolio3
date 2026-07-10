"""Learning agents."""

import random


class BaseAgent:
    def __init__(
        self, epsilon=0.15, epsilon_min=0.01, epsilon_decay=0.999999, rng=None
    ):
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

    def __init__(
        self,
        bet_sizes=(1, 2, 3, 4, 6, 8),
        epsilon=0.2,
        epsilon_min=0.02,
        epsilon_decay=0.999995,
        rng=None,
    ):
        self.bet_sizes = list(bet_sizes)
        self.Q = {}  # tc_bucket -> {bet_size: avg money profit per round}
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
        sse_over_n = (
            self.syy / n
            - 2 * a * self.sy / n
            - 2 * b * self.sxy / n
            + a * a
            + 2 * a * b * self.sx / n
            + b * b * self.sxx / n
        )
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
        var_x = max(self.sxx / n - mean_x**2, 1e-9)
        se2 = resid_var * (1.0 / n + (tc - mean_x) ** 2 / (n * var_x))
        return a + b * tc, se2**0.5


class KellyBetAgent:

    def __init__(
        self,
        bet_sizes=(0, 1, 2, 4, 8, 12),
        reference_tc=6,
        kelly_fraction=1.0,
        wong_z=1.0,
        min_n_for_ci=30,
        rng=None,
    ):
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

        return {tc: self.model.predict(tc) for tc in tc_range}
