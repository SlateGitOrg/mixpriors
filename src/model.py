"""Posterior sampling, and a budget optimiser that consumes the posterior.

THE DIFFERENTIATOR LIVES HERE (second half).

The usual MMM produces a point estimate of return per channel and hands the
optimiser that single number. The optimiser then allocates as if it were true,
and the recommendation is confidently wrong wherever the model is uncertain -
which, on 150 observations, is nearly everywhere.

Here the optimiser maximises EXPECTED response under the posterior. Two
consequences fall out that a point estimate cannot express:

  - a channel with a high mean and enormous uncertainty is not loaded up,
    because the expectation over the posterior is dominated by the draws where
    it does little;
  - the recommendation comes with a credible interval, so "reallocate GBP 7M"
    becomes "reallocate GBP 7M for +6% to +15%".

Sampling is Metropolis-Hastings, written out rather than delegated. In
production this is NumPyro/Stan with NUTS; the point of writing it here is that
the prior and likelihood are visible, and the acceptance of the proposal is
something a reviewer can follow.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .transforms import (
    CHANNELS, PRIORS, geometric_adstock, hill_saturation, log_beta_pdf,
    log_half_normal_pdf, log_lognormal_pdf, prior_for, seasonality,
)


@dataclass
class Params:
    baseline: float
    coefficients: dict[str, float]
    decays: dict[str, float]
    half_saturations: dict[str, float]
    noise_sd: float

    def copy(self) -> Params:
        return Params(
            self.baseline, dict(self.coefficients), dict(self.decays),
            dict(self.half_saturations), self.noise_sd,
        )


def predict(
    params: Params, spend: dict[str, list[float]], weeks: int,
) -> list[float]:
    contributions = channel_contributions(params, spend, weeks)
    return [
        params.baseline * (1 + seasonality(t))
        + sum(contributions[c][t] for c in CHANNELS)
        for t in range(weeks)
    ]


def channel_contributions(
    params: Params, spend: dict[str, list[float]], weeks: int,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for channel in CHANNELS:
        adstocked = geometric_adstock(spend[channel], params.decays[channel])
        out[channel] = [
            params.coefficients[channel] * params.baseline * hill_saturation(
                adstocked[t], params.half_saturations[channel], 1.4)
            for t in range(weeks)
        ]
    return out


def log_posterior(
    params: Params, spend: dict[str, list[float]], sales: list[float],
) -> float:
    weeks = len(sales)
    predicted = predict(params, spend, weeks)

    # Gaussian likelihood on the log scale.
    lp = 0.0
    for y, mu in zip(sales, predicted):
        if mu <= 0:
            return -math.inf
        resid = (math.log(y) - math.log(mu)) / params.noise_sd
        lp += -0.5 * resid * resid - math.log(params.noise_sd)

    # Priors, each one justified in transforms.py.
    for p in PRIORS:
        lp += log_beta_pdf(params.decays[p.name], *p.decay_prior)
        lp += log_lognormal_pdf(
            params.half_saturations[p.name], *p.half_saturation_prior)
        lp += log_half_normal_pdf(
            params.coefficients[p.name], p.coefficient_scale)

    if params.baseline <= 0 or params.noise_sd <= 0:
        return -math.inf

    # THE BASELINE PRIOR, and why it is tight.
    #
    # The baseline and the media coefficients are confounded: sales can be
    # explained by a high base with weak media, or a lower base with strong
    # media, and 156 weeks of aggregate data barely distinguishes them. With a
    # diffuse prior the sampler drifts to the high-baseline end, every
    # coefficient collapses toward zero, and the model reports that media does
    # almost nothing. That is the single most common way an MMM understates
    # media, and it looks like a well-fitting model the whole time.
    #
    # Practitioners pin the baseline from periods of zero or near-zero spend,
    # where base sales are observed directly. This prior encodes that
    # observation: base sales of about 1.1M, known to within a few percent.
    lp += log_lognormal_pdf(params.baseline, math.log(1_100_000), 0.03)
    lp += log_half_normal_pdf(params.noise_sd, 0.3)
    return lp


def initial_params() -> Params:
    return Params(
        baseline=1_000_000.0,
        coefficients={p.name: p.coefficient_scale * 0.5 for p in PRIORS},
        decays={p.name: p.decay_prior[0] / sum(p.decay_prior) for p in PRIORS},
        half_saturations={
            p.name: math.exp(p.half_saturation_prior[0]) for p in PRIORS},
        noise_sd=0.12,
    )


@dataclass
class Posterior:
    draws: list[Params] = field(default_factory=list)
    accepted: int = 0
    proposed: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    def credible_interval(
        self, extract, low: float = 0.05, high: float = 0.95,
    ) -> tuple[float, float]:
        values = sorted(extract(d) for d in self.draws)
        if not values:
            return (0.0, 0.0)
        lo = values[max(0, int(low * (len(values) - 1)))]
        hi = values[min(len(values) - 1, int(high * (len(values) - 1)))]
        return (lo, hi)

    def mean(self, extract) -> float:
        return sum(extract(d) for d in self.draws) / len(self.draws)

    def covers(self, extract, truth: float) -> bool:
        lo, hi = self.credible_interval(extract)
        return lo <= truth <= hi


def sample(
    spend: dict[str, list[float]], sales: list[float],
    draws: int = 4_000, burn_in: int = 1_500, thin: int = 4,
    step: float = 0.08, seed: int = 1, target_acceptance: float = 0.25,
) -> Posterior:
    """Metropolis-Hastings with per-channel block updates and step adaptation.

    Two things that are easy to leave out and that decide whether the
    posterior is usable:

      BLOCK UPDATES. Proposing a new value for all 28 parameters at once means
      almost every proposal is rejected - one bad component spoils the whole
      move. Updating one channel's three parameters at a time keeps the
      acceptance rate workable, and the chain actually explores.

      STEP ADAPTATION. A hand-tuned step size is tuned for one dataset. During
      burn-in the step is adjusted toward a target acceptance rate, so the
      sampler is not silently stuck on somebody else's data. Adaptation stops
      when burn-in ends, because adapting on the kept draws breaks the
      stationary distribution.
    """
    rnd = random.Random(seed)
    current = initial_params()
    current_lp = log_posterior(current, spend, sales)
    posterior = Posterior()

    # One step size per block: the channels, plus a global block.
    blocks = [*CHANNELS, "__global__"]
    steps = {b: step for b in blocks}
    block_attempts = {b: 0 for b in blocks}
    block_accepts = {b: 0 for b in blocks}

    total = draws + burn_in
    for i in range(total):
        block = blocks[i % len(blocks)]
        proposal = current.copy()
        sigma = steps[block]

        if block == "__global__":
            proposal.baseline *= math.exp(rnd.gauss(0, sigma))
            proposal.noise_sd *= math.exp(rnd.gauss(0, sigma))
        else:
            proposal.coefficients[block] = max(
                1e-8, proposal.coefficients[block] * math.exp(rnd.gauss(0, sigma)))
            d = proposal.decays[block] + rnd.gauss(0, sigma * 0.6)
            proposal.decays[block] = min(0.985, max(0.015, d))
            proposal.half_saturations[block] *= math.exp(rnd.gauss(0, sigma))

        proposal_lp = log_posterior(proposal, spend, sales)
        posterior.proposed += 1
        block_attempts[block] += 1

        if math.log(rnd.random() + 1e-300) < proposal_lp - current_lp:
            current, current_lp = proposal, proposal_lp
            posterior.accepted += 1
            block_accepts[block] += 1

        # Adapt only during burn-in.
        if i < burn_in and block_attempts[block] % 40 == 0:
            rate = block_accepts[block] / block_attempts[block]
            steps[block] *= math.exp(0.6 * (rate - target_acceptance))
            steps[block] = min(1.5, max(1e-3, steps[block]))

        if i >= burn_in and (i - burn_in) % thin == 0:
            posterior.draws.append(current.copy())

    return posterior


# ---------------------------------------------------------------------------
# Budget allocation under the posterior
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Allocation:
    weekly_spend: dict[str, float]
    expected_response: float
    response_low: float
    response_high: float

    @property
    def total(self) -> float:
        return sum(self.weekly_spend.values())


def expected_response(
    posterior: Posterior, weekly_spend: dict[str, float], weeks: int = 52,
) -> tuple[float, float, float]:
    """Response under EVERY posterior draw, over a flighted year.

    Two concavity traps, both avoided here, and both of which inflate the
    answer if you take the shortcut:

      PARAMETERS. Evaluating at the posterior MEAN is not the mean response,
      because the response curve is concave. That gap is where over-confident
      allocations come from.

      SPEND. Evaluating at AVERAGE weekly spend is not the response to a
      flighted plan, for the same reason. The model was fitted on bursty
      spend, so collapsing the plan to its mean here systematically
      over-credits exactly the channels that are bought in bursts - and the
      optimiser then pours money into them.
    """
    from .generator import flighting_profile

    profiles = {c: flighting_profile(c, weeks) for c in CHANNELS}
    totals: list[float] = []
    for draw in posterior.draws:
        total = 0.0
        for channel in CHANNELS:
            series = [weekly_spend[channel] * m for m in profiles[channel]]
            adstocked = geometric_adstock(series, draw.decays[channel])
            for value in adstocked:
                total += draw.coefficients[channel] * draw.baseline *                     hill_saturation(value, draw.half_saturations[channel], 1.4)
        totals.append(total)
    totals.sort()
    mean = sum(totals) / len(totals)
    lo = totals[max(0, int(0.05 * (len(totals) - 1)))]
    hi = totals[min(len(totals) - 1, int(0.95 * (len(totals) - 1)))]
    return mean, lo, hi


def _adstock_multipliers(
    profile: list[float], decay: float, max_lag: int = 8,
) -> list[float]:
    """Adstocked spend per week for a unit weekly level.

    Adstock is linear in the spend level, so the whole series for any level is
    this multiplied through. Precomputing it once per (profile, decay) is what
    makes the flighted evaluation affordable.
    """
    return geometric_adstock(profile, decay, max_lag)


def build_response_table(
    posterior: Posterior, levels: dict[str, list[float]],
    weeks: int = 52, max_draws: int = 80,
) -> dict[str, list[tuple[float, float, float]]]:
    """Expected annual response per channel at each candidate spend level.

    Returns channel -> [(mean, p05, p95)] aligned with `levels[channel]`.

    Built once, so the greedy allocator below is a table lookup rather than a
    full posterior sweep per candidate move. Without this the honest
    evaluation - every draw, every week, flighted - is too slow to run, and
    slow is how the honest version quietly gets replaced by the shortcut.
    """
    from .generator import flighting_profile

    draws = posterior.draws
    if len(draws) > max_draws:
        stride = len(draws) // max_draws
        draws = draws[::stride][:max_draws]

    profiles = {c: flighting_profile(c, weeks) for c in CHANNELS}
    table: dict[str, list[tuple[float, float, float]]] = {}

    for channel in CHANNELS:
        per_level: list[tuple[float, float, float]] = []
        multipliers = [
            _adstock_multipliers(profiles[channel], d.decays[channel])
            for d in draws
        ]
        for level in levels[channel]:
            totals: list[float] = []
            for draw, mult in zip(draws, multipliers):
                scale = draw.coefficients[channel] * draw.baseline
                half = draw.half_saturations[channel]
                totals.append(sum(
                    scale * hill_saturation(level * m, half, 1.4) for m in mult))
            totals.sort()
            mean = sum(totals) / len(totals)
            lo = totals[max(0, int(0.05 * (len(totals) - 1)))]
            hi = totals[min(len(totals) - 1, int(0.95 * (len(totals) - 1)))]
            per_level.append((mean, lo, hi))
        table[channel] = per_level
    return table


def optimise(
    posterior: Posterior, budget: float,
    minimums: dict[str, float] | None = None,
    maximums: dict[str, float] | None = None,
    steps: int = 120, weeks: int = 52, max_draws: int = 80,
) -> Allocation:
    """Greedy marginal allocation against the posterior expectation."""
    minimums = minimums or {c: budget * 0.01 for c in CHANNELS}
    maximums = maximums or {c: budget * 0.45 for c in CHANNELS}

    floor = sum(minimums.values())
    if floor > budget:
        raise ValueError("minimum constraints exceed the budget")

    increment = (budget - floor) / steps if steps else 0.0
    levels = {
        c: [minimums[c] + k * increment for k in range(steps + 1)]
        for c in CHANNELS
    }
    table = build_response_table(posterior, levels, weeks, max_draws)

    index = {c: 0 for c in CHANNELS}
    for _ in range(steps):
        best_channel = None
        best_gain = 0.0
        for channel in CHANNELS:
            k = index[channel]
            if k + 1 > steps:
                continue
            if levels[channel][k + 1] > maximums[channel] + 1e-9:
                continue
            gain = table[channel][k + 1][0] - table[channel][k][0]
            if gain > best_gain:
                best_gain, best_channel = gain, channel
        if best_channel is None:
            break
        index[best_channel] += 1

    allocation = {c: levels[c][index[c]] for c in CHANNELS}
    mean = sum(table[c][index[c]][0] for c in CHANNELS)
    lo = sum(table[c][index[c]][1] for c in CHANNELS)
    hi = sum(table[c][index[c]][2] for c in CHANNELS)
    return Allocation(allocation, mean, lo, hi)


def optimise_at_posterior_mean(
    posterior: Posterior, budget: float,
    minimums: dict[str, float] | None = None,
    maximums: dict[str, float] | None = None,
    steps: int = 120, weeks: int = 52,
) -> Allocation:
    """The common shortcut: collapse the posterior to its mean, then optimise.

    Included so the differentiator can be MEASURED rather than asserted. This
    is what most MMM pipelines do - take the fitted point estimate, hand it to
    an optimiser, and present the result. It throws away everything the model
    knows about what it does not know, and it loads up on channels whose high
    mean is carried by a handful of extreme draws.
    """
    collapsed = posterior.draws[0].copy()
    collapsed.baseline = posterior.mean(lambda d: d.baseline)
    for channel in CHANNELS:
        collapsed.coefficients[channel] = posterior.mean(
            lambda d, c=channel: d.coefficients[c])
        collapsed.decays[channel] = posterior.mean(
            lambda d, c=channel: d.decays[c])
        collapsed.half_saturations[channel] = posterior.mean(
            lambda d, c=channel: d.half_saturations[c])

    single = Posterior(draws=[collapsed], accepted=1, proposed=1)
    return optimise(single, budget, minimums, maximums, steps, weeks,
                    max_draws=1)


def indistinguishable_channels(
    posterior: Posterior, threshold: float = 0.5,
) -> list[tuple[str, str]]:
    """Channel pairs whose effects the data cannot separate.

    Reported rather than split arbitrarily. Two collinear channels will be
    given some division of credit by any fitting procedure, and presenting
    that division as a finding is how an MMM loses the room.
    """
    out: list[tuple[str, str]] = []
    for i, a in enumerate(CHANNELS):
        for b in CHANNELS[i + 1:]:
            lo_a, hi_a = posterior.credible_interval(lambda d: d.coefficients[a])
            lo_b, hi_b = posterior.credible_interval(lambda d: d.coefficients[b])
            overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
            span = max(hi_a - lo_a, hi_b - lo_b, 1e-9)
            if overlap / span > threshold:
                out.append((a, b))
    return out
