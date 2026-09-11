"""Adstock and saturation, with priors that carry the load.

THE DIFFERENTIATOR LIVES HERE (first half).

A CMO splits GBP 40M across nine channels. Weekly aggregate data gives roughly
150 observations to estimate nine channel effects plus seasonality and price.
An unconstrained fit will happily report that radio has a negative effect and
out-of-home returns 14x - both noise - and the budget moves accordingly.

Priors are what stop that. They are not a statistical nicety: with 150 rows and
20 parameters, the data alone does not identify the answer, so SOMETHING has to
supply the missing information. The choice is between priors you wrote down and
defended, and whatever the optimiser happens to land on.

Every prior below is stated with its justification, because "we used priors" is
not an argument and "we used a Beta(3,3) on adstock because carryover beyond
six weeks is not physically plausible for digital display" is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def geometric_adstock(spend: list[float], decay: float, max_lag: int = 8) -> list[float]:
    """Carryover: this week's effect includes a decayed share of past spend."""
    weights = [decay ** lag for lag in range(max_lag + 1)]
    total = sum(weights)
    out: list[float] = []
    for t in range(len(spend)):
        acc = 0.0
        for lag, w in enumerate(weights):
            if t - lag >= 0:
                acc += w * spend[t - lag]
        out.append(acc / total)
    return out


def hill_saturation(x: float, half_saturation: float, shape: float) -> float:
    """Diminishing returns. Output in [0, 1).

    Modelled on the spend that achieves half the maximum response, because
    that is a quantity a marketer can reason about and therefore challenge.
    """
    if x <= 0:
        return 0.0
    return x ** shape / (x ** shape + half_saturation ** shape)


@dataclass(frozen=True)
class ChannelPrior:
    name: str
    #: Beta prior on the adstock decay, as (alpha, beta).
    decay_prior: tuple[float, float]
    #: Log-normal prior on half-saturation spend, as (log-mean, log-sd).
    half_saturation_prior: tuple[float, float]
    #: Half-normal prior scale on the coefficient. Non-negative by
    #: construction: media spend does not reduce sales, and allowing a
    #: negative coefficient is how "radio hurts us" gets into a board pack.
    coefficient_scale: float
    justification: str


PRIORS: tuple[ChannelPrior, ...] = (
    ChannelPrior(
        "tv", (6.0, 3.0), (math.log(180_000), 0.6), 0.35,
        "TV carryover is long and well documented; half-saturation set near "
        "historical burst size",
    ),
    ChannelPrior(
        "radio", (4.0, 4.0), (math.log(60_000), 0.7), 0.20,
        "moderate carryover; smaller budgets saturate sooner",
    ),
    ChannelPrior(
        "ooh", (5.0, 3.0), (math.log(90_000), 0.7), 0.22,
        "outdoor carries over while the site is live",
    ),
    ChannelPrior(
        "paid_search", (1.5, 8.0), (math.log(220_000), 0.5), 0.40,
        "search is intent-capture with almost no carryover; decay prior is "
        "deliberately tight near zero",
    ),
    ChannelPrior(
        "paid_social", (3.0, 5.0), (math.log(140_000), 0.6), 0.30,
        "short carryover, saturates at moderate spend",
    ),
    ChannelPrior(
        "display", (3.0, 5.0), (math.log(70_000), 0.8), 0.15,
        "weak effect expected; wide saturation prior reflects genuine "
        "uncertainty",
    ),
    ChannelPrior(
        "affiliate", (2.0, 6.0), (math.log(50_000), 0.7), 0.18,
        "close to the sale, little carryover",
    ),
    ChannelPrior(
        "direct_mail", (7.0, 3.0), (math.log(40_000), 0.8), 0.20,
        "long carryover; small absolute budgets",
    ),
    ChannelPrior(
        "podcast", (4.0, 4.0), (math.log(30_000), 0.9), 0.12,
        "small channel, genuinely uncertain - the wide prior is the honest "
        "representation of that",
    ),
)

CHANNELS: tuple[str, ...] = tuple(p.name for p in PRIORS)


def prior_for(name: str) -> ChannelPrior:
    for p in PRIORS:
        if p.name == name:
            return p
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Log-densities for the priors. Used by the sampler.
# ---------------------------------------------------------------------------

def log_beta_pdf(x: float, a: float, b: float) -> float:
    if not 0.0 < x < 1.0:
        return -math.inf
    return (a - 1) * math.log(x) + (b - 1) * math.log(1 - x)


def log_lognormal_pdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return -math.inf
    z = (math.log(x) - mu) / sigma
    return -0.5 * z * z - math.log(x * sigma)


def log_half_normal_pdf(x: float, scale: float) -> float:
    if x < 0:
        return -math.inf
    return -0.5 * (x / scale) ** 2


def seasonality(week: int) -> float:
    return (
        0.09 * math.sin(2 * math.pi * week / 52)
        + 0.04 * math.cos(2 * math.pi * week / 26)
    )
