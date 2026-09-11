"""Weekly spend and sales with KNOWN response curves.

Parameter recovery is the only honest way to evaluate an MMM: there is no
held-out set that answers "what would sales have been at a different budget",
so goodness of fit tells you almost nothing about whether the allocation
advice is right.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .transforms import CHANNELS, geometric_adstock, hill_saturation, seasonality

#: TRUE parameters. Chosen so that two channels are deliberately collinear -
#: paid_search and paid_social move together - because a real media plan does
#: that and the model must be honest about not separating them.
TRUE_COEFFICIENTS: dict[str, float] = {
    "tv": 0.30, "radio": 0.10, "ooh": 0.12,
    "paid_search": 0.34, "paid_social": 0.22, "display": 0.04,
    "affiliate": 0.13, "direct_mail": 0.09, "podcast": 0.03,
}
TRUE_DECAYS: dict[str, float] = {
    "tv": 0.68, "radio": 0.48, "ooh": 0.60,
    "paid_search": 0.14, "paid_social": 0.34, "display": 0.30,
    "affiliate": 0.20, "direct_mail": 0.72, "podcast": 0.45,
}
TRUE_HALF_SATURATION: dict[str, float] = {
    "tv": 170_000.0, "radio": 55_000.0, "ooh": 85_000.0,
    "paid_search": 210_000.0, "paid_social": 130_000.0, "display": 65_000.0,
    "affiliate": 45_000.0, "direct_mail": 38_000.0, "podcast": 28_000.0,
}
TRUE_BASELINE = 1_100_000.0

#: Current weekly allocation, as the business spends it today.
INCUMBENT_WEEKLY: dict[str, float] = {
    "tv": 260_000.0, "radio": 55_000.0, "ooh": 70_000.0,
    "paid_search": 180_000.0, "paid_social": 110_000.0, "display": 85_000.0,
    "affiliate": 40_000.0, "direct_mail": 30_000.0, "podcast": 40_000.0,
}


#: Channels bought in bursts rather than continuously. The pattern matters:
#: saturation is concave, so a channel flighted at 1.8x for one week in three
#: delivers LESS total response than the same money spread evenly, and any
#: evaluation that uses average weekly spend will overstate it.
#: Each flighted channel bursts on a DIFFERENT phase. If they all burst on the
#: same weeks they are perfectly collinear, no model can separate them, and the
#: fit distributes their credit arbitrarily - TV's effect lands on
#: out-of-home and the optimiser then starves TV. Real plans stagger their
#: bursts, and a fixture that does not is testing an impossible problem rather
#: than a hard one.
FLIGHTED: dict[str, int] = {
    "tv": 0, "radio": 2, "ooh": 4, "direct_mail": 1, "podcast": 3,
}


def flighting_profile(channel: str, weeks: int = 52) -> list[float]:
    """Weekly multipliers averaging 1.0, so total spend is unchanged."""
    if channel not in FLIGHTED:
        return [1.0] * weeks
    phase = FLIGHTED[channel]
    raw = [1.8 if ((w + phase) // 3) % 3 == 0 else 0.6 for w in range(weeks)]
    scale = weeks / sum(raw)
    return [v * scale for v in raw]


@dataclass(frozen=True)
class Dataset:
    spend: dict[str, list[float]]
    sales: list[float]
    weeks: int

    def annual_budget(self) -> float:
        return sum(sum(v) for v in self.spend.values()) / self.weeks * 52


def true_response(weekly_spend: dict[str, float], weeks: int = 52) -> float:
    """Response under the TRUE curves, over a flighted year.

    Evaluated week by week with the channel's actual flighting rather than at
    average spend. Because saturation is concave, the response at the average
    is not the average response, and collapsing a flighted plan to its mean
    overstates it - which biases any optimiser built on that shortcut toward
    exactly the channels that are bought in bursts.
    """
    total = 0.0
    for channel in CHANNELS:
        profile = flighting_profile(channel, weeks)
        spend_series = [weekly_spend[channel] * m for m in profile]
        adstocked = geometric_adstock(spend_series, TRUE_DECAYS[channel])
        for value in adstocked:
            total += TRUE_COEFFICIENTS[channel] * TRUE_BASELINE * hill_saturation(
                value, TRUE_HALF_SATURATION[channel], 1.4)
    return total


def generate(weeks: int = 156, seed: int = 20260911, noise: float = 0.07) -> Dataset:
    rnd = random.Random(seed)
    spend: dict[str, list[float]] = {}

    for channel in CHANNELS:
        base = INCUMBENT_WEEKLY[channel]
        profile = flighting_profile(channel, weeks)

        # Quarter-to-quarter budget drift. Without it, spend sits in a narrow
        # band and the saturation curve is simply not identified: the data
        # never shows what happens at 2x or 0.4x, so the posterior follows the
        # prior and the optimiser extrapolates into territory the model has
        # never seen. Real budgets move, and an MMM dataset that does not
        # contain that movement cannot answer a reallocation question.
        quarters = [rnd.uniform(0.45, 1.75) for _ in range(weeks // 13 + 1)]

        spend[channel] = [
            max(0.0, base * quarters[w // 13] * profile[w]
                * rnd.uniform(0.85, 1.15))
            for w in range(weeks)
        ]

    # Deliberate collinearity, but PARTIAL: social tracks search with
    # substantial independent variation. Making it exactly proportional would
    # be unidentifiable rather than merely hard, and the honest finding - "the
    # data cannot fully separate these two" - requires that some separation is
    # possible in principle.
    ratio = INCUMBENT_WEEKLY["paid_social"] / INCUMBENT_WEEKLY["paid_search"]
    spend["paid_social"] = [
        max(0.0, ratio * v * rnd.uniform(0.62, 1.38))
        for v in spend["paid_search"]
    ]

    sales: list[float] = []
    for w in range(weeks):
        total = TRUE_BASELINE * (1 + seasonality(w))
        for channel in CHANNELS:
            adstocked = geometric_adstock(
                spend[channel], TRUE_DECAYS[channel])[w]
            total += TRUE_COEFFICIENTS[channel] * TRUE_BASELINE * hill_saturation(
                adstocked, TRUE_HALF_SATURATION[channel], 1.4)
        sales.append(total * math.exp(rnd.gauss(0.0, noise)))

    return Dataset(spend, sales, weeks)
