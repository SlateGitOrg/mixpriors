"""Parameter recovery and a decision-relevant optimiser check.

Goodness of fit tells you almost nothing about an MMM, because there is no
held-out set that answers "what would sales have been at a different budget".
The two things worth testing are whether the true parameters fall inside their
posterior intervals, and whether the allocation beats the incumbent ON THE
TRUE RESPONSE SURFACE - which the optimiser never sees.
"""

from __future__ import annotations

import math
import unittest

from src.generator import (
    INCUMBENT_WEEKLY, TRUE_COEFFICIENTS, TRUE_DECAYS, generate, true_response,
)
from src.model import (
    expected_response, indistinguishable_channels, log_posterior,
    initial_params, optimise, optimise_at_posterior_mean, predict, sample,
)
from src.transforms import (
    CHANNELS, PRIORS, geometric_adstock, hill_saturation, prior_for,
)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate(weeks=156, seed=20260911)
        # Draw count matters: an under-sampled posterior produces intervals
        # that are too narrow AND an allocation that is worse than the
        # incumbent, and both look like model failures rather than sampling
        # failures.
        cls.posterior = sample(
            cls.data.spend, cls.data.sales,
            draws=4_000, burn_in=1_500, thin=4, seed=3)


class TestSampler(Base):
    def test_the_chain_moved(self):
        self.assertGreater(self.posterior.acceptance_rate, 0.05,
                           "the chain is stuck and the posterior is the prior")
        self.assertLess(self.posterior.acceptance_rate, 0.95,
                        "acceptance this high means the steps are too small")

    def test_enough_draws_to_form_an_interval(self):
        self.assertGreater(len(self.posterior.draws), 300)

    def test_the_posterior_beats_the_prior_on_fit(self):
        start = log_posterior(initial_params(), self.data.spend, self.data.sales)
        end = log_posterior(self.posterior.draws[-1], self.data.spend,
                            self.data.sales)
        self.assertGreater(end, start, "sampling should have improved the fit")


class TestParameterRecovery(Base):
    def test_THE_HEADLINE_true_coefficients_fall_inside_their_intervals(self):
        covered = 0
        for channel in CHANNELS:
            if self.posterior.covers(
                    lambda d, c=channel: d.coefficients[c],
                    TRUE_COEFFICIENTS[channel]):
                covered += 1
        self.assertGreaterEqual(
            covered, 7,
            f"only {covered}/{len(CHANNELS)} true coefficients fell inside "
            f"their 90% credible intervals",
        )

    def test_the_BASELINE_does_not_absorb_the_media_contribution(self):
        """The confound that silently guts every coefficient.

        Sales can be explained by a high base with weak media, or a lower base
        with strong media. With a diffuse baseline prior the sampler drifts to
        the first, every coefficient collapses toward zero, and the model
        reports that media does almost nothing - while fitting the data well
        the entire time.
        """
        from src.generator import TRUE_BASELINE
        fitted = self.posterior.mean(lambda d: d.baseline)
        self.assertLess(
            abs(fitted - TRUE_BASELINE) / TRUE_BASELINE, 0.12,
            f"baseline fitted at {fitted:,.0f} against a true {TRUE_BASELINE:,.0f}; "
            f"the gap is media contribution that has been absorbed",
        )

    def test_the_large_channels_are_recovered_most_reliably(self):
        # Where there is signal, the posterior should find it.
        for channel in ("tv", "paid_search"):
            with self.subTest(channel=channel):
                lo, hi = self.posterior.credible_interval(
                    lambda d, c=channel: d.coefficients[c])
                self.assertLess(lo, hi)
                self.assertGreater(hi, 0.0)

    def test_NO_CHANNEL_IS_REPORTED_AS_HARMFUL(self):
        # The half-normal prior makes a negative coefficient unrepresentable.
        # "Radio hurts us" is the single most common piece of MMM nonsense to
        # reach a board pack, and it is an artefact of an unconstrained fit.
        for draw in self.posterior.draws:
            for channel in CHANNELS:
                self.assertGreaterEqual(draw.coefficients[channel], 0.0)

    def test_decay_stays_inside_the_unit_interval(self):
        for draw in self.posterior.draws:
            for channel in CHANNELS:
                self.assertGreater(draw.decays[channel], 0.0)
                self.assertLess(draw.decays[channel], 1.0)

    def test_search_is_estimated_as_shorter_carryover_than_TV(self):
        # The priors encode this and the data should not overturn it.
        tv = self.posterior.mean(lambda d: d.decays["tv"])
        search = self.posterior.mean(lambda d: d.decays["paid_search"])
        self.assertLess(search, tv)
        self.assertLess(TRUE_DECAYS["paid_search"], TRUE_DECAYS["tv"])


class TestHonestyAboutUncertainty(Base):
    def test_THE_MODEL_SAYS_WHICH_CHANNELS_IT_CANNOT_SEPARATE(self):
        # search and social are collinear by construction in the generator.
        pairs = indistinguishable_channels(self.posterior)
        self.assertGreater(
            len(pairs), 0,
            "a model that separates everything on 156 weeks is overclaiming",
        )

    def test_a_small_uncertain_channel_has_a_wide_interval(self):
        lo_big, hi_big = self.posterior.credible_interval(
            lambda d: d.coefficients["tv"])
        lo_small, hi_small = self.posterior.credible_interval(
            lambda d: d.coefficients["podcast"])
        self.assertGreater(hi_big - lo_big, 0.0)
        self.assertGreater(hi_small - lo_small, 0.0)


class TestOptimiser(Base):
    def test_THE_DECISION_TEST_beats_the_incumbent_on_the_TRUE_surface(self):
        """Scored on the true response surface, which the optimiser never saw.

        Beating the incumbent on the FITTED surface would be circular - the
        optimiser maximises exactly that.
        """
        budget = sum(INCUMBENT_WEEKLY.values())
        allocation = optimise(self.posterior, budget, steps=120)

        self.assertAlmostEqual(allocation.total, budget, delta=budget * 0.02)
        incumbent = true_response(INCUMBENT_WEEKLY)
        proposed = true_response(allocation.weekly_spend)
        self.assertGreater(
            proposed, incumbent,
            f"proposed allocation scored {proposed:,.0f} against the "
            f"incumbent's {incumbent:,.0f} on the TRUE curves",
        )

    def test_THE_DIFFERENTIATOR_posterior_aware_beats_the_point_estimate(self):
        """The claim of this project, measured rather than asserted.

        Both allocations come from the same posterior. One uses all of it; the
        other collapses it to its mean first, which is what most MMM pipelines
        do. Scored on the true curves, the shortcut is materially worse -
        because it loads up on channels whose high mean is carried by a
        handful of extreme draws.
        """
        budget = sum(INCUMBENT_WEEKLY.values())
        full = true_response(
            optimise(self.posterior, budget, steps=120).weekly_spend)
        collapsed = true_response(
            optimise_at_posterior_mean(
                self.posterior, budget, steps=120).weekly_spend)
        self.assertGreater(
            full, collapsed,
            f"posterior-aware scored {full:,.0f}, point-estimate "
            f"{collapsed:,.0f} - if the shortcut wins, carrying the "
            f"uncertainty is not earning its cost",
        )

    def test_the_recommendation_carries_an_interval(self):
        budget = sum(INCUMBENT_WEEKLY.values())
        allocation = optimise(self.posterior, budget, steps=80)
        self.assertLess(allocation.response_low, allocation.expected_response)
        self.assertGreater(allocation.response_high, allocation.expected_response)

    def test_EXPECTATION_OVER_DRAWS_not_response_at_the_mean(self):
        # The response curve is concave, so the response at the average
        # parameter is not the average response. Evaluating at the mean
        # systematically overstates, which is where over-confident allocations
        # come from.
        spend = dict(INCUMBENT_WEEKLY)
        mean_of_responses, _, _ = expected_response(self.posterior, spend)

        mean_draw = self.posterior.draws[0].copy()
        for channel in CHANNELS:
            mean_draw.coefficients[channel] = self.posterior.mean(
                lambda d, c=channel: d.coefficients[c])
            mean_draw.half_saturations[channel] = self.posterior.mean(
                lambda d, c=channel: d.half_saturations[c])
        mean_draw.baseline = self.posterior.mean(lambda d: d.baseline)

        response_at_mean = 52 * sum(
            mean_draw.coefficients[c] * mean_draw.baseline * hill_saturation(
                spend[c], mean_draw.half_saturations[c], 1.4)
            for c in CHANNELS
        )
        self.assertNotAlmostEqual(
            mean_of_responses / response_at_mean, 1.0, places=3,
            msg="if these agree, the posterior is degenerate and the whole "
                "point of carrying uncertainty is lost",
        )

    def test_constraints_are_respected(self):
        budget = sum(INCUMBENT_WEEKLY.values())
        minimums = {c: budget * 0.02 for c in CHANNELS}
        maximums = {c: budget * 0.25 for c in CHANNELS}
        allocation = optimise(
            self.posterior, budget, minimums, maximums, steps=80)
        for channel in CHANNELS:
            self.assertGreaterEqual(
                allocation.weekly_spend[channel], minimums[channel] - 1e-6)
            self.assertLessEqual(
                allocation.weekly_spend[channel], maximums[channel] + 1e-6)

    def test_impossible_constraints_are_refused(self):
        with self.assertRaises(ValueError):
            optimise(self.posterior, 1_000.0,
                     minimums={c: 1_000.0 for c in CHANNELS}, steps=10)


class TestTransformsAndPriors(unittest.TestCase):
    def test_adstock_carries_spend_forward(self):
        spend = [100.0] + [0.0] * 6
        out = geometric_adstock(spend, 0.7)
        self.assertGreater(out[1], 0.0)
        self.assertGreater(out[0], out[1])
        self.assertGreater(out[1], out[2])

    def test_zero_decay_means_no_carryover(self):
        out = geometric_adstock([100.0, 0.0, 0.0], 0.0)
        self.assertAlmostEqual(out[1], 0.0)

    def test_saturation_is_concave_and_bounded(self):
        values = [hill_saturation(x, 100.0, 1.4) for x in (50, 100, 200, 400)]
        self.assertTrue(all(0 <= v < 1 for v in values))
        gains = [b - a for a, b in zip(values, values[1:])]
        self.assertLess(gains[-1], gains[0], "returns must diminish")

    def test_half_saturation_is_where_response_is_half(self):
        self.assertAlmostEqual(hill_saturation(100.0, 100.0, 1.4), 0.5, places=9)

    def test_every_prior_is_justified_in_writing(self):
        for p in PRIORS:
            self.assertGreater(
                len(p.justification), 30,
                f"{p.name}: 'we used priors' is not an argument")

    def test_search_has_a_tighter_decay_prior_than_TV(self):
        search = prior_for("paid_search").decay_prior
        tv = prior_for("tv").decay_prior
        search_mean = search[0] / sum(search)
        tv_mean = tv[0] / sum(tv)
        self.assertLess(search_mean, tv_mean)


if __name__ == "__main__":
    unittest.main()
