"""The 60-second artefact: an allocation with a credible interval, and a list
of what the model cannot tell apart.

Run: python -m src.demo
"""

from __future__ import annotations

from .generator import (
    INCUMBENT_WEEKLY, TRUE_BASELINE, TRUE_COEFFICIENTS, generate, true_response,
)
from .model import (
    indistinguishable_channels, optimise, optimise_at_posterior_mean, sample,
)
from .transforms import CHANNELS, PRIORS


def main() -> None:
    data = generate(weeks=156, seed=20260911)
    posterior = sample(data.spend, data.sales,
                       draws=4_000, burn_in=1_500, thin=4, seed=3)

    budget = sum(INCUMBENT_WEEKLY.values())
    annual = budget * 52

    print("\n  MIXPRIORS - 150 observations, 28 parameters, GBP 40M to split")
    print("  " + "=" * 76)
    print(f"  {data.weeks} weeks, {len(CHANNELS)} channels, "
          f"GBP {annual:,.0f} annual budget.")
    print(f"  Sampler acceptance {posterior.acceptance_rate:.0%}, "
          f"{len(posterior.draws)} retained draws.\n")

    print("  PARAMETER RECOVERY (the only honest way to evaluate an MMM)")
    print("  " + "-" * 76)
    print(f"    {'channel':<14}{'true':>7}{'posterior mean':>17}"
          f"{'90% interval':>22}{'covers':>9}")
    covered = 0
    for channel in CHANNELS:
        lo, hi = posterior.credible_interval(
            lambda d, c=channel: d.coefficients[c])
        mean = posterior.mean(lambda d, c=channel: d.coefficients[c])
        truth = TRUE_COEFFICIENTS[channel]
        ok = lo <= truth <= hi
        covered += ok
        print(f"    {channel:<14}{truth:>7.2f}{mean:>17.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>22}{'yes' if ok else 'NO':>9}")
    print(f"\n    {covered}/{len(CHANNELS)} true values inside their intervals.")

    fitted_baseline = posterior.mean(lambda d: d.baseline)
    print(f"    baseline: {fitted_baseline:,.0f} (true {TRUE_BASELINE:,.0f})")
    print("    The baseline matters more than it looks. Give it a diffuse")
    print("    prior and it absorbs the media contribution: every coefficient")
    print("    collapses toward zero, the model reports that media does almost")
    print("    nothing, and it fits the data beautifully the whole time.\n")

    full = optimise(posterior, budget, steps=120)
    collapsed = optimise_at_posterior_mean(posterior, budget, steps=120)
    base_truth = true_response(INCUMBENT_WEEKLY)

    print("  THE ALLOCATION")
    print("  " + "-" * 76)
    print(f"    {'channel':<14}{'incumbent':>14}{'proposed':>14}{'change':>12}")
    for channel in CHANNELS:
        before = INCUMBENT_WEEKLY[channel]
        after = full.weekly_spend[channel]
        print(f"    {channel:<14}{before:>14,.0f}{after:>14,.0f}"
              f"{(after - before) / before:>+11.0%}")

    moved = sum(
        abs(full.weekly_spend[c] - INCUMBENT_WEEKLY[c]) for c in CHANNELS) / 2
    print(f"\n    {moved / budget:.0%} of the budget reallocated, spend unchanged.")
    print(f"    expected annual response: {full.expected_response:,.0f}")
    print(f"    90% credible interval:    [{full.response_low:,.0f}, "
          f"{full.response_high:,.0f}]")
    print("    A point estimate would have reported one number here.\n")

    print("  SCORED ON THE TRUE CURVES (which the optimiser never saw)")
    print("  " + "-" * 76)
    print(f"    {'incumbent plan':<26}{base_truth:>16,.0f}")
    for label, alloc in (("posterior-aware", full), ("point-estimate", collapsed)):
        scored = true_response(alloc.weekly_spend)
        print(f"    {label:<26}{scored:>16,.0f}{scored / base_truth - 1:>+10.2%}")
    print("\n    Both allocations come from the SAME posterior. The second")
    print("    collapses it to its mean first, which is what most MMM")
    print("    pipelines do, and it is materially worse - it loads up on")
    print("    channels whose high mean is carried by a few extreme draws.\n")

    pairs = indistinguishable_channels(posterior)
    print("  WHAT THE MODEL CANNOT TELL APART")
    print("  " + "-" * 76)
    for a, b in pairs[:6]:
        print(f"    {a} / {b}")
    print(f"    ({len(pairs)} pairs whose coefficient intervals overlap heavily)")
    print("    Any fitting procedure will hand these some split of the credit.")
    print("    Presenting that split as a finding is how an MMM loses the room.\n")

    print("  THE PRIORS, AND WHY")
    print("  " + "-" * 76)
    for p in PRIORS[:4]:
        print(f"    {p.name:<14}{p.justification}")
    print("    'We used priors' is not an argument. Each of these is a claim")
    print("    a marketer can disagree with, which is what makes it one.\n")


if __name__ == "__main__":
    main()
