import numpy as np

from src.credit.hazard_curve import PiecewiseHazardCurve


def test_survival_starts_at_one_and_decreases():
    curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.01, 0.02, 0.03]),
        recovery_rate=0.40,
    )

    assert np.isclose(curve.survival_probability(0.0), 1.0)
    assert curve.survival_probability(5.0) < curve.survival_probability(1.0)


def test_marginal_default_probability_non_negative():
    curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.01, 0.02, 0.03]),
        recovery_rate=0.40,
    )

    pd = curve.marginal_default_probability(
        np.array([0.0, 1.0, 3.0]),
        np.array([1.0, 3.0, 5.0]),
    )

    assert np.all(pd >= 0.0)