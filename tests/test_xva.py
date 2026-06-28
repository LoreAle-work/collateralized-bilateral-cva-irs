import numpy as np

from src.credit.hazard_curve import PiecewiseHazardCurve
from src.rates.curves import ZeroCurve
from src.xva.cva_dva import (
    compute_unilateral_cva,
    compute_unilateral_dva,
    compute_bilateral_first_to_default_cva_dva,
)


def test_first_to_default_cva_dva_below_unilateral_values():
    """
    First-to-default CVA/DVA should be below or equal to unilateral CVA/DVA.

    Reason:
    - FTD CVA is weighted by survival of the bank.
    - FTD DVA is weighted by survival of the counterparty.

    Since survival probabilities are <= 1, first-to-default values should not
    exceed their unilateral benchmarks.
    """
    time_grid = np.linspace(0.0, 5.0, 51)

    discount_curve = ZeroCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02]),
    )

    counterparty_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.01, 0.015, 0.02]),
        recovery_rate=0.40,
    )

    self_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.008, 0.012, 0.018]),
        recovery_rate=0.40,
    )

    # Simple deterministic exposure profiles for a clean unit test.
    expected_positive_exposure = np.linspace(0.0, 100_000.0, len(time_grid))
    expected_negative_exposure = np.linspace(0.0, 150_000.0, len(time_grid))

    unilateral_cva, _ = compute_unilateral_cva(
        time_grid=time_grid,
        expected_positive_exposure=expected_positive_exposure,
        counterparty_curve=counterparty_curve,
        discount_curve=discount_curve,
    )

    unilateral_dva, _ = compute_unilateral_dva(
        time_grid=time_grid,
        expected_negative_exposure=expected_negative_exposure,
        self_curve=self_curve,
        discount_curve=discount_curve,
    )

    bilateral = compute_bilateral_first_to_default_cva_dva(
        time_grid=time_grid,
        expected_positive_exposure=expected_positive_exposure,
        expected_negative_exposure=expected_negative_exposure,
        counterparty_curve=counterparty_curve,
        self_curve=self_curve,
        discount_curve=discount_curve,
        risk_free_value=0.0,
    )

    assert bilateral.cva <= unilateral_cva
    assert bilateral.dva <= unilateral_dva


def test_adjusted_value_identity():
    """
    The adjusted value must satisfy:

        V_adjusted = V_risk_free - CVA + DVA

    If this fails, the output table is lying, and tables already do enough
    damage without assistance.
    """
    time_grid = np.linspace(0.0, 5.0, 51)

    discount_curve = ZeroCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02]),
    )

    counterparty_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.01, 0.015, 0.02]),
        recovery_rate=0.40,
    )

    self_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.008, 0.012, 0.018]),
        recovery_rate=0.40,
    )

    risk_free_value = 12_345.0

    expected_positive_exposure = np.linspace(0.0, 100_000.0, len(time_grid))
    expected_negative_exposure = np.linspace(0.0, 150_000.0, len(time_grid))

    result = compute_bilateral_first_to_default_cva_dva(
        time_grid=time_grid,
        expected_positive_exposure=expected_positive_exposure,
        expected_negative_exposure=expected_negative_exposure,
        counterparty_curve=counterparty_curve,
        self_curve=self_curve,
        discount_curve=discount_curve,
        risk_free_value=risk_free_value,
    )

    expected_adjusted_value = risk_free_value - result.cva + result.dva

    assert np.isclose(result.adjusted_value, expected_adjusted_value)


def test_zero_exposure_gives_zero_cva_and_dva():
    """
    If both positive and negative exposure are zero, CVA and DVA should be zero.

    Revolutionary stuff: no exposure, no exposure adjustment.
    """
    time_grid = np.linspace(0.0, 5.0, 51)

    discount_curve = ZeroCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02]),
    )

    counterparty_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.01, 0.015, 0.02]),
        recovery_rate=0.40,
    )

    self_curve = PiecewiseHazardCurve(
        maturities=np.array([1.0, 3.0, 5.0]),
        hazard_rates=np.array([0.008, 0.012, 0.018]),
        recovery_rate=0.40,
    )

    zero_exposure = np.zeros(len(time_grid))

    result = compute_bilateral_first_to_default_cva_dva(
        time_grid=time_grid,
        expected_positive_exposure=zero_exposure,
        expected_negative_exposure=zero_exposure,
        counterparty_curve=counterparty_curve,
        self_curve=self_curve,
        discount_curve=discount_curve,
        risk_free_value=0.0,
    )

    assert np.isclose(result.cva, 0.0)
    assert np.isclose(result.dva, 0.0)
    assert np.isclose(result.adjusted_value, 0.0)