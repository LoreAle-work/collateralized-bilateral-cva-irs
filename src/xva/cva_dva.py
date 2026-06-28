from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.credit.hazard_curve import PiecewiseHazardCurve
from src.rates.curves import ZeroCurve


@dataclass
class XVABreakdown:
    """
    Container for CVA/DVA outputs.

    Attributes
    ----------
    cva:
        Credit Valuation Adjustment.

    dva:
        Debit Valuation Adjustment.

    adjusted_value:
        Risk-free value adjusted for CVA and DVA:

            V_adjusted = V_risk_free - CVA + DVA

    detail:
        Time-grid table containing discount factors, exposures,
        survival probabilities, marginal default probabilities,
        and period-by-period CVA/DVA contributions.
    """

    cva: float
    dva: float
    adjusted_value: float
    detail: pd.DataFrame


def _validate_time_grid(time_grid: np.ndarray) -> np.ndarray:
    """
    Validate and standardize a time grid.

    CVA/DVA calculations depend on adjacent intervals:
        [t_{i-1}, t_i]

    so the time grid must be one-dimensional and increasing.
    """
    time_grid = np.asarray(time_grid, dtype=float)

    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional.")

    if len(time_grid) < 2:
        raise ValueError("time_grid must contain at least two points.")

    if not np.all(np.diff(time_grid) > 0):
        raise ValueError("time_grid must be strictly increasing.")

    if not np.isclose(time_grid[0], 0.0):
        raise ValueError("time_grid must start at 0.")

    return time_grid


def _validate_exposure_profile(
    exposure: np.ndarray,
    time_grid: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate an exposure profile defined on the time grid.

    Expected exposure profiles should have the same length as time_grid.
    """
    exposure = np.asarray(exposure, dtype=float)

    if exposure.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if len(exposure) != len(time_grid):
        raise ValueError(f"{name} must have same length as time_grid.")

    if np.any(exposure < -1e-10):
        raise ValueError(f"{name} cannot contain negative values.")

    return np.maximum(exposure, 0.0)


def compute_unilateral_cva(
    time_grid: np.ndarray,
    expected_positive_exposure: np.ndarray,
    counterparty_curve: PiecewiseHazardCurve,
    discount_curve: ZeroCurve,
) -> tuple[float, pd.DataFrame]:
    """
    Compute unilateral collateralized CVA.

    Formula
    -------
        CVA = LGD_C * sum_i DF(0,t_i) * EE_c(t_i) * PD_C(t_{i-1},t_i)

    where:
        LGD_C = 1 - R_C
        EE_c(t_i) is collateralized expected positive exposure
        PD_C(t_{i-1},t_i) = Q_C(0,t_{i-1}) - Q_C(0,t_i)

    Interpretation
    --------------
    CVA is the expected discounted loss caused by counterparty default.
    It only uses positive exposure because losses occur when the trade
    is valuable to us and the counterparty fails.
    """
    time_grid = _validate_time_grid(time_grid)
    ee = _validate_exposure_profile(
        expected_positive_exposure,
        time_grid,
        "expected_positive_exposure",
    )

    t_start = time_grid[:-1]
    t_end = time_grid[1:]

    marginal_pd = counterparty_curve.marginal_default_probability(
        t_start,
        t_end,
    )

    discount_factors = discount_curve.discount_factor(t_end)

    exposure_at_end = ee[1:]

    contributions = (
        counterparty_curve.loss_given_default
        * discount_factors
        * exposure_at_end
        * marginal_pd
    )

    detail = pd.DataFrame({
        "t_start": t_start,
        "t_end": t_end,
        "discount_factor": discount_factors,
        "expected_positive_exposure": exposure_at_end,
        "counterparty_survival_start": counterparty_curve.survival_probability(t_start),
        "counterparty_survival_end": counterparty_curve.survival_probability(t_end),
        "counterparty_marginal_pd": marginal_pd,
        "cva_contribution": contributions,
    })

    return float(np.sum(contributions)), detail


def compute_unilateral_dva(
    time_grid: np.ndarray,
    expected_negative_exposure: np.ndarray,
    self_curve: PiecewiseHazardCurve,
    discount_curve: ZeroCurve,
) -> tuple[float, pd.DataFrame]:
    """
    Compute unilateral collateralized DVA.

    Formula
    -------
        DVA = LGD_B * sum_i DF(0,t_i) * ENE_c(t_i) * PD_B(t_{i-1},t_i)

    where:
        LGD_B = 1 - R_B
        ENE_c(t_i) is collateralized expected negative exposure
        PD_B(t_{i-1},t_i) = Q_B(0,t_{i-1}) - Q_B(0,t_i)

    Interpretation
    --------------
    DVA is the valuation benefit associated with our own default risk.
    It uses negative exposure because that is when we owe value to the
    counterparty.
    """
    time_grid = _validate_time_grid(time_grid)
    ene = _validate_exposure_profile(
        expected_negative_exposure,
        time_grid,
        "expected_negative_exposure",
    )

    t_start = time_grid[:-1]
    t_end = time_grid[1:]

    marginal_pd = self_curve.marginal_default_probability(
        t_start,
        t_end,
    )

    discount_factors = discount_curve.discount_factor(t_end)

    exposure_at_end = ene[1:]

    contributions = (
        self_curve.loss_given_default
        * discount_factors
        * exposure_at_end
        * marginal_pd
    )

    detail = pd.DataFrame({
        "t_start": t_start,
        "t_end": t_end,
        "discount_factor": discount_factors,
        "expected_negative_exposure": exposure_at_end,
        "self_survival_start": self_curve.survival_probability(t_start),
        "self_survival_end": self_curve.survival_probability(t_end),
        "self_marginal_pd": marginal_pd,
        "dva_contribution": contributions,
    })

    return float(np.sum(contributions)), detail


def compute_bilateral_first_to_default_cva_dva(
    time_grid: np.ndarray,
    expected_positive_exposure: np.ndarray,
    expected_negative_exposure: np.ndarray,
    counterparty_curve: PiecewiseHazardCurve,
    self_curve: PiecewiseHazardCurve,
    discount_curve: ZeroCurve,
    risk_free_value: float = 0.0,
) -> XVABreakdown:
    """
    Compute bilateral first-to-default CVA and DVA.

    First-to-default logic
    ----------------------
    In bilateral CVA/DVA, only the first default matters. If the bank defaults
    first, the counterparty cannot later generate CVA for the bank. If the
    counterparty defaults first, the bank cannot later generate DVA.

    Approximate formulas
    --------------------
        CVA_FTD =
            LGD_C * sum_i DF_i * EE_c(t_i)
            * PD_C(t_{i-1},t_i)
            * Q_B(0,t_i)

        DVA_FTD =
            LGD_B * sum_i DF_i * ENE_c(t_i)
            * PD_B(t_{i-1},t_i)
            * Q_C(0,t_i)

    where:
        Q_B(0,t_i) is survival probability of the bank/self
        Q_C(0,t_i) is survival probability of the counterparty

    Interpretation
    --------------
    The CVA contribution is weighted by the probability that the counterparty
    defaults in the interval and the bank survives to that time.

    The DVA contribution is weighted by the probability that the bank defaults
    in the interval and the counterparty survives to that time.

    This is still a simplified deterministic-credit implementation, but it
    captures the key first-to-default adjustment.
    """
    time_grid = _validate_time_grid(time_grid)

    ee = _validate_exposure_profile(
        expected_positive_exposure,
        time_grid,
        "expected_positive_exposure",
    )

    ene = _validate_exposure_profile(
        expected_negative_exposure,
        time_grid,
        "expected_negative_exposure",
    )

    t_start = time_grid[:-1]
    t_end = time_grid[1:]

    discount_factors = discount_curve.discount_factor(t_end)

    cp_marginal_pd = counterparty_curve.marginal_default_probability(
        t_start,
        t_end,
    )

    self_marginal_pd = self_curve.marginal_default_probability(
        t_start,
        t_end,
    )

    cp_survival_end = counterparty_curve.survival_probability(t_end)
    self_survival_end = self_curve.survival_probability(t_end)

    exposure_positive_at_end = ee[1:]
    exposure_negative_at_end = ene[1:]

    cva_contributions = (
        counterparty_curve.loss_given_default
        * discount_factors
        * exposure_positive_at_end
        * cp_marginal_pd
        * self_survival_end
    )

    dva_contributions = (
        self_curve.loss_given_default
        * discount_factors
        * exposure_negative_at_end
        * self_marginal_pd
        * cp_survival_end
    )

    cva = float(np.sum(cva_contributions))
    dva = float(np.sum(dva_contributions))

    adjusted_value = float(risk_free_value - cva + dva)

    detail = pd.DataFrame({
        "t_start": t_start,
        "t_end": t_end,
        "discount_factor": discount_factors,
        "collateralized_EE": exposure_positive_at_end,
        "collateralized_ENE": exposure_negative_at_end,
        "counterparty_survival": cp_survival_end,
        "self_survival": self_survival_end,
        "counterparty_marginal_pd": cp_marginal_pd,
        "self_marginal_pd": self_marginal_pd,
        "cva_contribution": cva_contributions,
        "dva_contribution": dva_contributions,
    })

    return XVABreakdown(
        cva=cva,
        dva=dva,
        adjusted_value=adjusted_value,
        detail=detail,
    )


def xva_summary_table(
    result: XVABreakdown,
) -> pd.DataFrame:
    """
    Create a compact CVA/DVA summary table.

    Useful for notebook display and README reporting.
    """
    return pd.DataFrame([
        {
            "metric": "CVA",
            "value": result.cva,
        },
        {
            "metric": "DVA",
            "value": result.dva,
        },
        {
            "metric": "Net XVA adjustment (-CVA + DVA)",
            "value": -result.cva + result.dva,
        },
        {
            "metric": "Risk-free value",
            "value": result.adjusted_value + result.cva - result.dva,
        },
        {
            "metric": "CVA/DVA-adjusted value",
            "value": result.adjusted_value,
        },
    ])