from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.credit.hazard_curve import PiecewiseHazardCurve
from src.rates.curves import ZeroCurve


@dataclass
class WrongWayRiskResult:
    """
    Container for wrong-way risk CVA results.

    Attributes
    ----------
    base_cva:
        Deterministic-intensity first-to-default CVA.

    wwr_cva:
        Path-dependent wrong-way-risk CVA.

    incremental_wwr_cva:
        Difference between wrong-way CVA and base CVA.

    relative_wwr_impact:
        Incremental WWR CVA divided by base CVA.

    detail:
        Time-grid table with exposure, pathwise default probability,
        and period CVA contributions.
    """

    base_cva: float
    wwr_cva: float
    incremental_wwr_cva: float
    relative_wwr_impact: float
    detail: pd.DataFrame


def simulate_lognormal_hazard_paths(
    base_hazard_rates: np.ndarray,
    time_grid: np.ndarray,
    rate_shocks: np.ndarray,
    volatility: float,
    correlation: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate path-dependent counterparty hazard rates.

    Model idea
    ----------
    We start from deterministic base hazard rates lambda_base(t), then add
    stochastic lognormal shocks:

        lambda_m(t_i)
        =
        lambda_base(t_i)
        * exp(
            volatility * sqrt(dt) * Z_credit_m,i
            - 0.5 * volatility^2 * dt
        )

    The credit shock is correlated with the interest-rate shock:

        Z_credit = correlation * Z_rate
                   + sqrt(1 - correlation^2) * Z_independent

    Why this matters
    ----------------
    If correlation is positive, counterparty credit intensity tends to rise
    when the selected rate shock rises.

    If the swap exposure also rises in those rate states, CVA increases.
    That is wrong-way risk.

    Parameters
    ----------
    base_hazard_rates:
        Deterministic hazard rate on the simulation grid, shape (n_times,).

    time_grid:
        Simulation time grid, shape (n_times,).

    rate_shocks:
        Standard-normal shocks used by the rate simulation, shape
        (n_paths, n_times - 1). We use these to induce correlation.

    volatility:
        Volatility of credit intensity shocks. This controls how stochastic
        the hazard rate is.

    correlation:
        Correlation between rate shocks and credit shocks. Must be between
        -1 and 1.

    seed:
        Random seed for independent credit noise.

    Returns
    -------
    np.ndarray
        Pathwise hazard rates, shape (n_paths, n_times).
    """
    base_hazard_rates = np.asarray(base_hazard_rates, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    rate_shocks = np.asarray(rate_shocks, dtype=float)

    if base_hazard_rates.ndim != 1:
        raise ValueError("base_hazard_rates must be one-dimensional.")

    if time_grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional.")

    if len(base_hazard_rates) != len(time_grid):
        raise ValueError("base_hazard_rates must have same length as time_grid.")

    if rate_shocks.ndim != 2:
        raise ValueError("rate_shocks must have shape (n_paths, n_times - 1).")

    if rate_shocks.shape[1] != len(time_grid) - 1:
        raise ValueError("rate_shocks must have n_times - 1 columns.")

    if volatility < 0:
        raise ValueError("volatility cannot be negative.")

    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be between -1 and 1.")

    n_paths, n_steps = rate_shocks.shape
    n_times = len(time_grid)

    rng = np.random.default_rng(seed)

    independent_shocks = rng.standard_normal(size=(n_paths, n_steps))

    credit_shocks = (
        correlation * rate_shocks
        + np.sqrt(1.0 - correlation**2) * independent_shocks
    )

    hazard_paths = np.empty((n_paths, n_times))

    # At t=0, use the deterministic starting hazard rate.
    hazard_paths[:, 0] = base_hazard_rates[0]

    dt = np.diff(time_grid)

    for i in range(1, n_times):
        # Lognormal multiplier keeps hazard rates positive.
        # The drift correction keeps the one-step multiplier roughly
        # centered around one for small dt.
        multiplier = np.exp(
            volatility * np.sqrt(dt[i - 1]) * credit_shocks[:, i - 1]
            - 0.5 * volatility**2 * dt[i - 1]
        )

        hazard_paths[:, i] = base_hazard_rates[i] * multiplier

    return hazard_paths


def pathwise_survival_probabilities(
    hazard_paths: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """
    Convert pathwise hazard rates into pathwise survival probabilities.

    Formula
    -------
        Q_m(0,t_i) = exp(-sum_j lambda_m(t_j) * dt_j)

    We approximate the integral using left-endpoint hazard rates.

    Parameters
    ----------
    hazard_paths:
        Pathwise hazard rates, shape (n_paths, n_times).

    time_grid:
        Simulation time grid, shape (n_times,).

    Returns
    -------
    np.ndarray
        Pathwise survival probabilities, shape (n_paths, n_times).
    """
    hazard_paths = np.asarray(hazard_paths, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)

    if hazard_paths.ndim != 2:
        raise ValueError("hazard_paths must be two-dimensional.")

    if len(time_grid) != hazard_paths.shape[1]:
        raise ValueError("time_grid length must match hazard_paths columns.")

    dt = np.diff(time_grid)

    # Integrated hazard starts at zero at t=0.
    cumulative_hazard = np.zeros_like(hazard_paths)

    # Use hazard over [t_{i-1}, t_i] to update cumulative hazard at t_i.
    increments = hazard_paths[:, :-1] * dt[None, :]
    cumulative_hazard[:, 1:] = np.cumsum(increments, axis=1)

    survival = np.exp(-cumulative_hazard)

    return survival


def pathwise_marginal_default_probabilities(
    pathwise_survival: np.ndarray,
) -> np.ndarray:
    """
    Compute pathwise marginal default probabilities.

    Formula
    -------
        PD_m(t_{i-1}, t_i) = Q_m(0,t_{i-1}) - Q_m(0,t_i)

    Returns an array with the same shape as survival. The first column is zero
    because there is no interval ending at t=0.
    """
    pathwise_survival = np.asarray(pathwise_survival, dtype=float)

    if pathwise_survival.ndim != 2:
        raise ValueError("pathwise_survival must be two-dimensional.")

    marginal_pd = np.zeros_like(pathwise_survival)
    marginal_pd[:, 1:] = (
        pathwise_survival[:, :-1]
        - pathwise_survival[:, 1:]
    )

    return marginal_pd


def compute_wrong_way_cva(
    time_grid: np.ndarray,
    collateralized_positive_exposure_paths: np.ndarray,
    counterparty_curve: PiecewiseHazardCurve,
    self_curve: PiecewiseHazardCurve,
    discount_curve: ZeroCurve,
    rate_shocks: np.ndarray,
    base_cva: float,
    hazard_volatility: float = 1.0,
    correlation: float = 0.5,
    seed: int | None = 42,
) -> WrongWayRiskResult:
    """
    Compute first-to-default CVA with stochastic counterparty intensity.

    Deterministic CVA uses:
        EE(t_i) * PD_C(t_{i-1},t_i)

    Wrong-way CVA uses:
        E[ E_m(t_i) * PD_C,m(t_{i-1},t_i) ]

    This captures dependence between exposure and counterparty credit quality.

    Parameters
    ----------
    time_grid:
        Simulation time grid.

    collateralized_positive_exposure_paths:
        Pathwise collateralized positive exposure, shape (n_paths, n_times).

    counterparty_curve:
        Base deterministic counterparty hazard curve.

    self_curve:
        Own/self deterministic hazard curve, used for first-to-default survival.

    discount_curve:
        Initial zero curve used for discounting.

    rate_shocks:
        Rate simulation shocks, shape (n_paths, n_times - 1).

    base_cva:
        Deterministic first-to-default CVA used for comparison.

    hazard_volatility:
        Volatility parameter for stochastic hazard rates.

    correlation:
        Correlation between rate shocks and counterparty credit shocks.

    seed:
        Random seed.

    Returns
    -------
    WrongWayRiskResult
        Wrong-way CVA result and time-detail table.
    """
    time_grid = np.asarray(time_grid, dtype=float)
    exposure_paths = np.asarray(collateralized_positive_exposure_paths, dtype=float)

    if exposure_paths.ndim != 2:
        raise ValueError("collateralized_positive_exposure_paths must be 2D.")

    if exposure_paths.shape[1] != len(time_grid):
        raise ValueError("exposure paths must have same number of columns as time_grid.")

    # Base deterministic hazard rate on the simulation grid.
    base_hazard_grid = counterparty_curve.hazard_rate(time_grid)

    # Simulate path-dependent hazard rates correlated with rate shocks.
    hazard_paths = simulate_lognormal_hazard_paths(
        base_hazard_rates=base_hazard_grid,
        time_grid=time_grid,
        rate_shocks=rate_shocks,
        volatility=hazard_volatility,
        correlation=correlation,
        seed=seed,
    )

    # Convert hazard paths into pathwise survival and default probabilities.
    cp_survival_paths = pathwise_survival_probabilities(
        hazard_paths=hazard_paths,
        time_grid=time_grid,
    )

    cp_marginal_pd_paths = pathwise_marginal_default_probabilities(
        pathwise_survival=cp_survival_paths,
    )

    # First-to-default survival of the bank/self remains deterministic here.
    self_survival = self_curve.survival_probability(time_grid)

    discount_factors = discount_curve.discount_factor(time_grid)

    # Expected product E[exposure * pathwise marginal PD].
    expected_exposure_times_pd = np.mean(
        exposure_paths * cp_marginal_pd_paths,
        axis=0,
    )

    cva_contributions = (
        counterparty_curve.loss_given_default
        * discount_factors
        * expected_exposure_times_pd
        * self_survival
    )

    # First time point has zero marginal PD by construction.
    wwr_cva = float(np.sum(cva_contributions[1:]))

    incremental = wwr_cva - base_cva

    relative_impact = (
        incremental / base_cva
        if base_cva != 0
        else np.nan
    )

    detail = pd.DataFrame({
        "time": time_grid,
        "discount_factor": discount_factors,
        "mean_collateralized_positive_exposure": exposure_paths.mean(axis=0),
        "mean_counterparty_hazard_rate": hazard_paths.mean(axis=0),
        "mean_counterparty_survival": cp_survival_paths.mean(axis=0),
        "mean_counterparty_marginal_pd": cp_marginal_pd_paths.mean(axis=0),
        "expected_exposure_times_pd": expected_exposure_times_pd,
        "self_survival": self_survival,
        "wwr_cva_contribution": cva_contributions,
    })

    return WrongWayRiskResult(
        base_cva=base_cva,
        wwr_cva=wwr_cva,
        incremental_wwr_cva=incremental,
        relative_wwr_impact=relative_impact,
        detail=detail,
    )