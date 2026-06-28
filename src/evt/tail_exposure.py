from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import genpareto


@dataclass
class EVTTailFit:
    """
    Container for EVT tail-fit results at one time point.

    Attributes
    ----------
    time:
        Time point in years.

    threshold:
        Threshold u used for exceedances.

    threshold_quantile:
        Quantile level used to define the threshold.

    n_observations:
        Total number of exposure observations.

    n_exceedances:
        Number of observations above the threshold.

    exceedance_probability:
        Empirical probability of exceeding the threshold.

    shape:
        GPD shape parameter xi.

    scale:
        GPD scale parameter beta.

    empirical_quantile:
        Empirical Monte Carlo quantile at the target level.

    evt_quantile:
        EVT-smoothed quantile estimate at the target level.
    """

    time: float
    threshold: float
    threshold_quantile: float
    n_observations: int
    n_exceedances: int
    exceedance_probability: float
    shape: float
    scale: float
    empirical_quantile: float
    evt_quantile: float


def fit_gpd_tail_quantile(
    exposures: np.ndarray,
    time: float,
    target_quantile: float = 0.999,
    threshold_quantile: float = 0.95,
    min_exceedances: int = 50,
) -> EVTTailFit:
    """
    Fit a Generalized Pareto Distribution to the upper tail of exposure.

    Method
    ------
    Given exposure observations X, choose threshold:

        u = empirical quantile of X at threshold_quantile

    Define exceedances:

        Y = X - u | X > u

    Fit a GPD to Y:

        Y ~ GPD(xi, beta)

    Then estimate the target high quantile of X.

    EVT quantile formula
    --------------------
    Let p_u = P(X > u), estimated by n_exceedances / n.

    For target quantile q > threshold_quantile:

        P(X > x_q) = 1 - q

    Under the GPD approximation:

        x_q = u + beta / xi * [ ((1 - q) / p_u)^(-xi) - 1 ]

    If xi is close to zero, use the exponential limit:

        x_q = u + beta * log(p_u / (1 - q))

    Why this matters
    ----------------
    Empirical Monte Carlo quantiles can be noisy at very high confidence levels.
    EVT uses the shape of the tail above a high threshold to produce a smoother
    estimate of extreme PFE.
    """
    exposures = np.asarray(exposures, dtype=float)

    if exposures.ndim != 1:
        raise ValueError("exposures must be one-dimensional.")

    if not 0 < threshold_quantile < 1:
        raise ValueError("threshold_quantile must be between 0 and 1.")

    if not 0 < target_quantile < 1:
        raise ValueError("target_quantile must be between 0 and 1.")

    if target_quantile <= threshold_quantile:
        raise ValueError("target_quantile must be above threshold_quantile.")

    if min_exceedances <= 0:
        raise ValueError("min_exceedances must be positive.")

    n_observations = len(exposures)

    threshold = float(np.quantile(exposures, threshold_quantile))
    exceedances = exposures[exposures > threshold] - threshold
    n_exceedances = len(exceedances)

    empirical_quantile = float(np.quantile(exposures, target_quantile))

    # If the tail has too few exceedances, return NaNs for the EVT fit.
    # This avoids pretending that a GPD fit with five tail points is science.
    if n_exceedances < min_exceedances:
        return EVTTailFit(
            time=float(time),
            threshold=threshold,
            threshold_quantile=threshold_quantile,
            n_observations=n_observations,
            n_exceedances=n_exceedances,
            exceedance_probability=n_exceedances / n_observations,
            shape=np.nan,
            scale=np.nan,
            empirical_quantile=empirical_quantile,
            evt_quantile=np.nan,
        )

    # Fit GPD to exceedances. We fix location at zero because exceedances
    # are already measured relative to the threshold.
    shape, location, scale = genpareto.fit(exceedances, floc=0.0)

    exceedance_probability = n_exceedances / n_observations
    tail_probability = 1.0 - target_quantile

    # Compute EVT quantile using the GPD tail approximation.
    if np.isclose(shape, 0.0):
        evt_quantile = threshold + scale * np.log(
            exceedance_probability / tail_probability
        )
    else:
        evt_quantile = threshold + (scale / shape) * (
            (tail_probability / exceedance_probability) ** (-shape) - 1.0
        )

    return EVTTailFit(
        time=float(time),
        threshold=threshold,
        threshold_quantile=threshold_quantile,
        n_observations=n_observations,
        n_exceedances=n_exceedances,
        exceedance_probability=exceedance_probability,
        shape=float(shape),
        scale=float(scale),
        empirical_quantile=empirical_quantile,
        evt_quantile=float(evt_quantile),
    )


def evt_pfe_profile(
    exposure_paths: np.ndarray,
    time_grid: np.ndarray,
    target_quantile: float = 0.999,
    threshold_quantile: float = 0.95,
    min_exceedances: int = 50,
    start_index: int = 1,
) -> pd.DataFrame:
    """
    Compute EVT-smoothed PFE across a full exposure time grid.

    Parameters
    ----------
    exposure_paths:
        Pathwise positive exposure, shape (n_paths, n_times).

    time_grid:
        Time grid, shape (n_times,).

    target_quantile:
        Extreme quantile to estimate, e.g. 0.995 or 0.999.

    threshold_quantile:
        Threshold quantile for GPD exceedances, e.g. 0.95.

    min_exceedances:
        Minimum number of exceedances required to fit the GPD.

    start_index:
        First time index to fit. We usually skip t=0 because exposure may be
        exactly zero for an at-market swap.

    Returns
    -------
    pd.DataFrame
        One row per time point with empirical and EVT PFE estimates.
    """
    exposure_paths = np.asarray(exposure_paths, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)

    if exposure_paths.ndim != 2:
        raise ValueError("exposure_paths must be two-dimensional.")

    if exposure_paths.shape[1] != len(time_grid):
        raise ValueError("exposure_paths columns must match time_grid length.")

    fits = []

    for i in range(start_index, len(time_grid)):
        fit = fit_gpd_tail_quantile(
            exposures=exposure_paths[:, i],
            time=time_grid[i],
            target_quantile=target_quantile,
            threshold_quantile=threshold_quantile,
            min_exceedances=min_exceedances,
        )

        fits.append(fit.__dict__)

    return pd.DataFrame(fits)


def evt_summary_at_peak(
    evt_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the time point where EVT PFE is highest.

    This gives a compact table useful for README/report discussion.
    """
    valid = evt_profile.dropna(subset=["evt_quantile"]).copy()

    if valid.empty:
        return pd.DataFrame([
            {
                "metric": "No valid EVT fits",
                "value": np.nan,
            }
        ])

    peak_row = valid.loc[valid["evt_quantile"].idxmax()]

    return pd.DataFrame([
        {
            "metric": "Peak EVT PFE time",
            "value": peak_row["time"],
        },
        {
            "metric": "Peak empirical PFE",
            "value": peak_row["empirical_quantile"],
        },
        {
            "metric": "Peak EVT PFE",
            "value": peak_row["evt_quantile"],
        },
        {
            "metric": "Threshold at peak",
            "value": peak_row["threshold"],
        },
        {
            "metric": "GPD shape at peak",
            "value": peak_row["shape"],
        },
        {
            "metric": "GPD scale at peak",
            "value": peak_row["scale"],
        },
        {
            "metric": "Exceedances at peak",
            "value": peak_row["n_exceedances"],
        },
    ])

def filter_evt_profile(
    evt_profile: pd.DataFrame,
    min_threshold: float = 1_000.0,
) -> pd.DataFrame:
    """
    Filter EVT profile to time points where the tail threshold is meaningful.

    Near maturity, many exposure paths can collapse to zero. Fitting a GPD
    to nearly-zero exposure tails is not very informative and can create
    unstable-looking tail parameters.

    Parameters
    ----------
    evt_profile:
        EVT profile returned by evt_pfe_profile.

    min_threshold:
        Minimum threshold required to keep a time point.

    Returns
    -------
    pd.DataFrame
        Filtered EVT profile.
    """
    return evt_profile.loc[
        evt_profile["threshold"] >= min_threshold
    ].copy()