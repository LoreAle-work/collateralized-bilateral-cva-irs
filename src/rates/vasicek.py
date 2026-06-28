from dataclasses import dataclass

import numpy as np


@dataclass
class VasicekSimulationResult:
    rates: np.ndarray
    time_grid: np.ndarray


def simulate_vasicek_exact(
    r0: float,
    a: float,
    b: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = 42,
) -> VasicekSimulationResult:
    """
    Simulate Vasicek short-rate paths using exact discretization.

    Parameters
    ----------
    r0:
        Initial short rate.
    a:
        Mean reversion speed.
    b:
        Long-run mean.
    sigma:
        Short-rate volatility.
    maturity:
        Simulation horizon in years.
    n_steps:
        Number of simulation steps.
    n_paths:
        Number of Monte Carlo paths.
    seed:
        Random seed.

    Returns
    -------
    VasicekSimulationResult
        rates has shape (n_paths, n_steps + 1).
    """
    rng = np.random.default_rng(seed)

    dt = maturity / n_steps
    time_grid = np.linspace(0.0, maturity, n_steps + 1)

    rates = np.empty((n_paths, n_steps + 1))
    rates[:, 0] = r0

    exp_term = np.exp(-a * dt)
    mean_coeff = b * (1 - exp_term)
    variance = sigma**2 * (1 - np.exp(-2 * a * dt)) / (2 * a)
    std = np.sqrt(variance)

    shocks = rng.standard_normal(size=(n_paths, n_steps))

    for i in range(n_steps):
        rates[:, i + 1] = (
            rates[:, i] * exp_term
            + mean_coeff
            + std * shocks[:, i]
        )

    return VasicekSimulationResult(rates=rates, time_grid=time_grid)