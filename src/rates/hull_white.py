from dataclasses import dataclass
import numpy as np
from src.rates.curves import ZeroCurve
"""
Hull-White one-factor short-rate model.

The model is:

    dr_t = [theta(t) - a r_t] dt + sigma dW_t

where theta(t) is constructed from the initial zero curve so that the model
fits the observed term structure.

In this project, the dynamic parameters a and sigma are estimated historically
from the ECB short-rate proxy using the AR(1) representation of a Gaussian
mean-reverting short-rate process. This is a practical statistical calibration,
not a full market calibration to swaptions or caps/floors. Parameter uncertainty
is therefore addressed through calibration-window comparison and stress testing.
"""

@dataclass
class HullWhiteParams:
    """
    Parameters for the one-factor Hull-White model.

    Model:
        dr_t = [theta(t) - a r_t] dt + sigma dW_t
    """
    a: float
    sigma: float


@dataclass
class HullWhiteSimulationResult:
    rates: np.ndarray
    time_grid: np.ndarray
    theta: np.ndarray
    discount_factors: np.ndarray
    shocks: np.ndarray


def hull_white_theta(
    t: float | np.ndarray,
    curve: ZeroCurve,
    a: float,
    sigma: float,
) -> float | np.ndarray:
    """
    Compute Hull-White theta(t):

        theta(t) = df(0,t)/dt + a f(0,t)
                   + (sigma^2 / (2a)) * (1 - exp(-2at))

    where f(0,t) is the instantaneous forward rate.
    """
    t_arr = np.asarray(t, dtype=float)

    f = curve.instantaneous_forward_rate(t_arr)
    dfdt = curve.forward_rate_derivative(t_arr)

    theta = dfdt + a * f + (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * t_arr))

    if np.isscalar(t):
        return float(theta)

    return theta


def simulate_hull_white_euler(
    r0: float,
    curve: ZeroCurve,
    a: float,
    sigma: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = 42,
) -> HullWhiteSimulationResult:
    """
    Simulate Hull-White one-factor short-rate paths using Euler discretization.

    This implementation is transparent and easy to audit. The generated
    Gaussian shocks are stored because later modules, such as wrong-way risk,
    need to correlate counterparty credit shocks with interest-rate shocks.

    Parameters
    ----------
    r0:
        Initial short rate.

    curve:
        Initial zero curve used to compute theta(t).

    a:
        Mean reversion speed.

    sigma:
        Short-rate volatility.

    maturity:
        Simulation horizon in years.

    n_steps:
        Number of time steps.

    n_paths:
        Number of Monte Carlo paths.

    seed:
        Random seed.

    Returns
    -------
    HullWhiteSimulationResult
        rates:
            Simulated short rates, shape (n_paths, n_steps + 1).

        time_grid:
            Time grid in years, shape (n_steps + 1,).

        theta:
            Theta values on the time grid, shape (n_steps + 1,).

        discount_factors:
            Pathwise money-market discount factors from 0 to t_i,
            shape (n_paths, n_steps + 1).

        shocks:
            Standard-normal shocks used to simulate the short-rate paths,
            shape (n_paths, n_steps).
    """
    if a <= 0:
        raise ValueError("Mean reversion parameter a must be positive.")

    if sigma <= 0:
        raise ValueError("Volatility sigma must be positive.")

    rng = np.random.default_rng(seed)

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    time_grid = np.linspace(0.0, maturity, n_steps + 1)

    # Avoid derivative issues exactly at zero by using a tiny positive time.
    theta_grid = hull_white_theta(
        np.maximum(time_grid, 1e-8),
        curve=curve,
        a=a,
        sigma=sigma,
    )

    # Store simulated short-rate paths.
    rates = np.empty((n_paths, n_steps + 1))
    rates[:, 0] = r0

    # Store Gaussian shocks so wrong-way risk can correlate credit intensity
    # shocks with the same shocks that generated the interest-rate paths.
    shocks = rng.standard_normal(size=(n_paths, n_steps))

    for i in range(n_steps):
        # Euler step:
        #   r(t+dt) = r(t) + [theta(t) - a r(t)] dt + sigma sqrt(dt) Z
        drift = theta_grid[i] - a * rates[:, i]

        rates[:, i + 1] = (
            rates[:, i]
            + drift * dt
            + sigma * sqrt_dt * shocks[:, i]
        )

    # Pathwise money-market discount factors:
    #   DF(0,t_i) = exp(- integral_0^t_i r_s ds)
    #
    # These are diagnostic pathwise discount factors. Swap pricing uses
    # Hull-White conditional zero-coupon bond prices P(t,T), not these
    # realized discount factors.
    short_rate_integral = np.zeros_like(rates)
    short_rate_integral[:, 1:] = np.cumsum(rates[:, :-1] * dt, axis=1)

    discount_factors = np.exp(-short_rate_integral)

    return HullWhiteSimulationResult(
        rates=rates,
        time_grid=time_grid,
        theta=theta_grid,
        discount_factors=discount_factors,
        shocks=shocks,
    )

def hull_white_B(
    t: float,
    T: float | np.ndarray,
    a: float,
) -> float | np.ndarray:
    """
    Compute the Hull-White B(t,T) function:

        B(t,T) = (1 - exp(-a(T-t))) / a

    This controls the sensitivity of the zero-coupon bond price
    to the current short rate r_t.
    """
    T_arr = np.asarray(T, dtype=float)

    tau = np.maximum(T_arr - t, 0.0)
    B = (1.0 - np.exp(-a * tau)) / a

    if np.isscalar(T):
        return float(B)

    return B


def hull_white_zero_coupon_price(
    t: float,
    T: float | np.ndarray,
    r_t: float | np.ndarray,
    curve: ZeroCurve,
    a: float,
    sigma: float,
) -> float | np.ndarray:
    """
    Price a zero-coupon bond P(t,T) under the one-factor Hull-White model.

    Formula:

        P(t,T) = A(t,T) exp(-B(t,T) r_t)

    with:

        B(t,T) = (1 - exp(-a(T-t))) / a

        A(t,T) =
            P(0,T)/P(0,t)
            * exp(
                B(t,T) f(0,t)
                - sigma^2/(4a) * (1 - exp(-2at)) * B(t,T)^2
            )

    Why this matters
    ----------------
    For mark-to-market pricing at time t, we need conditional bond
    prices P(t,T), not realized pathwise discount factors from 0 to T.
    Using realized path discount factors makes the time-0 swap value vary
    by path, which is incorrect.

    Parameters
    ----------
    t:
        Valuation time.
    T:
        Bond maturity or array of maturities.
    r_t:
        Current simulated short rate at time t. Can be scalar or array
        of shape (n_paths,).
    curve:
        Initial zero curve.
    a:
        Hull-White mean reversion.
    sigma:
        Hull-White volatility.

    Returns
    -------
    P(t,T):
        Conditional zero-coupon bond price.
    """
    T_arr = np.asarray(T, dtype=float)
    r_arr = np.asarray(r_t, dtype=float)

    if np.isclose(t, 0.0):

        p0T = curve.discount_factor(T_arr)

        if r_arr.ndim == 0:

            return p0T

        return np.tile(p0T, (r_arr.shape[0], 1))

    B = hull_white_B(t=t, T=T_arr, a=a)

    p0T = curve.discount_factor(T_arr)

    p0t = curve.discount_factor(t)

    f0t = curve.instantaneous_forward_rate(t)

    convexity_adjustment = (

        (sigma**2 / (4.0 * a))

        * (1.0 - np.exp(-2.0 * a * t))

        * B**2

    )

    A = (p0T / p0t) * np.exp(B * f0t - convexity_adjustment)

    if r_arr.ndim == 0:

        price = A * np.exp(-B * r_arr)

    else:

        price = A[None, :] * np.exp(-r_arr[:, None] * B[None, :])

    return price