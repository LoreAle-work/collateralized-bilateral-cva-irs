from dataclasses import dataclass
import numpy as np
from src.rates.curves import ZeroCurve
from src.rates.hull_white import hull_white_zero_coupon_price

@dataclass
class InterestRateSwap:
    """
    Plain-vanilla fixed-for-floating interest rate swap.

    This class represents the contract definition only:
    - notional
    - fixed rate
    - maturity
    - payment frequency
    - payer/receiver direction

    The pricing functions below use simulated short-rate paths and pathwise
    discount factors to approximate the mark-to-market value of the swap
    through time.

    Convention:
    - payer swap: pay fixed, receive floating
    - receiver swap: receive fixed, pay floating
    """

    notional: float
    fixed_rate: float
    maturity: float
    payment_frequency: int = 2
    swap_type: str = "payer"

    def __post_init__(self) -> None:
        if self.notional <= 0:
            raise ValueError("notional must be positive.")

        if self.fixed_rate < 0:
            raise ValueError("fixed_rate cannot be negative.")

        if self.maturity <= 0:
            raise ValueError("maturity must be positive.")

        if self.payment_frequency <= 0:
            raise ValueError("payment_frequency must be positive.")

        if self.swap_type not in {"payer", "receiver"}:
            raise ValueError("swap_type must be either 'payer' or 'receiver'.")

    @property
    def payment_times(self) -> np.ndarray:
        """
        Return payment dates in years.

        Example:
        maturity = 5, payment_frequency = 2
        gives semiannual payments:
            0.5, 1.0, 1.5, ..., 5.0
        """
        dt = 1.0 / self.payment_frequency
        n_payments = int(round(self.maturity * self.payment_frequency))

        return np.arange(1, n_payments + 1) * dt

    @property
    def accrual_factor(self) -> float:
        """
        Year fraction for each fixed payment.

        For semiannual payments, this is 0.5.
        For annual payments, this is 1.0.
        """
        return 1.0 / self.payment_frequency


def interpolate_pathwise_discount_factors(
    time_grid: np.ndarray,
    discount_factors: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Interpolate pathwise discount factors to target payment times.

    Parameters
    ----------
    time_grid:
        Simulation time grid, shape (n_steps + 1,).
    discount_factors:
        Pathwise discount factors from time 0 to each simulation time,
        shape (n_paths, n_steps + 1).
    target_times:
        Target maturities/payment dates in years.

    Returns
    -------
    np.ndarray
        Interpolated pathwise discount factors, shape (n_paths, n_targets).

    Why this exists
    ---------------
    The simulation grid is usually daily, while swap payments are usually
    semiannual or annual. We need discount factors exactly at payment dates,
    so we interpolate them path by path.
    """
    n_paths = discount_factors.shape[0]
    out = np.empty((n_paths, len(target_times)))

    for m in range(n_paths):
        out[m] = np.interp(target_times, time_grid, discount_factors[m])

    return out


def price_swap_at_time_index(
    swap: InterestRateSwap,
    time_index: int,
    time_grid: np.ndarray,
    discount_factors: np.ndarray,
) -> np.ndarray:
    """
    Approximate the mark-to-market value of a swap at a given simulation time.

    Parameters
    ----------
    swap:
        InterestRateSwap object.
    time_index:
        Index of the valuation time on the simulation grid.
    time_grid:
        Simulation time grid.
    discount_factors:
        Pathwise discount factors from time 0 to each simulation time,
        shape (n_paths, n_steps + 1).

    Returns
    -------
    np.ndarray
        Swap values at the chosen time for every path, shape (n_paths,).

    Method
    ------
    At time t, we reprice the remaining swap cash flows.

    We use pathwise forward discount factors:

        P_m(t,T) = P_m(0,T) / P_m(0,t)

    Then approximate a payer swap as:

        V_payer(t) = N * [floating_leg(t) - fixed_leg(t)]

    where:

        fixed_leg(t) = K * sum_i alpha * P(t,T_i)

        floating_leg(t) approx = 1 - P(t,T_n)

    This is a simplified single-curve swap valuation. It is good enough for
    the counterparty exposure engine. Later, a multi-curve setup can replace
    this without changing the CVA logic.
    """
    t = time_grid[time_index]
    n_paths = discount_factors.shape[0]

    # Remaining payment dates after valuation time t.
    remaining_payment_times = swap.payment_times[swap.payment_times > t]

    # At or after maturity, the swap has no remaining value.
    if len(remaining_payment_times) == 0:
        return np.zeros(n_paths)

    # Interpolate pathwise discount factors P(0,T_i) at remaining payment dates.
    df_0_payments = interpolate_pathwise_discount_factors(
        time_grid=time_grid,
        discount_factors=discount_factors,
        target_times=remaining_payment_times,
    )

    # Pathwise discount factor to valuation time P(0,t).
    df_0_t = discount_factors[:, time_index]

    # Convert P(0,T) into P(t,T) path by path:
    # P(t,T) = P(0,T) / P(0,t)
    df_t_payments = df_0_payments / df_0_t[:, None]

    alpha = swap.accrual_factor

    # Fixed leg PV from t onward:
    # N * K * sum alpha_i P(t,T_i)
    fixed_leg = (
        swap.notional
        * swap.fixed_rate
        * alpha
        * np.sum(df_t_payments, axis=1)
    )

    # Floating leg approximation:
    # N * (1 - P(t,T_n))
    # This is standard for a par floating leg immediately after reset in a
    # simplified single-curve setting.
    df_t_maturity = df_t_payments[:, -1]
    floating_leg = swap.notional * (1.0 - df_t_maturity)

    payer_value = floating_leg - fixed_leg

    if swap.swap_type == "payer":
        return payer_value

    return -payer_value


def price_swap_exposure_cube(
    swap: InterestRateSwap,
    time_grid: np.ndarray,
    discount_factors: np.ndarray,
) -> np.ndarray:
    """
    Price the swap at every simulation time and on every Monte Carlo path.

    Returns
    -------
    np.ndarray
        Swap values, shape (n_paths, n_steps + 1).

    Why this matters
    ----------------
    CVA depends on future exposure, and future exposure depends on the future
    mark-to-market value of the derivative. This function creates the full
    pathwise swap value matrix:

        V_m(t_i)

    which is the input for exposure, collateral, CVA, and DVA.
    """
    n_paths, n_times = discount_factors.shape
    values = np.empty((n_paths, n_times))

    for i in range(n_times):
        values[:, i] = price_swap_at_time_index(
            swap=swap,
            time_index=i,
            time_grid=time_grid,
            discount_factors=discount_factors,
        )

    return values


def positive_exposure(values: np.ndarray) -> np.ndarray:
    """
    Positive exposure:

        E(t) = max(V(t), 0)

    This matters for CVA because it is the amount the bank can lose if the
    counterparty defaults.
    """
    return np.maximum(values, 0.0)


def negative_exposure(values: np.ndarray) -> np.ndarray:
    """
    Negative exposure:

        NE(t) = max(-V(t), 0)

    This matters for DVA because it is the amount the bank owes the
    counterparty when the bank itself defaults.
    """
    return np.maximum(-values, 0.0)


def expected_exposure(exposures: np.ndarray) -> np.ndarray:
    """
    Expected exposure at each time:

        EE(t_i) = average over Monte Carlo paths of E_m(t_i)
    """
    return exposures.mean(axis=0)


def potential_future_exposure(
    exposures: np.ndarray,
    quantile: float = 0.95,
) -> np.ndarray:
    """
    Potential Future Exposure at each time:

        PFE_q(t_i) = q-quantile of exposure across paths

    PFE is a tail exposure measure. It is not an expected loss.
    """
    return np.quantile(exposures, quantile, axis=0)

def price_swap_at_time_index_hull_white(
    swap: InterestRateSwap,
    time_index: int,
    time_grid: np.ndarray,
    rates: np.ndarray,
    curve: ZeroCurve,
    a: float,
    sigma: float,
) -> np.ndarray:
    """
    Price the swap at a simulation time using Hull-White zero-coupon
    bond prices.

    This is the correct exposure-pricing function for the Hull-White
    model.

    At time t, for each path m, we use the simulated short rate r_m(t)
    and the Hull-White affine bond pricing formula to compute P_m(t,T)
    for all remaining swap payment dates.

    This avoids the previous mistake of using realized pathwise discount
    factors from 0 to T as if they were conditional bond prices.
    """
    t = float(time_grid[time_index])
    r_t = rates[:, time_index]
    n_paths = rates.shape[0]

    remaining_payment_times = swap.payment_times[swap.payment_times > t]

    if len(remaining_payment_times) == 0:
        return np.zeros(n_paths)

    # Conditional zero-coupon prices P_m(t,T_i)
    # Shape: (n_paths, n_remaining_payments)
    p_t_payments = hull_white_zero_coupon_price(
        t=t,
        T=remaining_payment_times,
        r_t=r_t,
        curve=curve,
        a=a,
        sigma=sigma,
    )

    alpha = swap.accrual_factor

    # Fixed leg:
    # N * K * sum_i alpha * P(t,T_i)
    fixed_leg = (
        swap.notional
        * swap.fixed_rate
        * alpha
        * np.sum(p_t_payments, axis=1)
    )

    # Floating leg approximation:
    # N * (1 - P(t,T_n))
    # This is a simplified single-curve representation.
    p_t_maturity = p_t_payments[:, -1]
    floating_leg = swap.notional * (1.0 - p_t_maturity)

    payer_value = floating_leg - fixed_leg

    if swap.swap_type == "payer":
        return payer_value

    return -payer_value


def price_swap_exposure_cube_hull_white(
    swap: InterestRateSwap,
    time_grid: np.ndarray,
    rates: np.ndarray,
    curve: ZeroCurve,
    a: float,
    sigma: float,
) -> np.ndarray:
    """
    Price the swap at every simulation time and on every path using
    Hull-White conditional zero-coupon bond prices.

    Output:
        values[m, i] = V_m(t_i)

    This is the correct input for exposure generation under the
    Hull-White model.
    """
    n_paths, n_times = rates.shape
    values = np.empty((n_paths, n_times))

    for i in range(n_times):
        values[:, i] = price_swap_at_time_index_hull_white(
            swap=swap,
            time_index=i,
            time_grid=time_grid,
            rates=rates,
            curve=curve,
            a=a,
            sigma=sigma,
        )

    return values

def par_swap_rate_from_curve(
    maturity: float,
    payment_frequency: int,
    curve,
) -> float:
    """
    Compute the par fixed rate for a plain-vanilla interest rate swap
    using the initial zero curve.

    Formula
    -------
        K_par = (1 - P(0,T_n)) / sum_i alpha_i P(0,T_i)

    where:
        - T_i are the fixed-leg payment dates
        - alpha_i is the accrual factor
        - P(0,T_i) are discount factors from the initial zero curve

    Why this matters
    ----------------
    If the fixed rate is set equal to the par swap rate, the initial
    value of the swap should be close to zero. This is the right setup
    for exposure simulation because the trade starts approximately
    at-market.
    """
    if maturity <= 0:
        raise ValueError("maturity must be positive.")

    if payment_frequency <= 0:
        raise ValueError("payment_frequency must be positive.")

    alpha = 1.0 / payment_frequency
    n_payments = int(round(maturity * payment_frequency))
    payment_times = np.arange(1, n_payments + 1) * alpha

    discount_factors = curve.discount_factor(payment_times)

    annuity = alpha * np.sum(discount_factors)
    maturity_discount_factor = discount_factors[-1]

    par_rate = (1.0 - maturity_discount_factor) / annuity

    return float(par_rate)