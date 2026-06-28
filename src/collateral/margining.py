from dataclasses import dataclass

import numpy as np


@dataclass
class CollateralAgreement:
    """
    Simplified collateral agreement for bilateral derivative exposure.

    Parameters
    ----------
    threshold:
        Unsecured exposure allowed before collateral is posted.
        Example: threshold = 100_000 means the first 100k of exposure
        remains uncollateralized.

    minimum_transfer_amount:
        Minimum collateral movement required before collateral is updated.
        Example: if MTA = 10_000, small collateral changes below 10k are ignored.

    margin_lag_steps:
        Number of simulation steps by which collateral is delayed.
        Example:
            margin_lag_steps = 0 means collateral is based on current MTM.
            margin_lag_steps = 5 means collateral is based on MTM from 5 steps ago.

    bilateral:
        If True, both parties post collateral depending on the sign of the MTM.
        If False, only positive exposure collateral is modeled.
    """

    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    margin_lag_steps: int = 0
    bilateral: bool = True

    def __post_init__(self) -> None:
        if self.threshold < 0:
            raise ValueError("threshold cannot be negative.")

        if self.minimum_transfer_amount < 0:
            raise ValueError("minimum_transfer_amount cannot be negative.")

        if self.margin_lag_steps < 0:
            raise ValueError("margin_lag_steps cannot be negative.")


def compute_collateral_balance(
    values: np.ndarray,
    agreement: CollateralAgreement,
) -> np.ndarray:
    """
    Compute pathwise collateral balances from simulated mark-to-market values.

    Parameters
    ----------
    values:
        Simulated derivative values V_m(t_i), shape (n_paths, n_times).
        Positive value means the counterparty owes the bank.
        Negative value means the bank owes the counterparty.

    agreement:
        CollateralAgreement object.

    Returns
    -------
    np.ndarray
        Collateral balance C_m(t_i), shape (n_paths, n_times).

        Convention:
        - positive collateral means collateral held by the bank
        - negative collateral means collateral posted by the bank

    Method
    ------
    Collateral is based on lagged mark-to-market values:

        V_lag(t_i) = V(t_i - lag)

    If V_lag is positive, the counterparty posts collateral to the bank:

        C(t_i) = max(V_lag - threshold, 0)

    If V_lag is negative and bilateral=True, the bank posts collateral:

        C(t_i) = -max(-V_lag - threshold, 0)

    The minimum transfer amount is implemented by only updating collateral
    when the change from the previous collateral balance exceeds the MTA.

    Why this matters
    ----------------
    CVA and DVA depend on exposure after collateral. Real collateral does not
    always perfectly match current mark-to-market because of thresholds, lag,
    and minimum transfer amounts. This function makes that imperfection explicit.
    """
    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape (n_paths, n_times).")

    n_paths, n_times = values.shape
    collateral = np.zeros_like(values)

    lag = agreement.margin_lag_steps

    for i in range(n_times):
        # Use lagged mark-to-market for collateral calculation.
        # If i - lag < 0, no collateral has been exchanged yet.
        lagged_index = i - lag

        if lagged_index < 0:
            target_collateral = np.zeros(n_paths)
        else:
            lagged_values = values[:, lagged_index]

            positive_collateral = np.maximum(
                lagged_values - agreement.threshold,
                0.0,
            )

            if agreement.bilateral:
                negative_collateral = np.maximum(
                    -lagged_values - agreement.threshold,
                    0.0,
                )

                target_collateral = positive_collateral - negative_collateral
            else:
                target_collateral = positive_collateral

        if i == 0:
            collateral[:, i] = target_collateral
        else:
            previous_collateral = collateral[:, i - 1]
            collateral_change = target_collateral - previous_collateral

            update_mask = (
                np.abs(collateral_change)
                >= agreement.minimum_transfer_amount
            )

            collateral[:, i] = previous_collateral
            collateral[update_mask, i] = target_collateral[update_mask]

    return collateral


def collateralized_positive_exposure(
    values: np.ndarray,
    collateral: np.ndarray,
) -> np.ndarray:
    """
    Compute collateralized positive exposure:

        E_c(t) = max(V(t) - C(t), 0)

    where positive collateral means collateral held by the bank.

    This feeds CVA.
    """
    if values.shape != collateral.shape:
        raise ValueError("values and collateral must have the same shape.")

    return np.maximum(values - collateral, 0.0)


def collateralized_negative_exposure(
    values: np.ndarray,
    collateral: np.ndarray,
) -> np.ndarray:
    """
    Compute collateralized negative exposure:

        NE_c(t) = max(-V(t) + C(t), 0)

    Explanation
    -----------
    If V(t) is negative, the bank owes the counterparty.

    If collateral is negative, it means the bank has posted collateral.
    That reduces the remaining negative exposure.

    Formula:

        NE_c(t) = max(-(V(t) - C(t)), 0)
                = max(-V(t) + C(t), 0)

    This feeds DVA.
    """
    if values.shape != collateral.shape:
        raise ValueError("values and collateral must have the same shape.")

    return np.maximum(-values + collateral, 0.0)


def exposure_reduction_summary(
    uncollateralized_positive: np.ndarray,
    collateralized_positive: np.ndarray,
    uncollateralized_negative: np.ndarray,
    collateralized_negative: np.ndarray,
) -> dict:
    """
    Summarize the effect of collateral on positive and negative exposure.

    This is useful for quickly checking whether the collateral agreement
    behaves as expected.
    """
    avg_ee_before = float(uncollateralized_positive.mean())
    avg_ee_after = float(collateralized_positive.mean())

    avg_ene_before = float(uncollateralized_negative.mean())
    avg_ene_after = float(collateralized_negative.mean())

    ee_reduction = (
        1.0 - avg_ee_after / avg_ee_before
        if avg_ee_before > 0
        else np.nan
    )

    ene_reduction = (
        1.0 - avg_ene_after / avg_ene_before
        if avg_ene_before > 0
        else np.nan
    )

    return {
        "average_positive_exposure_before": avg_ee_before,
        "average_positive_exposure_after": avg_ee_after,
        "positive_exposure_reduction_pct": ee_reduction,
        "average_negative_exposure_before": avg_ene_before,
        "average_negative_exposure_after": avg_ene_after,
        "negative_exposure_reduction_pct": ene_reduction,
    }


def compute_initial_margin_profile(
    values: np.ndarray,
    margin_period_steps: int = 10,
    quantile: float = 0.99,
) -> np.ndarray:
    """
    Compute a stylized initial margin profile from simulated future
    mark-to-market changes over a margin period of risk.

    Method
    ------
    For each time t_i, compute pathwise adverse positive changes:

        adverse_change_m(t_i) = max(V_m(t_{i+h}) - V_m(t_i), 0)

    Then define initial margin as the chosen quantile across paths:

        IM(t_i) = quantile(adverse_change_m(t_i))

    This is a simplified VaR-style initial margin proxy.

    Parameters
    ----------
    values:
        Simulated mark-to-market values, shape (n_paths, n_times).

    margin_period_steps:
        Number of simulation steps in the margin period of risk.

    quantile:
        Quantile level used for initial margin, e.g. 0.99.

    Returns
    -------
    np.ndarray
        Initial margin profile, shape (n_times,). Same value is applied
        across all paths at a given time.
    """
    if values.ndim != 2:
        raise ValueError("values must have shape (n_paths, n_times).")

    if margin_period_steps <= 0:
        raise ValueError("margin_period_steps must be positive.")

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")

    n_paths, n_times = values.shape
    im_profile = np.zeros(n_times)

    for i in range(n_times):
        end_index = min(i + margin_period_steps, n_times - 1)

        adverse_changes = np.maximum(
            values[:, end_index] - values[:, i],
            0.0,
        )

        im_profile[i] = np.quantile(adverse_changes, quantile)

    return im_profile


def collateralized_positive_exposure_with_initial_margin(
    values: np.ndarray,
    collateral: np.ndarray,
    initial_margin: np.ndarray,
) -> np.ndarray:
    """
    Compute collateralized positive exposure after variation margin and
    initial margin:

        E_c,IM(t) = max(V(t) - C(t) - IM(t), 0)

    Initial margin is treated as additional collateral held against
    counterparty default exposure.
    """
    if values.shape != collateral.shape:
        raise ValueError("values and collateral must have the same shape.")

    initial_margin = np.asarray(initial_margin, dtype=float)

    if initial_margin.ndim != 1:
        raise ValueError("initial_margin must be one-dimensional.")

    if initial_margin.shape[0] != values.shape[1]:
        raise ValueError("initial_margin length must match number of time points.")

    return np.maximum(
        values - collateral - initial_margin[None, :],
        0.0,
    )