from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class VasicekParams:
    a: float
    b: float
    sigma: float
    phi: float
    c: float
    residual_std: float
    delta_t: float


def prepare_short_rate_data(
    df: pd.DataFrame,
    date_col: str = "date",
    rate_col: str = "spot_3M",
) -> pd.DataFrame:
    """
    Prepare a short-rate time series for Vasicek calibration.

    Rates must be in decimals, not percentages.
    Example: 2.5% should be 0.025.
    """
    out = df[[date_col, rate_col]].copy()
    out = out.rename(columns={date_col: "date", rate_col: "short_rate"})
    out["date"] = pd.to_datetime(out["date"])
    out["short_rate"] = pd.to_numeric(out["short_rate"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)

    return out


def estimate_vasicek_ols(
    rates: pd.Series,
    delta_t: float = 1 / 252,
) -> VasicekParams:
    """
    Estimate Vasicek parameters using AR(1):

        r_{t+dt} = c + phi r_t + epsilon_t

    Then recover:
        a = -log(phi) / dt
        b = c / (1 - phi)
        sigma = std(epsilon) * sqrt(2a / (1 - exp(-2a dt)))
    """
    r_t = rates.iloc[:-1].to_numpy()
    r_next = rates.iloc[1:].to_numpy()

    X = sm.add_constant(r_t)
    model = sm.OLS(r_next, X).fit()

    c = float(model.params[0])
    phi = float(model.params[1])

    if phi <= 0 or phi >= 1:
        raise ValueError(
            f"Estimated phi={phi:.6f}. For stationary Vasicek, phi should be in (0, 1)."
        )

    a = -np.log(phi) / delta_t
    b = c / (1 - phi)

    residuals = model.resid
    residual_std = float(np.std(residuals, ddof=1))

    sigma = residual_std * np.sqrt(
        2 * a / (1 - np.exp(-2 * a * delta_t))
    )

    return VasicekParams(
        a=a,
        b=b,
        sigma=sigma,
        phi=phi,
        c=c,
        residual_std=residual_std,
        delta_t=delta_t,
    )

def estimate_historical_hw_parameters(
    rates: pd.Series,
    delta_t: float = 1 / 252,
) -> VasicekParams:
    """
    Estimate historical mean-reversion and volatility parameters for a
    one-factor Hull-White model using the AR(1) representation of a
    Gaussian mean-reverting short-rate process.

    The Hull-White model is an extended Vasicek model:

        dr_t = [theta(t) - a r_t] dt + sigma dW_t

    where theta(t) is chosen to fit the initial term structure. The AR(1)
    estimation provides historical estimates of a and sigma. The constant
    long-run mean b from the Vasicek representation is not used in the
    Hull-White drift, because Hull-White replaces it with theta(t).

    This is a historical/statistical calibration, not a full market
    calibration to swaptions or caps/floors.
    """
    return estimate_vasicek_ols(rates=rates, delta_t=delta_t)
