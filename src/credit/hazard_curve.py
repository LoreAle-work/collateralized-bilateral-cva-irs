from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PiecewiseHazardCurve:
    """
    Piecewise-constant hazard-rate curve.

    The hazard rate lambda(t) is assumed constant inside maturity buckets:

        lambda(t) = lambda_i,  for t_{i-1} < t <= t_i

    This lets us compute survival probabilities:

        Q(0,t) = exp(- integral_0^t lambda(u) du)

    In discrete piecewise form:

        Q(0,t_n) = exp(- sum_i lambda_i * delta_t_i)

    Parameters
    ----------
    maturities:
        Array of maturities in years, e.g. [1, 3, 5, 7, 10].

    hazard_rates:
        Array of hazard rates for each maturity bucket, in decimal form.
        Example: 2% hazard rate should be 0.02.

    recovery_rate:
        Assumed recovery rate after default.
        Example: 40% recovery should be 0.40.
    """

    maturities: np.ndarray
    hazard_rates: np.ndarray
    recovery_rate: float = 0.40

    def __post_init__(self) -> None:
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.hazard_rates = np.asarray(self.hazard_rates, dtype=float)

        if len(self.maturities) != len(self.hazard_rates):
            raise ValueError("maturities and hazard_rates must have the same length.")

        if np.any(self.maturities <= 0):
            raise ValueError("All maturities must be positive.")

        if np.any(self.hazard_rates < 0):
            raise ValueError("Hazard rates cannot be negative.")

        if not 0 <= self.recovery_rate < 1:
            raise ValueError("recovery_rate must be in [0, 1).")

        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.hazard_rates = self.hazard_rates[order]

    @property
    def loss_given_default(self) -> float:
        """
        Loss given default:

            LGD = 1 - recovery_rate
        """
        return 1.0 - self.recovery_rate

    def hazard_rate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Return the piecewise-constant hazard rate at time t.

        For t beyond the final maturity, we use the last available hazard rate.
        This is a simple flat extrapolation.
        """
        t_arr = np.asarray(t, dtype=float)

        indices = np.searchsorted(self.maturities, t_arr, side="left")
        indices = np.clip(indices, 0, len(self.hazard_rates) - 1)

        rates = self.hazard_rates[indices]

        if np.isscalar(t):
            return float(rates)

        return rates

    def cumulative_hazard(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Compute cumulative hazard:

            H(0,t) = integral_0^t lambda(u) du

        using the piecewise-constant hazard-rate curve.
        """
        t_arr = np.asarray(t, dtype=float)

        def _single_cumulative_hazard(x: float) -> float:
            if x <= 0:
                return 0.0

            total = 0.0
            previous = 0.0

            for maturity, hazard in zip(self.maturities, self.hazard_rates):
                interval_end = min(x, maturity)
                dt = interval_end - previous

                if dt > 0:
                    total += hazard * dt

                previous = maturity

                if x <= maturity:
                    return total

            # If x is beyond the last maturity, extrapolate with last hazard rate.
            total += self.hazard_rates[-1] * (x - self.maturities[-1])

            return total

        if np.isscalar(t):
            return float(_single_cumulative_hazard(float(t_arr)))

        return np.array([_single_cumulative_hazard(float(x)) for x in t_arr])

    def survival_probability(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Compute survival probability:

            Q(0,t) = exp(-H(0,t))

        where H(0,t) is cumulative hazard.
        """
        cumulative = self.cumulative_hazard(t)
        survival = np.exp(-cumulative)

        if np.isscalar(t):
            return float(survival)

        return survival

    def marginal_default_probability(
        self,
        t_start: float | np.ndarray,
        t_end: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Compute marginal default probability between t_start and t_end:

            PD(t_start, t_end) = Q(0,t_start) - Q(0,t_end)

        This is the probability of default occurring in the interval,
        under the deterministic hazard-rate curve.
        """
        q_start = self.survival_probability(t_start)
        q_end = self.survival_probability(t_end)

        return q_start - q_end


def hazard_rates_from_spreads(
    spreads: np.ndarray,
    recovery_rate: float,
) -> np.ndarray:
    """
    Convert credit spreads into approximate hazard rates.

    Approximation
    -------------
        spread ≈ lambda * (1 - recovery)

    so:

        lambda ≈ spread / (1 - recovery)

    Parameters
    ----------
    spreads:
        Credit spreads in decimal form.
        Example: 100 bps should be 0.01.

    recovery_rate:
        Recovery rate in decimal form.
        Example: 40% should be 0.40.

    Returns
    -------
    np.ndarray
        Approximate hazard rates.

    Important
    ---------
    This is a simplified approximation, not full CDS bootstrapping.
    It is acceptable for a transparent educational CVA engine, but we
    document it as a model limitation.
    """
    spreads = np.asarray(spreads, dtype=float)

    if not 0 <= recovery_rate < 1:
        raise ValueError("recovery_rate must be in [0, 1).")

    if np.any(spreads < 0):
        raise ValueError("spreads cannot be negative.")

    lgd = 1.0 - recovery_rate

    return spreads / lgd


def build_hazard_curve_from_spread_table(
    spread_table: pd.DataFrame,
    spread_column: str,
    maturity_column: str = "maturity_years",
    recovery_rate: float = 0.40,
) -> PiecewiseHazardCurve:
    """
    Build a PiecewiseHazardCurve from a credit spread table.

    Expected input example
    ----------------------
        maturity_years | counterparty_base | self_base
              1        |      0.0050       |  0.0040
              3        |      0.0080       |  0.0065
              5        |      0.0120       |  0.0100

    The chosen spread column should be in decimal form, not bps.

    Why this exists
    ---------------
    CVA/DVA requires survival probabilities and marginal default
    probabilities. Credit spreads are easier to specify, so this function
    converts spread scenarios into hazard curves.
    """
    if maturity_column not in spread_table.columns:
        raise KeyError(f"Missing maturity column: {maturity_column}")

    if spread_column not in spread_table.columns:
        raise KeyError(f"Missing spread column: {spread_column}")

    maturities = spread_table[maturity_column].to_numpy(dtype=float)
    spreads = spread_table[spread_column].to_numpy(dtype=float)

    hazard_rates = hazard_rates_from_spreads(
        spreads=spreads,
        recovery_rate=recovery_rate,
    )

    return PiecewiseHazardCurve(
        maturities=maturities,
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
    )


def load_credit_spread_scenarios(csv_path: str) -> pd.DataFrame:
    """
    Load the stylized credit spread scenario file.

    The raw CSV stores spreads in basis points, such as:

        counterparty_base_bps = 120

    This function converts every *_bps column into a decimal spread column:

        counterparty_base = 0.0120

    Why this matters
    ----------------
    Hazard-rate formulas require spreads in decimal form, not basis points.
    Converting once in a loader reduces unit mistakes later. Unit mistakes
    are how models commit tiny financial arson.
    """
    df = pd.read_csv(csv_path)

    bps_columns = [col for col in df.columns if col.endswith("_bps")]

    for col in bps_columns:
        decimal_col = col.replace("_bps", "")
        df[decimal_col] = df[col] / 10_000.0

    return df


def credit_curve_summary(
    curve: PiecewiseHazardCurve,
    time_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Create a summary table of hazard rates, survival probabilities,
    cumulative default probabilities, and marginal default probabilities.

    This is useful for notebook inspection and README/report tables.
    """
    time_grid = np.asarray(time_grid, dtype=float)

    survival = curve.survival_probability(time_grid)
    cumulative_pd = 1.0 - survival

    marginal_pd = np.zeros_like(time_grid)
    marginal_pd[1:] = curve.marginal_default_probability(
        time_grid[:-1],
        time_grid[1:],
    )

    return pd.DataFrame({
        "time": time_grid,
        "hazard_rate": curve.hazard_rate(time_grid),
        "survival_probability": survival,
        "cumulative_default_probability": cumulative_pd,
        "marginal_default_probability": marginal_pd,
    })