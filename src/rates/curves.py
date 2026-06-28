from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d


@dataclass
class ZeroCurve:
    """
    Simple continuously compounded zero curve.

    Parameters
    ----------
    maturities:
        Maturities in years.
    zero_rates:
        Continuously compounded zero rates in decimal form.
        Example: 2.5% should be 0.025.
    interpolation:
        Interpolation method. Use "linear" or "cubic".
    """
    maturities: np.ndarray
    zero_rates: np.ndarray
    interpolation: str = "cubic"

    def __post_init__(self) -> None:
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.zero_rates = np.asarray(self.zero_rates, dtype=float)

        if len(self.maturities) != len(self.zero_rates):
            raise ValueError("maturities and zero_rates must have the same length.")

        if np.any(self.maturities <= 0):
            raise ValueError("All maturities must be positive.")

        order = np.argsort(self.maturities)
        self.maturities = self.maturities[order]
        self.zero_rates = self.zero_rates[order]

        if self.interpolation == "linear":
            self._interp = interp1d(
                self.maturities,
                self.zero_rates,
                kind="linear",
                fill_value="extrapolate",
            )
        elif self.interpolation == "cubic":
            self._interp = CubicSpline(
                self.maturities,
                self.zero_rates,
                extrapolate=True,
            )
        else:
            raise ValueError("interpolation must be 'linear' or 'cubic'.")

    def zero_rate(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Return interpolated zero rate R(0,T).
        """
        T_arr = np.asarray(T, dtype=float)
        rates = self._interp(T_arr)

        if np.isscalar(T):
            return float(rates)

        return rates

    def discount_factor(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        Return P(0,T) = exp(-R(0,T) T).
        """
        T_arr = np.asarray(T, dtype=float)
        rates = self.zero_rate(T_arr)
        dfs = np.exp(-rates * T_arr)

        if np.isscalar(T):
            return float(dfs)

        return dfs

    def instantaneous_forward_rate(
        self,
        T: float | np.ndarray,
        eps: float = 1e-4,
    ) -> float | np.ndarray:
        """
        Approximate instantaneous forward rate:

            f(0,T) = - d ln P(0,T) / dT

        using a central finite difference.

        This is sufficient for the Hull-White drift construction.
        """
        T_arr = np.asarray(T, dtype=float)

        T_minus = np.maximum(T_arr - eps, 1e-8)
        T_plus = T_arr + eps

        log_p_plus = np.log(self.discount_factor(T_plus))
        log_p_minus = np.log(self.discount_factor(T_minus))

        forwards = -(log_p_plus - log_p_minus) / (T_plus - T_minus)

        if np.isscalar(T):
            return float(forwards)

        return forwards

    def forward_rate_derivative(
        self,
        T: float | np.ndarray,
        eps: float = 1e-4,
    ) -> float | np.ndarray:
        """
        Approximate df(0,T)/dT using central finite difference.
        """
        T_arr = np.asarray(T, dtype=float)

        T_minus = np.maximum(T_arr - eps, 1e-8)
        T_plus = T_arr + eps

        f_plus = self.instantaneous_forward_rate(T_plus)
        f_minus = self.instantaneous_forward_rate(T_minus)

        derivative = (f_plus - f_minus) / (T_plus - T_minus)

        if np.isscalar(T):
            return float(derivative)

        return derivative


def maturity_label_to_years(label: str) -> float:
    """
    Convert maturity labels like '3M', '1Y', '10Y' to years.
    """
    label = label.upper().strip()

    if label.endswith("M"):
        return float(label[:-1]) / 12.0

    if label.endswith("Y"):
        return float(label[:-1])

    raise ValueError(f"Unsupported maturity label: {label}")


def build_zero_curve_from_ecb_row(
    row: pd.Series,
    maturity_labels: Iterable[str] = ("3M", "1Y", "2Y", "5Y", "7Y", "10Y"),
    interpolation: str = "cubic",
) -> ZeroCurve:
    """
    Build a ZeroCurve object from one row of ECB spot yield curve data.

    Expected column names:
        spot_3M, spot_1Y, spot_2Y, ...

    Rates should already be in decimal form.
    """
    maturities = []
    zero_rates = []

    for label in maturity_labels:
        col = f"spot_{label}"

        if col not in row.index:
            raise KeyError(f"Missing column: {col}")

        value = row[col]

        if pd.notna(value):
            maturities.append(maturity_label_to_years(label))
            zero_rates.append(float(value))

    return ZeroCurve(
        maturities=np.array(maturities),
        zero_rates=np.array(zero_rates),
        interpolation=interpolation,
    )


def load_latest_ecb_zero_curve(
    csv_path: str,
    maturity_labels: Iterable[str] = ("3M", "1Y", "2Y", "5Y", "7Y", "10Y"),
    interpolation: str = "cubic",
    rates_are_percent: bool = True,
) -> ZeroCurve:
    """
    Load latest available ECB spot curve from CSV and build a ZeroCurve.

    Parameters
    ----------
    csv_path:
        Path to ECB yield curve CSV.
    rates_are_percent:
        If True, convert values like 2.5 to 0.025.
        If False, assume already decimal.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").dropna(subset=[f"spot_{m}" for m in maturity_labels])

    if df.empty:
        raise ValueError("No valid rows found for requested maturities.")

    latest = df.iloc[-1].copy()

    if rates_are_percent:
        for label in maturity_labels:
            latest[f"spot_{label}"] = latest[f"spot_{label}"] / 100.0

    return build_zero_curve_from_ecb_row(
        latest,
        maturity_labels=maturity_labels,
        interpolation=interpolation,
    )