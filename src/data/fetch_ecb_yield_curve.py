import io
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"


def fetch_ecb_series(
    dataflow: str,
    key: str,
    start_period: str = "2004-09-06",
    end_period: str | None = None,
) -> pd.DataFrame:
    """
    Fetch a single ECB time series from the ECB Data Portal API.

    Parameters
    ----------
    dataflow:
        ECB dataflow ID. For yield curves, use "YC".
    key:
        ECB SDMX series key.
        Example: "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
    start_period:
        Start date in YYYY-MM-DD format.
    end_period:
        Optional end date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, value.
    """
    url = f"{ECB_BASE_URL}/{dataflow}/{key}"

    params = {
        "startPeriod": start_period,
        "format": "csvdata",
    }

    if end_period is not None:
        params["endPeriod"] = end_period

    headers = {
        "Accept": "text/csv"
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    # ECB CSV columns can vary slightly, but usually include TIME_PERIOD and OBS_VALUE.
    if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        raise ValueError(
            f"Unexpected ECB response format. Columns returned: {df.columns.tolist()}"
        )

    out = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date").reset_index(drop=True)

    return out


def fetch_yield_curve_spot_rates(
    maturities: Iterable[str],
    start_period: str = "2004-09-06",
    end_period: str | None = None,
) -> pd.DataFrame:
    """
    Fetch multiple ECB AAA euro-area yield curve spot rates.

    Parameters
    ----------
    maturities:
        Iterable of maturity labels, e.g. ["3M", "1Y", "2Y", "5Y", "7Y", "10Y"].
    start_period:
        Start date in YYYY-MM-DD format.
    end_period:
        Optional end date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by date, with one column per maturity.
        Values are in percent per annum.
    """
    frames = []

    for maturity in maturities:
        key = f"B.U2.EUR.4F.G_N_A.SV_C_YM.SR_{maturity}"

        series = fetch_ecb_series(
            dataflow="YC",
            key=key,
            start_period=start_period,
            end_period=end_period,
        )

        series = series.rename(columns={"value": f"spot_{maturity}"})
        frames.append(series)

    df = frames[0]

    for frame in frames[1:]:
        df = df.merge(frame, on="date", how="outer")

    df = df.sort_values("date").reset_index(drop=True)

    return df


def main() -> None:
    maturities = ["3M", "1Y", "2Y", "5Y", "7Y", "10Y"]

    df = fetch_yield_curve_spot_rates(
        maturities=maturities,
        start_period="2004-09-06",
    )

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "ecb_aaa_spot_yield_curve.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved ECB yield curve data to {output_path}")
    print(df.tail())


if __name__ == "__main__":
    main()

    