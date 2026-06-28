import numpy as np

from src.evt.tail_exposure import fit_gpd_tail_quantile


def test_evt_quantile_returns_finite_value_for_positive_tail():
    rng = np.random.default_rng(42)
    exposures = rng.lognormal(mean=10.0, sigma=0.25, size=10_000)

    fit = fit_gpd_tail_quantile(
        exposures=exposures,
        time=1.0,
        target_quantile=0.999,
        threshold_quantile=0.95,
        min_exceedances=100,
    )

    assert np.isfinite(fit.evt_quantile)
    assert fit.evt_quantile > fit.threshold