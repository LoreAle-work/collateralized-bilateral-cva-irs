import numpy as np

from src.derivatives.swap import (
    InterestRateSwap,
    par_swap_rate_from_curve,
    price_swap_exposure_cube_hull_white,
)
from src.rates.curves import ZeroCurve


def test_par_swap_starts_at_zero_under_flat_curve():
    """
    A par payer swap should have zero initial value.

    This test uses a flat zero curve and deterministic flat short-rate paths.
    At t=0, all paths should give the same swap value, and that value should
    be approximately zero when the fixed rate is set to the par swap rate.
    """
    curve = ZeroCurve(
        maturities=np.array([0.5, 1.0, 2.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
    )

    notional = 1_000_000
    maturity = 5.0
    payment_frequency = 2

    fixed_rate = par_swap_rate_from_curve(
        maturity=maturity,
        payment_frequency=payment_frequency,
        curve=curve,
    )

    swap = InterestRateSwap(
        notional=notional,
        fixed_rate=fixed_rate,
        maturity=maturity,
        payment_frequency=payment_frequency,
        swap_type="payer",
    )

    time_grid = np.linspace(0.0, maturity, 51)

    # Deterministic flat rate paths make the test simple and reproducible.
    rates = np.full((10, len(time_grid)), 0.02)

    values = price_swap_exposure_cube_hull_white(
        swap=swap,
        time_grid=time_grid,
        rates=rates,
        curve=curve,
        a=0.10,
        sigma=0.01,
    )

    assert np.isclose(values[:, 0].mean(), 0.0, atol=1e-6)
    assert np.isclose(values[:, 0].std(), 0.0, atol=1e-12)


def test_off_market_payer_swap_above_par_has_negative_initial_value():
    """
    A payer swap paying a fixed rate above par should have negative value
    at inception.

    This confirms that the pricing function is not artificially forcing all
    initial swap values to zero. Only the par swap should start at zero.
    """
    curve = ZeroCurve(
        maturities=np.array([0.5, 1.0, 2.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
    )

    notional = 1_000_000
    maturity = 5.0
    payment_frequency = 2

    par_rate = par_swap_rate_from_curve(
        maturity=maturity,
        payment_frequency=payment_frequency,
        curve=curve,
    )

    off_market_swap = InterestRateSwap(
        notional=notional,
        fixed_rate=par_rate + 0.005,
        maturity=maturity,
        payment_frequency=payment_frequency,
        swap_type="payer",
    )

    time_grid = np.linspace(0.0, maturity, 51)
    rates = np.full((10, len(time_grid)), 0.02)

    values = price_swap_exposure_cube_hull_white(
        swap=off_market_swap,
        time_grid=time_grid,
        rates=rates,
        curve=curve,
        a=0.10,
        sigma=0.01,
    )

    assert values[:, 0].mean() < 0.0
    assert np.isclose(values[:, 0].std(), 0.0, atol=1e-12)


def test_receiver_swap_is_negative_of_payer_swap():
    """
    For the same fixed rate and terms, a receiver swap should be the negative
    of a payer swap.

    This checks the swap_type sign convention. Tiny revolutionary idea:
    signs should mean something.
    """
    curve = ZeroCurve(
        maturities=np.array([0.5, 1.0, 2.0, 3.0, 5.0]),
        zero_rates=np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
    )

    notional = 1_000_000
    maturity = 5.0
    payment_frequency = 2

    fixed_rate = par_swap_rate_from_curve(
        maturity=maturity,
        payment_frequency=payment_frequency,
        curve=curve,
    )

    payer_swap = InterestRateSwap(
        notional=notional,
        fixed_rate=fixed_rate + 0.001,
        maturity=maturity,
        payment_frequency=payment_frequency,
        swap_type="payer",
    )

    receiver_swap = InterestRateSwap(
        notional=notional,
        fixed_rate=fixed_rate + 0.001,
        maturity=maturity,
        payment_frequency=payment_frequency,
        swap_type="receiver",
    )

    time_grid = np.linspace(0.0, maturity, 51)
    rates = np.full((10, len(time_grid)), 0.02)

    payer_values = price_swap_exposure_cube_hull_white(
        swap=payer_swap,
        time_grid=time_grid,
        rates=rates,
        curve=curve,
        a=0.10,
        sigma=0.01,
    )

    receiver_values = price_swap_exposure_cube_hull_white(
        swap=receiver_swap,
        time_grid=time_grid,
        rates=rates,
        curve=curve,
        a=0.10,
        sigma=0.01,
    )

    assert np.allclose(receiver_values, -payer_values, atol=1e-8)