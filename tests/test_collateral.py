import numpy as np

from src.collateral.margining import (
    CollateralAgreement,
    compute_collateral_balance,
    collateralized_positive_exposure,
    collateralized_negative_exposure,
)


def test_perfect_collateral_eliminates_exposure():
    values = np.array([
        [0.0, 100.0, -50.0],
        [0.0, 200.0, -25.0],
    ])

    agreement = CollateralAgreement(
        threshold=0.0,
        minimum_transfer_amount=0.0,
        margin_lag_steps=0,
        bilateral=True,
    )

    collateral = compute_collateral_balance(values, agreement)

    pos = collateralized_positive_exposure(values, collateral)
    neg = collateralized_negative_exposure(values, collateral)

    assert np.isclose(pos.max(), 0.0)
    assert np.isclose(neg.max(), 0.0)