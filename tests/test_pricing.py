"""Price optimisation against the Lerner closed form and a numeric grid search."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from elasticity.pricing import (
    PriceBand,
    evaluate_policy,
    expected_profit,
    lerner_price,
    recommend_price,
    status_quo,
)
from elasticity.simulate import SimConfig, make_design


@pytest.mark.parametrize("eps", [-1.2, -1.8, -2.5, -4.0])
def test_lerner_price_matches_numeric_optimum(eps):
    cost, p0, vol = 2.0, 3.0, 100.0
    numeric = minimize_scalar(
        lambda p: -expected_profit(p, cost, eps, p0, vol), bounds=(cost, 50 * cost),
        method="bounded", options={"xatol": 1e-10},
    ).x
    assert lerner_price(cost, eps) == pytest.approx(numeric, rel=1e-6)
    # Lerner condition: markup over price equals -1 / elasticity
    p = lerner_price(cost, eps)
    assert (p - cost) / p == pytest.approx(-1 / eps)


@pytest.mark.parametrize("eps", [-0.5, -1.0, 0.3])
def test_no_interior_optimum_when_inelastic(eps):
    assert np.isinf(lerner_price(1.0, eps))
    price, flagged = recommend_price(np.array([1.0]), np.array([eps]), np.array([1.5]))
    assert flagged[0]
    assert price[0] == pytest.approx(PriceBand().upper * 1.5)


@pytest.mark.parametrize("eps", [-0.6, -1.05, -1.5, -2.2, -3.0, -6.0])
def test_recommended_price_is_the_grid_optimum_within_band(eps):
    cost, p0, vol, band = 1.0, 1.6, 50.0, PriceBand(0.8, 1.25)
    grid = np.linspace(band.lower * p0, band.upper * p0, 200_001)
    best = grid[np.argmax(expected_profit(grid, cost, eps, p0, vol))]
    price, _ = recommend_price(np.array([cost]), np.array([eps]), np.array([p0]), band)
    assert price[0] == pytest.approx(best, abs=2e-5)


def test_policy_with_true_elasticities_matches_oracle_and_bias_costs_profit():
    design = make_design(SimConfig(n_stores=3, products_per_category=3), seed=1)
    truth = design.category_elasticity
    assert evaluate_policy(design, truth).share_of_oracle == pytest.approx(1.0)
    biased = evaluate_policy(design, truth * 0.5)  # elasticities biased towards zero
    assert biased.share_of_oracle < 0.99
    assert biased.flagged.any()  # halving -1.2 and -1.8 gives |eps| < 1
    assert 0 < status_quo(design).share_of_oracle <= 1.0
