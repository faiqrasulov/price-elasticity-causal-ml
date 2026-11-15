"""The simulator: structure, determinism, and the true elasticity without confounding."""

import numpy as np
import pandas as pd
import pytest

from elasticity.estimators import iv_2sls, naive_ols, twfe
from elasticity.simulate import CATEGORIES, SimConfig, make_design, simulate_panel

SMALL = SimConfig(n_stores=3, products_per_category=2, n_weeks=20)

# No confounding: prices react to nothing related to demand, promotions have no display or
# feature lift, and products within a category share the same regular price and base volume.
NO_CONFOUNDING = SimConfig(
    n_stores=10,
    products_per_category=4,
    n_weeks=104,
    price_response=0.0,
    hidden_confounding=0.0,
    display_lift=0.0,
    feature_lift=0.0,
    price_dispersion=0.0,
    brand_volume=0.0,
    volume_sd=0.0,
)


def test_panel_is_balanced_deterministic_and_hides_the_truth():
    design = make_design(SMALL, seed=0)
    a = simulate_panel(design, SMALL, seed=5)
    b = simulate_panel(design, SMALL, seed=5)
    pd.testing.assert_frame_equal(a.data, b.data)
    n = SMALL.n_stores * len(CATEGORIES) * SMALL.products_per_category * SMALL.n_weeks
    assert len(a.data) == n
    assert not a.data.duplicated(["store", "product", "week"]).any()
    assert not a.data.isna().any().any()
    assert "u" not in a.data and "elasticity" not in a.data
    assert len(a.hidden_signal) == n
    assert not np.allclose(a.data["log_qty"], simulate_panel(design, SMALL, seed=6).data["log_qty"])


def test_design_truth_matches_category_parameters():
    true_cat = [c.elasticity for c in CATEGORIES]
    np.testing.assert_allclose(make_design(SMALL, seed=0).category_elasticity, true_cat)
    # with product-level heterogeneity the category mean is still the category parameter
    hetero = make_design(SMALL.with_(products_per_category=5, elasticity_sd=0.2), seed=0)
    np.testing.assert_allclose(hetero.category_elasticity, true_cat)
    assert hetero.elasticity.std() > 0.2


@pytest.fixture(scope="module")
def unconfounded():
    design = make_design(NO_CONFOUNDING, seed=3)
    return design, simulate_panel(design, NO_CONFOUNDING, seed=4).data


def test_ols_recovers_true_elasticity_without_confounding(unconfounded):
    design, df = unconfounded
    truth = design.category_elasticity
    for est in (naive_ols(df, len(truth)), twfe(df, len(truth))):
        err = est.by_category - truth
        assert np.all(np.abs(err) < 4 * est.by_category_se), err
        assert np.all(np.abs(err) < 0.15), err
    fe = twfe(df, len(truth))
    assert abs(fe.pooled - design.average_elasticity) < 4 * fe.pooled_se


def test_confounded_pricing_biases_ols_but_not_iv():
    cfg = NO_CONFOUNDING.with_(price_response=0.3, hidden_confounding=1.0)
    design = make_design(cfg, seed=3)
    df = simulate_panel(design, cfg, seed=4).data
    fe, iv = twfe(df, 5), iv_2sls(df, 5)
    assert fe.pooled - design.average_elasticity > 0.3  # biased towards zero
    assert abs(iv.pooled - design.average_elasticity) < 4 * iv.pooled_se
    assert iv.diagnostics["first_stage_f"] > 100
