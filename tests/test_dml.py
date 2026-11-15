"""Double ML: recovery of a known effect, fold construction, and the final-stage errors."""

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from elasticity.dml import dml_plr, group_folds

THETA = -1.5


@pytest.fixture(scope="module")
def plr_data():
    """Partially linear model with nonlinear confounding: D = m(X) + v, Y = theta D + g(X) + e."""
    rng = np.random.default_rng(42)
    n = 4000
    x = rng.uniform(-2, 2, size=(n, 4))
    m = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.5 * (x[:, 2] > 0)
    g = 1.5 * np.sin(2 * x[:, 0]) + x[:, 1] ** 2 + np.cos(x[:, 3])
    d = m + rng.normal(size=n)
    y = THETA * d + g + 0.5 * rng.normal(size=n)
    groups = np.arange(n) // 4
    category = (x[:, 3] > 0).astype(int)
    return y, d, x, groups, category


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=0
    )


def test_dml_recovers_known_effect(plr_data):
    y, d, x, groups, category = plr_data
    res = dml_plr(y, d, x, _hgb(), _hgb(), fold_groups=groups, clusters=groups,
                  by_group=category, n_folds=5, seed=0)
    assert abs(res.theta - THETA) < 0.06  # small finite-sample nuisance bias remains
    np.testing.assert_allclose(res.theta_by_group, THETA, atol=0.08)
    # the naive regression is badly biased in the same data
    naive = np.polyfit(d, y, 1)[0]
    assert abs(naive - THETA) > 10 * abs(res.theta - THETA)


def test_dml_with_misspecified_linear_nuisances_stays_biased(plr_data):
    y, d, x, groups, _ = plr_data
    res = dml_plr(y, d, x, LinearRegression(), LinearRegression(), fold_groups=groups,
                  clusters=groups, n_folds=5, seed=0)
    assert abs(res.theta - THETA) > 0.1  # orthogonality does not fix a wrong nuisance model


def test_final_stage_errors_equal_clustered_residual_regression(plr_data):
    y, d, x, groups, category = plr_data
    res = dml_plr(y, d, x, LinearRegression(), LinearRegression(), fold_groups=groups,
                  clusters=groups, by_group=category, n_folds=3, seed=1)
    ref = sm.OLS(res.y_resid, res.d_resid).fit(cov_type="cluster", cov_kwds={"groups": groups})
    assert res.theta == pytest.approx(ref.params[0], rel=1e-10)
    assert res.se == pytest.approx(ref.bse[0], rel=1e-8)
    # per-group slopes are the separate residual-on-residual slopes
    for c in (0, 1):
        sel = category == c
        slope = (res.d_resid[sel] @ res.y_resid[sel]) / (res.d_resid[sel] @ res.d_resid[sel])
        assert res.theta_by_group[c] == pytest.approx(slope, rel=1e-10)


def test_group_folds_keep_groups_together_and_are_balanced():
    groups = np.repeat(np.arange(103), 7)
    folds = group_folds(groups, 5, seed=3)
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1
    sizes = np.bincount(folds) // 7
    assert sizes.max() - sizes.min() <= 1
    np.testing.assert_array_equal(folds, group_folds(groups, 5, seed=3))
