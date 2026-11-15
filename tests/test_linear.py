"""OLS, 2SLS and the within transformation against statsmodels and closed forms."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from elasticity.linear import ols, tsls, within_transform


@pytest.fixture
def unbalanced_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n_unit, n_t = 30, 12
    unit = np.repeat(np.arange(n_unit), n_t)
    week = np.tile(np.arange(n_t), n_unit)
    cluster = unit // 3
    a = rng.normal(size=n_unit)[unit]
    b = rng.normal(size=n_t)[week]
    x = 0.8 * a + 0.5 * b + rng.normal(size=len(unit))
    w = rng.normal(size=len(unit))
    y = -1.3 * x + 0.4 * w + 2 * a + b + rng.normal(size=len(unit))
    df = pd.DataFrame({"unit": unit, "week": week, "cluster": cluster, "x": x, "w": w, "y": y})
    keep = rng.random(len(df)) > 0.15  # drop rows -> unbalanced, needs iterated demeaning
    return df.loc[keep].reset_index(drop=True)


def _dummies(codes: np.ndarray) -> np.ndarray:
    return pd.get_dummies(codes, drop_first=True, dtype=np.float64).to_numpy()


def test_within_transform_equals_dummy_regression(unbalanced_panel):
    df = unbalanced_panel
    y, xw = df["y"].to_numpy(), df[["x", "w"]].to_numpy()
    fe = [df["unit"].to_numpy(), df["week"].to_numpy()]
    dm = within_transform(np.column_stack([y, xw]), fe)
    dummies = np.column_stack([_dummies(fe[0]), _dummies(fe[1])])
    ref = sm.OLS(y, sm.add_constant(np.column_stack([xw, dummies]))).fit(
        cov_type="cluster", cov_kwds={"groups": df["cluster"].to_numpy()}
    )
    n_absorbed = dummies.shape[1] + 1  # dummies + constant
    res = ols(dm[:, 0], dm[:, 1:], clusters=df["cluster"].to_numpy(), df_absorbed=n_absorbed)
    np.testing.assert_allclose(res.coef, ref.params[1:3], rtol=1e-8)
    np.testing.assert_allclose(res.se, ref.bse[1:3], rtol=1e-6)


def test_within_transform_balanced_one_sweep_matches_closed_form():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, 5))
    unit, week = np.repeat(np.arange(8), 5), np.tile(np.arange(5), 8)
    closed = v - v.mean(1, keepdims=True) - v.mean(0, keepdims=True) + v.mean()
    np.testing.assert_allclose(within_transform(v.reshape(-1), [unit, week]), closed.reshape(-1))


def test_ols_robust_errors_match_statsmodels(unbalanced_panel):
    df = unbalanced_panel
    x = sm.add_constant(df[["x", "w"]].to_numpy())
    y, g = df["y"].to_numpy(), df["cluster"].to_numpy()
    cl = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": g})
    hc1 = sm.OLS(y, x).fit(cov_type="HC1")
    np.testing.assert_allclose(ols(y, x, clusters=g).se, cl.bse, rtol=1e-8)
    np.testing.assert_allclose(ols(y, x).se, hc1.bse, rtol=1e-8)
    np.testing.assert_allclose(ols(y, x).coef, cl.params, rtol=1e-10)


@pytest.fixture
def iv_data():
    rng = np.random.default_rng(11)
    n = 4000
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    conf = rng.normal(size=n)
    d = 0.7 * z + 0.5 * w + conf + 0.5 * rng.normal(size=n)
    y = -2.0 * d + 0.3 * w + 1.5 * conf + rng.normal(size=n)
    groups = np.arange(n) // 20
    return y, d, z, w, groups


def test_tsls_matches_manual_two_stage(iv_data):
    y, d, z, w, groups = iv_data
    exog = sm.add_constant(w)
    first = sm.OLS(d, np.column_stack([z, exog])).fit()
    second = sm.OLS(y, np.column_stack([first.fittedvalues, exog])).fit()
    res = tsls(y, d, z, exog=exog, clusters=groups)
    np.testing.assert_allclose(res.coef, second.params, rtol=1e-10)
    # closed form in matrix notation: (X'Pz X)^-1 X'Pz y
    x_mat, z_mat = np.column_stack([d, exog]), np.column_stack([z, exog])
    pz = z_mat @ np.linalg.solve(z_mat.T @ z_mat, z_mat.T)
    np.testing.assert_allclose(res.coef, np.linalg.solve(x_mat.T @ pz @ x_mat, x_mat.T @ pz @ y))
    # IV removes the confounding that biases OLS
    assert abs(res.coef[0] + 2.0) < 4 * res.se[0]
    assert ols(y, np.column_stack([d, exog])).coef[0] > -1.6


def test_tsls_first_stage_f_is_squared_robust_t(iv_data):
    y, d, z, w, groups = iv_data
    exog = sm.add_constant(w)
    first = sm.OLS(d, np.column_stack([z, exog])).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    res = tsls(y, d, z, exog=exog, clusters=groups)
    np.testing.assert_allclose(res.first_stage_f[0], first.tvalues[0] ** 2, rtol=1e-8)


def test_tsls_just_identified_ratio_without_controls(iv_data):
    y, d, z, _, _ = iv_data
    res = tsls(y - y.mean(), d - d.mean(), z - z.mean())
    np.testing.assert_allclose(res.coef[0], np.cov(z, y)[0, 1] / np.cov(z, d)[0, 1], rtol=1e-10)
