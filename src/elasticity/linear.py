"""Linear building blocks: OLS and 2SLS with cluster-robust errors, and the within transformation.

Written in numpy (rather than calling statsmodels) because the Monte Carlo runs thousands of
regressions; the tests check every function against statsmodels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionResult:
    """Coefficients and cluster-robust (CR1) standard errors."""

    coef: np.ndarray
    se: np.ndarray
    n_obs: int
    n_clusters: int


@dataclass(frozen=True)
class IVResult(RegressionResult):
    """2SLS result plus the cluster-robust first-stage F statistic of each endogenous regressor."""

    first_stage_f: np.ndarray


def _codes(groups: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(groups, return_inverse=True)
    codes = codes.reshape(-1)
    return codes, int(codes.max()) + 1


def _group_sums(values: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Sum the rows of ``values`` (N, K) within groups -> (G, K)."""
    k = values.shape[1]
    return np.column_stack([np.bincount(codes, values[:, j], minlength=n_groups) for j in range(k)])


def sandwich_cov(
    x: np.ndarray,
    resid: np.ndarray,
    clusters: np.ndarray | None,
    bread: np.ndarray | None = None,
    df_absorbed: int = 0,
) -> np.ndarray:
    """Cluster-robust (CR1) or, without clusters, heteroskedasticity-robust (HC1) covariance.

    ``bread`` defaults to ``inv(x'x)``. ``df_absorbed`` counts parameters removed before the
    regression (fixed effects not nested in the clusters) for the small-sample correction.
    """
    n, k = x.shape
    if bread is None:
        bread = np.linalg.inv(x.T @ x)
    scores = x * resid[:, None]
    k_total = k + df_absorbed
    if clusters is None:
        meat = scores.T @ scores
        factor = n / (n - k_total)
    else:
        codes, g = _codes(clusters)
        sums = _group_sums(scores, codes, g)
        meat = sums.T @ sums
        factor = g / (g - 1) * (n - 1) / (n - k_total)
    return factor * bread @ meat @ bread


def ols(
    y: np.ndarray,
    x: np.ndarray,
    clusters: np.ndarray | None = None,
    df_absorbed: int = 0,
) -> RegressionResult:
    """Least squares of ``y`` on the columns of ``x`` (add a constant column yourself)."""
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ (x.T @ y)
    resid = y - x @ coef
    cov = sandwich_cov(x, resid, clusters, bread=xtx_inv, df_absorbed=df_absorbed)
    n_clusters = 0 if clusters is None else _codes(clusters)[1]
    return RegressionResult(coef, np.sqrt(np.diag(cov)), len(y), n_clusters)


def tsls(
    y: np.ndarray,
    endog: np.ndarray,
    instruments: np.ndarray,
    exog: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    df_absorbed: int = 0,
) -> IVResult:
    """Two-stage least squares; coefficients are ordered ``[endog..., exog...]``.

    ``instruments`` are the excluded instruments; ``exog`` the included controls (they act as
    their own instruments). The first-stage F is a cluster-robust Wald test of the excluded
    instruments in each first-stage regression, divided by their number.
    """
    endog = endog.reshape(len(y), -1)
    instruments = instruments.reshape(len(y), -1)
    exog = np.empty((len(y), 0)) if exog is None else exog.reshape(len(y), -1)
    z = np.column_stack([instruments, exog])
    x = np.column_stack([endog, exog])
    ztz_inv = np.linalg.inv(z.T @ z)
    pi = ztz_inv @ (z.T @ endog)  # first-stage coefficients
    endog_hat = z @ pi
    x_hat = np.column_stack([endog_hat, exog])
    bread = np.linalg.inv(x_hat.T @ x_hat)
    coef = bread @ (x_hat.T @ y)
    resid = y - x @ coef  # structural residual uses the actual regressors
    cov = sandwich_cov(x_hat, resid, clusters, bread=bread, df_absorbed=df_absorbed)

    k_inst = instruments.shape[1]
    f_stats = []
    for j in range(endog.shape[1]):
        v1 = endog[:, j] - endog_hat[:, j]
        cov_pi = sandwich_cov(z, v1, clusters, bread=ztz_inv, df_absorbed=df_absorbed)
        b = pi[:k_inst, j]
        f_stats.append(float(b @ np.linalg.solve(cov_pi[:k_inst, :k_inst], b)) / k_inst)
    n_clusters = 0 if clusters is None else _codes(clusters)[1]
    return IVResult(coef, np.sqrt(np.diag(cov)), len(y), n_clusters, np.array(f_stats))


def within_transform(
    values: np.ndarray,
    fixed_effects: Sequence[np.ndarray],
    tol: float = 1e-10,
    max_iter: int = 500,
) -> np.ndarray:
    """Remove one or more sets of fixed effects by alternating projections.

    Subtracts group means for each fixed-effect dimension in turn until nothing changes
    (exact after one sweep for a balanced two-way panel; also correct for unbalanced panels).
    By Frisch-Waugh-Lovell, OLS on the transformed data equals OLS with all the dummies.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    squeeze = out.ndim == 1
    if squeeze:
        out = out[:, None]
    groups = []
    for fe in fixed_effects:
        codes, g = _codes(fe)
        groups.append((codes, g, np.bincount(codes, minlength=g).astype(np.float64)))
    for _ in range(max_iter):
        max_change = 0.0
        for codes, g, counts in groups:
            means = _group_sums(out, codes, g) / counts[:, None]
            out -= means[codes]
            max_change = max(max_change, float(np.abs(means).max()))
        if max_change < tol:
            break
    else:
        raise RuntimeError("within_transform did not converge")
    return out[:, 0] if squeeze else out
