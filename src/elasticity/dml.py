"""Double/debiased machine learning for the partially linear model (Chernozhukov et al. 2018).

    log q = theta * log p + g(X) + e,     log p = m(X) + v

Nuisances l(X) = E[log q | X] and m(X) = E[log p | X] are fitted with K-fold cross-fitting
(folds are whole clusters of observations, so no unit is used to predict itself); theta solves
the Neyman-orthogonal partialling-out moment E[(Y - l - theta (D - m)) (D - m)] = 0, i.e. it is
the slope of the outcome residual on the price residual. Heterogeneous effects by group use
the same residuals with one slope per group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import RegressorMixin, clone

from elasticity.linear import ols


@dataclass(frozen=True)
class DMLResult:
    """Cross-fitted residuals and the final-stage estimates."""

    theta: float
    se: float
    theta_by_group: np.ndarray
    se_by_group: np.ndarray
    y_resid: np.ndarray
    d_resid: np.ndarray
    r2_outcome: float  # out-of-fold R^2 of the outcome nuisance
    r2_treatment: float  # out-of-fold R^2 of the price nuisance


def group_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign whole groups to ``n_folds`` folds at random (balanced by group count)."""
    uniq, codes = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    fold_of_group = rng.permutation(len(uniq)) % n_folds
    return fold_of_group[codes.reshape(-1)]


def cross_fit(
    x: np.ndarray,
    target: np.ndarray,
    learner: RegressorMixin,
    folds: np.ndarray,
) -> np.ndarray:
    """Out-of-fold predictions of ``target`` from ``x``."""
    pred = np.empty(len(target), dtype=np.float64)
    for k in np.unique(folds):
        test = folds == k
        model = clone(learner).fit(x[~test], target[~test])
        pred[test] = model.predict(x[test])
    return pred


def _r2(y: np.ndarray, resid: np.ndarray) -> float:
    return float(1 - resid.var() / y.var())


def dml_plr(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    outcome_learner: RegressorMixin,
    treatment_learner: RegressorMixin,
    fold_groups: np.ndarray,
    clusters: np.ndarray,
    by_group: np.ndarray | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> DMLResult:
    """Partially linear DML estimate of the effect of ``d`` on ``y`` controlling for ``x``.

    Standard errors come from the cluster-robust sandwich of the orthogonal score (equivalently,
    CR1 standard errors of the residual-on-residual regression).
    """
    folds = group_folds(fold_groups, n_folds, seed)
    y_res = y - cross_fit(x, y, outcome_learner, folds)
    d_res = d - cross_fit(x, d, treatment_learner, folds)

    pooled = ols(y_res, d_res[:, None], clusters=clusters)
    if by_group is None:
        theta_g = se_g = np.array([])
    else:
        codes = np.unique(by_group, return_inverse=True)[1].reshape(-1)
        design = np.zeros((len(y), codes.max() + 1))
        design[np.arange(len(y)), codes] = d_res
        res_g = ols(y_res, design, clusters=clusters)
        theta_g, se_g = res_g.coef, res_g.se
    return DMLResult(
        theta=float(pooled.coef[0]),
        se=float(pooled.se[0]),
        theta_by_group=theta_g,
        se_by_group=se_g,
        y_resid=y_res,
        d_resid=d_res,
        r2_outcome=_r2(y, y_res),
        r2_treatment=_r2(d, d_res),
    )
