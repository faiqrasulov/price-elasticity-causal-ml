"""The six elasticity estimators, each returning a pooled and a per-category estimate.

Every estimator reads only observed columns of the panel (never the hidden demand signal or
the true elasticities). Standard errors are clustered by product: wholesale-cost shocks,
promotions and feature ads are set chain-wide per product and week, so errors and price
variation are correlated across stores of the same product.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from elasticity.dml import dml_plr
from elasticity.linear import ols, tsls, within_transform

CLUSTER = "product"
TIME_VARYING_CONTROLS = ("promo", "display", "feature", "event", "temperature")
# Time-varying confounders for the DML learners (store x product effects are removed exactly by
# the within transformation first). Store and category let the trees learn store-specific
# weather and category-specific seasonality.
HGB_FEATURES = (
    "category",
    "store",
    "week_of_year",
    "holiday",
    "event",
    "temperature",
    "promo",
    "display",
    "feature",
)
HGB_CATEGORICAL = (0, 1)  # category, store


@dataclass(frozen=True)
class Estimate:
    """Pooled (average) elasticity and one elasticity per category, with standard errors."""

    pooled: float
    pooled_se: float
    by_category: np.ndarray
    by_category_se: np.ndarray
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DMLSettings:
    """Cross-fitting and learner settings for the two DML variants."""

    n_folds: int = 5
    hgb_max_iter: int = 100
    hgb_learning_rate: float = 0.2
    hgb_max_leaf_nodes: int = 31
    hgb_max_bins: int = 63
    seed: int = 0


def fourier(week_of_year: np.ndarray, order: int) -> np.ndarray:
    """Yearly Fourier terms sin/cos(2 pi k w / 52), k = 1..order."""
    angle = 2 * np.pi * np.asarray(week_of_year, dtype=np.float64)[:, None] / 52.0
    k = np.arange(1, order + 1)[None, :]
    return np.column_stack([np.sin(k * angle), np.cos(k * angle)])


def one_hot(codes: np.ndarray, n: int, drop_first: bool = False) -> np.ndarray:
    out = np.zeros((len(codes), n))
    out[np.arange(len(codes)), np.asarray(codes, dtype=np.int64)] = 1.0
    return out[:, 1:] if drop_first else out


def _interact(d: np.ndarray, category: np.ndarray, n_cat: int) -> np.ndarray:
    """Columns d * 1{category == c} for every category."""
    return one_hot(category, n_cat) * d[:, None]


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=np.float64)


def _estimate_from(
    pooled: tuple[np.ndarray, np.ndarray],
    by_cat: tuple[np.ndarray, np.ndarray],
    n_cat: int,
    **diagnostics: float,
) -> Estimate:
    return Estimate(
        pooled=float(pooled[0][0]),
        pooled_se=float(pooled[1][0]),
        by_category=np.asarray(by_cat[0][:n_cat], dtype=np.float64),
        by_category_se=np.asarray(by_cat[1][:n_cat], dtype=np.float64),
        diagnostics=dict(diagnostics),
    )


def naive_ols(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """log q ~ log p (pooled); per category: separate slope and intercept per category."""
    y, d, cat = _col(df, "log_qty"), _col(df, "log_price"), df["category"].to_numpy()
    cl = df[CLUSTER].to_numpy()
    const = np.ones((len(y), 1))
    r = ols(y, np.column_stack([d, const]), clusters=cl)
    rc = ols(y, np.column_stack([_interact(d, cat, n_cat), one_hot(cat, n_cat)]), clusters=cl)
    return _estimate_from((r.coef, r.se), (rc.coef, rc.se), n_cat)


def observed_controls(df: pd.DataFrame, n_cat: int) -> np.ndarray:
    """Flat-regression controls: promotions, holiday, event, weather, traffic, season, category."""
    cols = [_col(df, c) for c in ("promo", "display", "feature", "holiday", "event")]
    cols += [_col(df, "temperature"), _col(df, "log_traffic")]
    return np.column_stack(
        cols
        + [fourier(df["week_of_year"].to_numpy(), 2)]
        + [one_hot(df["category"].to_numpy(), n_cat, drop_first=True), np.ones(len(df))]
    )


def ols_controls(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """log q ~ log p + observed controls (no product or store fixed effects)."""
    y, d, cat = _col(df, "log_qty"), _col(df, "log_price"), df["category"].to_numpy()
    cl = df[CLUSTER].to_numpy()
    w = observed_controls(df, n_cat)
    r = ols(y, np.column_stack([d, w]), clusters=cl)
    rc = ols(y, np.column_stack([_interact(d, cat, n_cat), w]), clusters=cl)
    return _estimate_from((r.coef, r.se), (rc.coef, rc.se), n_cat)


def _within(df: pd.DataFrame, n_cat: int) -> dict[str, np.ndarray]:
    """Two-way within transformation (store x product and week fixed effects)."""
    cat = df["category"].to_numpy()
    d, z = _col(df, "log_price"), _col(df, "log_cost")
    blocks = {
        "y": _col(df, "log_qty")[:, None],
        "d": d[:, None],
        "d_cat": _interact(d, cat, n_cat),
        "w": np.column_stack([_col(df, c) for c in TIME_VARYING_CONTROLS]),
        "z": z[:, None],
        "z_cat": _interact(z, cat, n_cat),
    }
    unit = _unit(df)
    fixed_effects = [unit, df["week"].to_numpy()]
    stacked = within_transform(np.column_stack(list(blocks.values())), fixed_effects)
    out, start = {}, 0
    for name, block in blocks.items():
        out[name] = stacked[:, start : start + block.shape[1]]
        start += block.shape[1]
    out["y"] = out["y"][:, 0]
    return out


def twfe(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """Two-way fixed effects (store x product, week) + time-varying controls."""
    m = _within(df, n_cat)
    cl = df[CLUSTER].to_numpy()
    dof = int(df["week"].nunique()) - 1  # week effects are not nested in product clusters
    r = ols(m["y"], np.column_stack([m["d"], m["w"]]), clusters=cl, df_absorbed=dof)
    rc = ols(m["y"], np.column_stack([m["d_cat"], m["w"]]), clusters=cl, df_absorbed=dof)
    return _estimate_from((r.coef, r.se), (rc.coef, rc.se), n_cat)


def iv_2sls(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """2SLS with the wholesale cost as instrument, on top of the two-way fixed effects."""
    m = _within(df, n_cat)
    cl = df[CLUSTER].to_numpy()
    dof = int(df["week"].nunique()) - 1
    r = tsls(m["y"], m["d"], m["z"], exog=m["w"], clusters=cl, df_absorbed=dof)
    rc = tsls(m["y"], m["d_cat"], m["z_cat"], exog=m["w"], clusters=cl, df_absorbed=dof)
    return _estimate_from(
        (r.coef, r.se),
        (rc.coef, rc.se),
        n_cat,
        first_stage_f=float(r.first_stage_f[0]),
        min_category_first_stage_f=float(rc.first_stage_f.min()),
    )


def hgb_features(df: pd.DataFrame) -> np.ndarray:
    """Time-varying confounders for the boosting learner (store and category as categorical)."""
    return np.column_stack([_col(df, c) for c in HGB_FEATURES])


def lasso_features(df: pd.DataFrame, n_cat: int) -> np.ndarray:
    """Flexible linear basis: category-specific seasonality (Fourier), holiday, event and
    temperature splines, plus promotion, display and feature flags."""
    cat = one_hot(df["category"].to_numpy(), n_cat)
    temp = _col(df, "temperature")[:, None]
    # B-splines sum to one, so drop the last basis function to avoid exact collinearity.
    splines = SplineTransformer(n_knots=5, degree=3).fit_transform(temp)[:, :-1]
    per_cat = np.column_stack(
        [fourier(df["week_of_year"].to_numpy(), 3), _col(df, "holiday"), _col(df, "event"), splines]
    )
    interactions = (cat[:, :, None] * per_cat[:, None, :]).reshape(len(df), -1)
    flags = np.column_stack([_col(df, c) for c in ("promo", "display", "feature")])
    return np.column_stack([interactions, flags])


def _hgb(s: DMLSettings) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=s.hgb_max_iter,
        learning_rate=s.hgb_learning_rate,
        max_leaf_nodes=s.hgb_max_leaf_nodes,
        max_bins=s.hgb_max_bins,
        min_samples_leaf=20,
        early_stopping=False,
        categorical_features=list(HGB_CATEGORICAL),
        random_state=s.seed,
    )


def bic_lasso_alpha(x: np.ndarray, target: np.ndarray) -> float:
    """Lasso penalty chosen by BIC along the LARS path on the full sample (standardised x)."""
    return float(LassoLarsIC(criterion="bic").fit(StandardScaler().fit_transform(x), target).alpha_)


def _lasso(alpha: float):
    return make_pipeline(
        StandardScaler(), Lasso(alpha=alpha, precompute=True, max_iter=10_000, tol=1e-4)
    )


def _unit(df: pd.DataFrame) -> np.ndarray:
    return df["store"].to_numpy() * (int(df["product"].max()) + 1) + df["product"].to_numpy()


def _dml(
    df: pd.DataFrame,
    x: np.ndarray,
    outcome_learner,
    price_learner,
    s: DMLSettings,
    **diagnostics: float,
) -> Estimate:
    """Within-transform log q and log p by store x product, then cross-fit the nuisances with
    folds made of whole products (the cluster level, so shared cost shocks cannot leak from
    the training folds into the held-out fold)."""
    unit = _unit(df)
    res = dml_plr(
        y=within_transform(_col(df, "log_qty"), [unit]),
        d=within_transform(_col(df, "log_price"), [unit]),
        x=x,
        outcome_learner=outcome_learner,
        treatment_learner=price_learner,
        fold_groups=df[CLUSTER].to_numpy(),
        clusters=df[CLUSTER].to_numpy(),
        by_group=df["category"].to_numpy(),
        n_folds=s.n_folds,
        seed=s.seed,
    )
    return Estimate(
        pooled=res.theta,
        pooled_se=res.se,
        by_category=res.theta_by_group,
        by_category_se=res.se_by_group,
        diagnostics={"r2_outcome": res.r2_outcome, "r2_price": res.r2_treatment, **diagnostics},
    )


def dml_lasso(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """DML (partially linear) with lasso nuisances on a flexible, within-transformed basis.

    Each nuisance's penalty is chosen once by BIC on the full sample (like DoubleML's default
    of tuning outside the folds); the lasso is then refitted within every cross-fitting fold.
    """
    s = settings or DMLSettings()
    unit = _unit(df)
    x = within_transform(lasso_features(df, n_cat), [unit])
    a_y = bic_lasso_alpha(x, within_transform(_col(df, "log_qty"), [unit]))
    a_d = bic_lasso_alpha(x, within_transform(_col(df, "log_price"), [unit]))
    return _dml(df, x, _lasso(a_y), _lasso(a_d), s, alpha_outcome=a_y, alpha_price=a_d)


def dml_boosting(df: pd.DataFrame, n_cat: int, settings: DMLSettings | None = None) -> Estimate:
    """DML (partially linear) with histogram gradient-boosting nuisances."""
    s = settings or DMLSettings()
    return _dml(df, hgb_features(df), _hgb(s), _hgb(s), s)


EstimatorFn = Callable[[pd.DataFrame, int, DMLSettings | None], Estimate]

ESTIMATORS: dict[str, tuple[str, EstimatorFn]] = {
    "naive_ols": ("Naive OLS", naive_ols),
    "ols_controls": ("OLS + controls", ols_controls),
    "twfe": ("Two-way FE", twfe),
    "iv_2sls": ("2SLS (cost IV)", iv_2sls),
    "dml_lasso": ("DML (lasso)", dml_lasso),
    "dml_boosting": ("DML (boosting)", dml_boosting),
}
