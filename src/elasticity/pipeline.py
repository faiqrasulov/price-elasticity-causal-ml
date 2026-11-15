"""End-to-end experiments: one large simulated panel per scenario, then a Monte Carlo study.

Scenarios (same market, same random draws; only the pricing rule differs):

* ``observed``: the pricing manager reacts only to demand drivers that are in the data
  (seasonality, holidays, weather, local events); every confounder is observed.
* ``hidden``: the manager also reacts to a demand signal the analyst never sees.

``run`` writes ``metrics.json`` and the figures; every number in the README comes from it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from threadpoolctl import threadpool_limits

from elasticity.estimators import ESTIMATORS, DMLSettings, Estimate
from elasticity.pricing import PriceBand, evaluate_policy, status_quo
from elasticity.simulate import MarketDesign, SimConfig, make_design, simulate_panel

SCENARIOS: dict[str, float] = {"observed": 0.0, "hidden": 1.0}  # -> hidden_confounding
Z95 = 1.959963984540054


@dataclass(frozen=True)
class RunSettings:
    """Sizes and seeds of the full reproduction (``make run``)."""

    main_config: SimConfig = SimConfig()
    mc_config: SimConfig = SimConfig(n_stores=4, n_weeks=104)
    mc_reps: int = 150
    design_seed: int = 7
    main_seed: int = 2026
    mc_seed: int = 10_000
    dml: DMLSettings = DMLSettings()
    band: PriceBand = PriceBand()
    n_threads: int = 4  # BLAS/OpenMP threads for the main analysis
    n_jobs: int = 4  # worker processes for the Monte Carlo (one thread each)


def estimate_all(df, n_cat: int, dml: DMLSettings) -> dict[str, Estimate]:
    """Run every estimator on one panel."""
    return {key: fn(df, n_cat, dml) for key, (_, fn) in ESTIMATORS.items()}


def _r(x: float, nd: int = 4) -> float:
    return float(round(float(x), nd))


def _truth(design: MarketDesign) -> dict:
    return {
        "average_elasticity": _r(design.average_elasticity),
        "by_category": {
            n: _r(e) for n, e in zip(design.category_names, design.category_elasticity, strict=True)
        },
    }


def main_analysis(s: RunSettings, design: MarketDesign) -> tuple[dict, dict]:
    """Estimate everything on one large panel per scenario; returns (metrics, raw estimates)."""
    names = design.category_names
    metrics: dict = {}
    raw: dict = {}
    for scenario, lam in SCENARIOS.items():
        cfg = s.main_config.with_(hidden_confounding=lam)
        panel = simulate_panel(design, cfg, seed=s.main_seed)
        ests = estimate_all(panel.data, len(names), s.dml)
        raw[scenario] = ests
        out = {}
        for key, e in ests.items():
            pricing = evaluate_policy(design, e.by_category, s.band)
            out[key] = {
                "label": ESTIMATORS[key][0],
                "average_elasticity": _r(e.pooled),
                "average_se": _r(e.pooled_se),
                "by_category": {
                    n: {"estimate": _r(b), "se": _r(v), "ci95": [_r(b - Z95 * v), _r(b + Z95 * v)]}
                    for n, b, v in zip(names, e.by_category, e.by_category_se, strict=True)
                },
                "profit_share_of_oracle": _r(pricing.share_of_oracle),
                "profit_left_on_table": _r(1 - pricing.share_of_oracle),
                "profit_left_by_category": {
                    n: _r(1 - v) for n, v in zip(names, pricing.share_by_category, strict=True)
                },
                "categories_without_interior_optimum": int(
                    sum(pricing.flagged[design.category == c].any() for c in range(len(names)))
                ),
                "price_change_by_category": {
                    n: _r(pricing.price_change[design.category == c].mean())
                    for c, n in enumerate(names)
                },
                "diagnostics": {k: _r(v, 6) for k, v in e.diagnostics.items()},
            }
        metrics[scenario] = {"n_rows": len(panel.data), "estimators": out}
    return metrics, raw


@dataclass(frozen=True)
class Replication:
    """Everything the Monte Carlo keeps from one simulated history."""

    estimates: np.ndarray  # (n_estimators, 1 + n_categories): average, then per category
    se: np.ndarray
    profit_share: np.ndarray  # (n_estimators,)
    profit_share_by_category: np.ndarray  # (n_estimators, n_categories)
    flagged_categories: np.ndarray  # (n_estimators,)
    first_stage_f: float


def replicate(design: MarketDesign, cfg: SimConfig, seed: int, s: RunSettings) -> Replication:
    """Simulate one history and score every estimator (single-threaded, for worker processes)."""
    n_cat = len(design.category_names)
    with threadpool_limits(limits=1):
        ests = estimate_all(simulate_panel(design, cfg, seed=seed).data, n_cat, s.dml)
    pricing = [evaluate_policy(design, e.by_category, s.band) for e in ests.values()]
    return Replication(
        estimates=np.array([np.r_[e.pooled, e.by_category] for e in ests.values()]),
        se=np.array([np.r_[e.pooled_se, e.by_category_se] for e in ests.values()]),
        profit_share=np.array([p.share_of_oracle for p in pricing]),
        profit_share_by_category=np.array([p.share_by_category for p in pricing]),
        flagged_categories=np.array(
            [sum(p.flagged[design.category == c].any() for c in range(n_cat)) for p in pricing]
        ),
        first_stage_f=float(ests["iv_2sls"].diagnostics["first_stage_f"]),
    )


def monte_carlo(s: RunSettings, main_design: MarketDesign) -> tuple[dict, dict]:
    """Re-simulate the history ``mc_reps`` times per scenario and score every estimator.

    Replications run in ``n_jobs`` worker processes; each is seeded on its own, so the result
    does not depend on scheduling. Replication r uses the same random draws in both scenarios
    (common random numbers), so the scenarios differ only in the pricing rule.
    """
    keys = list(ESTIMATORS)
    # Same products (same design seed), fewer stores and weeks to keep the study fast.
    design = make_design(s.mc_config, seed=s.design_seed)
    if not np.array_equal(design.regular_price, main_design.regular_price):
        raise ValueError("Monte Carlo and main panel must share the product assortment")
    metrics: dict = {}
    raw: dict = {}
    for scenario, lam in SCENARIOS.items():
        cfg = s.mc_config.with_(hidden_confounding=lam)
        reps: list[Replication] = Parallel(n_jobs=s.n_jobs)(
            delayed(replicate)(design, cfg, s.mc_seed + r, s) for r in range(s.mc_reps)
        )
        est = np.stack([r.estimates for r in reps])
        se = np.stack([r.se for r in reps])
        share = np.stack([r.profit_share for r in reps])
        share_cat = np.stack([r.profit_share_by_category for r in reps])
        flagged = np.stack([r.flagged_categories for r in reps])
        f_stat = np.array([r.first_stage_f for r in reps])
        raw[scenario] = {"estimates": est, "se": se, "profit_share": share}
        metrics[scenario] = _summarise(keys, est, se, share, share_cat, flagged, design)
        metrics[scenario]["iv_first_stage_f_median"] = _r(np.median(f_stat), 1)
        metrics[scenario]["iv_first_stage_f_min"] = _r(np.min(f_stat), 1)
    return metrics, raw


def _summarise(keys, est, se, share, share_cat, flagged, design: MarketDesign) -> dict:
    names = design.category_names
    truth_avg, truth_cat = design.average_elasticity, design.category_elasticity
    # Few-cluster check: t(G - 1) critical values, G = products (clusters) behind each estimate.
    products = np.bincount(design.category)
    t_cat = stats.t.ppf(0.975, products - 1)
    out = {}
    for j, key in enumerate(keys):
        avg = est[:, j, 0]
        cat_err = est[:, j, 1:] - truth_cat
        cat_cover = np.abs(cat_err) <= Z95 * se[:, j, 1:]
        cat_cover_t = np.abs(cat_err) <= t_cat * se[:, j, 1:]
        avg_cover = np.abs(avg - truth_avg) <= Z95 * se[:, j, 0]
        loss = 1 - share[:, j]
        out[key] = {
            "label": ESTIMATORS[key][0],
            "average_elasticity": {
                "mean": _r(avg.mean()),
                "bias": _r(avg.mean() - truth_avg),
                "sd": _r(avg.std(ddof=1)),
                "rmse": _r(np.sqrt(np.mean((avg - truth_avg) ** 2))),
                "ci95_coverage": _r(avg_cover.mean()),
            },
            "category": {
                "mean_abs_bias": _r(np.abs(cat_err.mean(axis=0)).mean()),
                "rmse": _r(np.sqrt(np.mean(cat_err**2))),
                "ci95_coverage": _r(cat_cover.mean()),
                "ci95_coverage_t_critical": _r(cat_cover_t.mean()),
                "by_category": {
                    n: {
                        "mean": _r(est[:, j, 1 + c].mean()),
                        "bias": _r(cat_err[:, c].mean()),
                        "rmse": _r(np.sqrt(np.mean(cat_err[:, c] ** 2))),
                        "ci95_coverage": _r(cat_cover[:, c].mean()),
                        "mean_se": _r(se[:, j, 1 + c].mean()),
                    }
                    for c, n in enumerate(names)
                },
            },
            "profit_left_on_table": {
                "mean": _r(loss.mean()),
                "median": _r(np.median(loss)),
                "p90": _r(np.quantile(loss, 0.9)),
                "mean_by_category": {
                    n: _r(1 - share_cat[:, j, c].mean()) for c, n in enumerate(names)
                },
            },
            "flagged_categories_per_estimate": _r(flagged[:, j].mean(), 3),
        }
    return out


def run(out: Path, settings: RunSettings | None = None, make_figures: bool = True) -> dict:
    """Full reproduction: main analysis + Monte Carlo -> ``out/metrics.json`` (+ figures)."""
    s = settings or RunSettings()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    design = make_design(s.main_config, seed=s.design_seed)
    sq = status_quo(design, s.band)
    with threadpool_limits(limits=s.n_threads):
        main_metrics, main_raw = main_analysis(s, design)
        t_main = time.time() - t0
        mc_metrics, mc_raw = monte_carlo(s, design)
    metrics = {
        "settings": {
            "main_panel": {
                "stores": s.main_config.n_stores,
                "products": design.n_products,
                "weeks": s.main_config.n_weeks,
            },
            "monte_carlo_panel": {
                "stores": s.mc_config.n_stores,
                "products": design.n_products,
                "weeks": s.mc_config.n_weeks,
                "replications_per_scenario": s.mc_reps,
            },
            "scenarios": {k: {"hidden_confounding": v} for k, v in SCENARIOS.items()},
            "seeds": {"design": s.design_seed, "main": s.main_seed, "monte_carlo": s.mc_seed},
            "simulation": s.main_config.to_dict(),
            "dml": asdict(s.dml),
            "price_band": asdict(s.band),
        },
        "truth": _truth(design),
        "status_quo_profit_share_of_oracle": _r(sq.share_of_oracle),
        "status_quo_profit_left_by_category": {
            n: _r(1 - v) for n, v in zip(design.category_names, sq.share_by_category, strict=True)
        },
        "oracle_price_change_by_category": _oracle_moves(design, s.band),
        "main": main_metrics,
        "monte_carlo": mc_metrics,
        "runtime_seconds": {"main": _r(t_main, 1), "total": _r(time.time() - t0, 1)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if make_figures:
        from elasticity.figures import make_all

        make_all(metrics, design, main_raw, mc_raw, s.band, out / "figures")
    return metrics


def _oracle_moves(design: MarketDesign, band: PriceBand) -> dict:
    oracle = evaluate_policy(design, design.category_elasticity, band)
    return {
        n: _r(oracle.price_change[design.category == c].mean())
        for c, n in enumerate(design.category_names)
    }
