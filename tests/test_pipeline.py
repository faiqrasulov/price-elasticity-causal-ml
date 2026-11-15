"""End-to-end smoke test of the reproduction pipeline on a tiny panel."""

import json

import numpy as np

from elasticity.cli import main
from elasticity.estimators import ESTIMATORS, DMLSettings
from elasticity.pipeline import RunSettings, run
from elasticity.simulate import SimConfig

TINY = RunSettings(
    main_config=SimConfig(n_stores=3, products_per_category=3, n_weeks=40),
    mc_config=SimConfig(n_stores=2, products_per_category=3, n_weeks=30),
    mc_reps=2,
    dml=DMLSettings(n_folds=2, hgb_max_iter=10),
    n_threads=1,
    n_jobs=1,
)


def test_pipeline_writes_metrics_and_figures(tmp_path):
    metrics = run(tmp_path, TINY)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["truth"] == metrics["truth"]
    for scenario in ("observed", "hidden"):
        assert set(saved["main"][scenario]["estimators"]) == set(ESTIMATORS)
        mc = saved["monte_carlo"][scenario]
        for key in ESTIMATORS:
            assert np.isfinite(mc[key]["average_elasticity"]["mean"])
            assert 0 <= mc[key]["category"]["ci95_coverage"] <= 1
            assert mc[key]["profit_left_on_table"]["mean"] >= -1e-9
    for name in ("estimator_bias", "category_elasticities", "profit_left_on_table", "causal_dag"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 10_000


def test_cli_simulate_writes_panel(tmp_path):
    out = tmp_path / "panel.csv.gz"
    main(["simulate", "--out", str(out), "--stores", "2", "--weeks", "8"])
    assert out.stat().st_size > 1000
