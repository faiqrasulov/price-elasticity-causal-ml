"""Command line: ``elasticity run`` reproduces every reported number; ``elasticity simulate``
writes one simulated panel to disk for inspection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from elasticity.pipeline import SCENARIOS, RunSettings, run
from elasticity.simulate import make_design, simulate_panel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="elasticity", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="main analysis + Monte Carlo -> metrics.json + figures")
    p_run.add_argument("--out", type=Path, default=Path("reports"))
    p_run.add_argument("--reps", type=int, default=None, help="Monte Carlo replications")
    p_run.add_argument("--threads", type=int, default=None, help="BLAS/OpenMP threads")
    p_run.add_argument("--jobs", type=int, default=None, help="Monte Carlo worker processes")

    p_sim = sub.add_parser("simulate", help="write the main simulated panel (CSV)")
    p_sim.add_argument("--out", type=Path, default=Path("data/simulated_panel.csv.gz"))
    p_sim.add_argument("--scenario", choices=list(SCENARIOS), default="hidden")
    p_sim.add_argument("--stores", type=int, default=None)
    p_sim.add_argument("--weeks", type=int, default=None)

    args = parser.parse_args(argv)
    settings = RunSettings()
    if args.command == "run":
        if args.reps is not None:
            settings = replace(settings, mc_reps=args.reps)
        if args.threads is not None:
            settings = replace(settings, n_threads=args.threads)
        if args.jobs is not None:
            settings = replace(settings, n_jobs=args.jobs)
        m = run(args.out, settings)
        print(f"wrote {args.out / 'metrics.json'} in {m['runtime_seconds']['total']:.0f} s")
        for scenario in SCENARIOS:
            truth = m["truth"]["average_elasticity"]
            print(f"\nMonte Carlo, scenario '{scenario}' (true average elasticity {truth}):")
            for e in m["monte_carlo"][scenario].values():
                if not isinstance(e, dict):
                    continue
                a, c = e["average_elasticity"], e["category"]
                print(
                    f"  {e['label']:16s} mean {a['mean']:+.3f}  bias {a['bias']:+.3f}  "
                    f"rmse {a['rmse']:.3f}  category coverage {c['ci95_coverage']:.0%}  "
                    f"profit left {e['profit_left_on_table']['mean']:.1%}"
                )
    else:
        cfg = settings.main_config.with_(hidden_confounding=SCENARIOS[args.scenario])
        if args.stores is not None:
            cfg = cfg.with_(n_stores=args.stores)
        if args.weeks is not None:
            cfg = cfg.with_(n_weeks=args.weeks)
        design = make_design(cfg, seed=settings.design_seed)
        panel = simulate_panel(design, cfg, seed=settings.main_seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        panel.data.to_csv(args.out, index=False, float_format="%.6g")
        print(f"wrote {len(panel.data):,} rows to {args.out}")


if __name__ == "__main__":
    main()
