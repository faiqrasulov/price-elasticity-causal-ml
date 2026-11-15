"""Report figures (static PNGs for the README)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from elasticity.estimators import ESTIMATORS  # noqa: E402
from elasticity.pricing import (  # noqa: E402
    PriceBand,
    baseline_volume,
    expected_profit,
    recommend_price,
)
from elasticity.simulate import MarketDesign  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
COLORS = {
    "naive_ols": "#b0aea5",
    "ols_controls": "#7d7b73",
    "twfe": "#eb6834",
    "iv_2sls": "#1baf7a",
    "dml_lasso": "#7fb2ec",
    "dml_boosting": "#2a78d6",
}
SCENARIO_TITLES = {
    "observed": "All confounders observed",
    "hidden": "Manager also reacts to a hidden demand signal",
}

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _hbox(ax: plt.Axes, vals: np.ndarray, y: float, color: str, h: float = 0.55) -> None:
    """Horizontal box plot (quartiles, median, 1.5 IQR whiskers) drawn from primitives."""
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    lo = vals[vals >= q1 - 1.5 * (q3 - q1)].min()
    hi = vals[vals <= q3 + 1.5 * (q3 - q1)].max()
    ax.barh(y, q3 - q1, left=q1, height=h, color=color, alpha=0.85, zorder=2)
    ax.vlines(med, y - h / 2, y + h / 2, color=INK, lw=1.4, zorder=4)
    ax.hlines([y, y], [lo, q3], [q1, hi], color=MUTED, lw=1, zorder=1)
    ax.vlines([lo, hi], y - h / 4, y + h / 4, color=MUTED, lw=1, zorder=1)


def fig_sampling_distributions(metrics: dict, mc_raw: dict, path: Path) -> None:
    """Monte Carlo distribution of each estimator's average elasticity vs the truth."""
    keys = list(ESTIMATORS)
    truth = metrics["truth"]["average_elasticity"]
    reps = metrics["settings"]["monte_carlo_panel"]["replications_per_scenario"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for ax, scenario in zip(axes, ("observed", "hidden"), strict=True):
        data = [mc_raw[scenario]["estimates"][:, j, 0] for j in range(len(keys))]
        pos = np.arange(len(keys))[::-1]
        rng = np.random.default_rng(0)
        for y, (vals, key) in zip(pos, zip(data, keys, strict=True), strict=True):
            _hbox(ax, vals, y, COLORS[key])
            ax.scatter(vals, y + rng.uniform(-0.18, 0.18, len(vals)), s=5, color=INK, alpha=0.25,
                       lw=0, zorder=3)
            bias = metrics["monte_carlo"][scenario][key]["average_elasticity"]["bias"]
            ax.text(1.02, y, f"bias {bias:+.2f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8.5, color=INK2)
        ax.axvline(truth, color=INK, ls="--", lw=1.2, zorder=1)
        ax.text(truth, -0.62, f"truth {truth:.2f} ", fontsize=9, color=INK, ha="right")
        ax.set_ylim(-0.8, len(keys) - 0.5)
        ax.set_yticks(pos, [ESTIMATORS[k][0] for k in keys])
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.set_xlabel("Estimated average price elasticity")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        f"Which estimators recover the true elasticity? {reps} simulated sales histories per "
        "scenario", x=0.01, ha="left", fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(w_pad=6)
    _save(fig, path)


def fig_category_estimates(metrics: dict, path: Path) -> None:
    """Per-category estimates with 95% CIs vs the true elasticity (one large panel)."""
    keys = ["naive_ols", "twfe", "iv_2sls", "dml_boosting"]
    truth = metrics["truth"]["by_category"]
    names = list(truth)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    offsets = np.linspace(-0.27, 0.27, len(keys))
    for ax, scenario in zip(axes, ("observed", "hidden"), strict=True):
        ests = metrics["main"][scenario]["estimators"]
        x = np.arange(len(names))
        for i, n in enumerate(names):
            ax.plot([i - 0.4, i + 0.4], [truth[n]] * 2, color=INK, lw=2.2, zorder=4,
                    label="True elasticity" if i == 0 else None)
        for off, key in zip(offsets, keys, strict=True):
            b = np.array([ests[key]["by_category"][n]["estimate"] for n in names])
            lo = np.array([ests[key]["by_category"][n]["ci95"][0] for n in names])
            hi = np.array([ests[key]["by_category"][n]["ci95"][1] for n in names])
            ax.errorbar(x + off, b, yerr=[b - lo, hi - b], fmt="o", ms=5.5, capsize=2.5,
                        color=COLORS[key], label=ESTIMATORS[key][0], zorder=5)
        ax.axhline(0, color=AXIS, lw=1)
        ax.axhline(-1, color=MUTED, lw=0.9, ls=":")
        ax.text(len(names) - 0.5, -1.03, "unit elastic", fontsize=8, color=MUTED, ha="right",
                va="top")
        ax.set_xticks(x, names)
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Price elasticity (95% CI)")
    axes[0].legend(loc="lower left", ncol=2)
    n_rows = metrics["main"]["observed"]["n_rows"]
    fig.suptitle(
        f"Elasticity by category on one simulated panel ({n_rows:,} store-product-weeks)",
        x=0.01, ha="left", fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, path)


def fig_profit(metrics: dict, design: MarketDesign, band: PriceBand, path: Path) -> None:
    """Left: one category's true profit curve with each estimator's price. Right: MC profit loss."""
    keys = list(ESTIMATORS)
    name, scenario = "Household", "hidden"
    cat = design.category_names.index(name)
    idx = np.flatnonzero(design.category == cat)
    cost, p0 = design.unit_cost[idx], design.regular_price[idx]
    eps, vol = design.elasticity[idx], baseline_volume(design)[idx]
    grid = np.linspace(0.75, 1.32, 400)

    def cat_profit(mult: np.ndarray) -> np.ndarray:
        return np.array([expected_profit(m * p0, cost, eps, p0, vol).sum() for m in mult])

    oracle_mult = recommend_price(cost, eps, p0, band)[0] / p0
    best = expected_profit(oracle_mult * p0, cost, eps, p0, vol).sum()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    ax.axvspan(band.lower, band.upper, color=GRID, alpha=0.45, lw=0)
    ax.plot(grid, cat_profit(grid) / best, color=INK, lw=2)
    ax.axvline(1.0, color=MUTED, ls=":", lw=1)
    ax.text(1.0, 0.805, " current price", fontsize=8, color=MUTED)
    ests = metrics["main"][scenario]["estimators"]
    moves = {key: ests[key]["price_change_by_category"][name] for key in keys}
    order = sorted(keys, key=lambda k: moves[k])
    for key, ytxt in zip(order, np.linspace(0.955, 0.83, len(keys)), strict=True):
        m = moves[key]
        prof = cat_profit(np.array([m]))[0] / best
        ax.plot(m, prof, "o", ms=8, color=COLORS[key], zorder=5, mec=SURFACE)
        est = ests[key]["by_category"][name]["estimate"]
        ax.annotate(f"{ESTIMATORS[key][0]} ({est:.2f}): {prof - 1:+.1%}", (m, prof),
                    xytext=(0.76, ytxt), fontsize=8.5, color=COLORS[key], va="center",
                    arrowprops={"arrowstyle": "-", "color": COLORS[key], "lw": 0.6})
    ax.plot(oracle_mult.mean(), 1.0, "*", ms=13, color=INK, zorder=6)
    ax.text(oracle_mult.mean(), 1.006, "oracle", ha="center", fontsize=8.5)
    eps_c = design.category_elasticity[cat]
    ax.set_title(f"{name} (true elasticity {eps_c:.1f}), hidden-signal panel")
    ax.set_xlabel("New price / current price (shaded: allowed band)")
    ax.set_ylabel("True profit / oracle profit")
    ax.set_ylim(0.8, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    mc = metrics["monte_carlo"]
    y = np.arange(len(keys))[::-1]
    h = 0.38
    legend = {"observed": "All confounders observed", "hidden": "Hidden demand signal"}
    for k, (sc, shade) in enumerate((("observed", 0.95), ("hidden", 0.45))):
        vals = [mc[sc][key]["profit_left_on_table"]["mean"] for key in keys]
        yy = y + (h / 2 if k == 0 else -h / 2)
        ax2.barh(yy, vals, height=h, color=[COLORS[key] for key in keys], alpha=shade)
        for yv, v in zip(yy, vals, strict=True):
            ax2.text(v + 0.0012, yv, f"{v:.1%}", va="center", fontsize=8, color=INK2)
        ax2.barh([np.nan], [np.nan], color=MUTED, alpha=shade, label=legend[sc])
    ax2.set_yticks(y, [ESTIMATORS[k][0] for k in keys])
    ax2.set_xlim(0, 0.065)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.set_xlabel("Total profit left on the table vs oracle (mean of Monte Carlo runs)")
    ax2.set_title("Cost of a biased elasticity, all 50 products")
    ax2.legend(loc="center right", fontsize=8.5, handlelength=1.2)
    ax2.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=3)
    _save(fig, path)


def _node(ax, xy, text, color, dashed=False):
    box = FancyBboxPatch((xy[0] - 0.13, xy[1] - 0.075), 0.26, 0.15,
                         boxstyle="round,pad=0.01,rounding_size=0.03", fc=SURFACE, ec=color,
                         lw=1.8, ls="--" if dashed else "-")
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=9.5, color=INK)


def _edge(ax, a, b, color=INK2, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=color, lw=1.4,
                                 ls="--" if dashed else "-", shrinkA=34, shrinkB=34))


def fig_dag(path: Path) -> None:
    """Causal diagram of the simulator."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-0.03, 1)
    ax.axis("off")
    pos = {
        "Z": (0.15, 0.5),
        "D": (0.55, 0.5),
        "Y": (1.35, 0.5),
        "X": (0.75, 0.88),
        "P": (1.15, 0.88),
        "B": (0.75, 0.15),
        "U": (1.15, 0.15),
    }
    _node(ax, pos["Z"], "Wholesale cost\n(instrument)", COLORS["iv_2sls"])
    _node(ax, pos["D"], "Price\n(log p)", INK)
    _node(ax, pos["Y"], "Units sold\n(log q)", INK)
    _node(ax, pos["X"], "Season, holidays,\nweather, events", COLORS["dml_boosting"])
    _node(ax, pos["P"], "Promotion,\ndisplay, feature ad", COLORS["dml_boosting"])
    _node(ax, pos["B"], "Brand strength\n(product effect)", COLORS["twfe"])
    _node(ax, pos["U"], "Hidden demand\nsignal (scenario 2)", MUTED, dashed=True)
    _edge(ax, pos["Z"], pos["D"])
    _edge(ax, pos["D"], pos["Y"], color=INK)
    ax.text(0.95, 0.525, "elasticity (target)", ha="center", fontsize=9, fontweight="bold")
    for src in ("X", "P", "B"):
        _edge(ax, pos[src], pos["D"])
        _edge(ax, pos[src], pos["Y"])
    _edge(ax, pos["U"], pos["D"], dashed=True)
    _edge(ax, pos["U"], pos["Y"], dashed=True)
    ax.text(0.0, -0.01, "Blue: observed confounders (controls, DML).  Orange: time-invariant, "
            "removed by fixed effects.  Green: moves price only (IV).  Dashed: never observed.",
            fontsize=8.5, color=INK2)
    ax.set_title("Why the price-demand correlation is not the elasticity", fontsize=12.5,
                 fontweight="bold")
    _save(fig, path)


def make_all(metrics: dict, design: MarketDesign, main_raw: dict, mc_raw: dict, band: PriceBand,
             out: Path) -> None:
    """Write all report figures to ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig_sampling_distributions(metrics, mc_raw, out / "estimator_bias.png")
        fig_category_estimates(metrics, out / "category_elasticities.png")
        fig_profit(metrics, design, band, out / "profit_left_on_table.png")
        fig_dag(out / "causal_dag.png")
