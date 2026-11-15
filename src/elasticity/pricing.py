"""Profit-maximising prices under constant-elasticity demand, and the cost of a biased estimate.

With demand q = V (p / P0)^eps and unit cost c, profit (p - c) q is maximised at the Lerner
price p* = c * eps / (1 + eps) when eps < -1. When eps >= -1 demand is inelastic and profit
rises with price without limit: there is no interior optimum. Prices are therefore chosen
inside a guardrail band around the current price (as retailers do), and estimates with
eps >= -1 are flagged; their constrained optimum is the top of the band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elasticity.simulate import MarketDesign


@dataclass(frozen=True)
class PriceBand:
    """Allowed new price as a multiple of the current regular price."""

    lower: float = 0.80
    upper: float = 1.25


def lerner_price(cost: np.ndarray | float, elasticity: np.ndarray | float) -> np.ndarray:
    """Unconstrained profit-maximising price c * eps / (1 + eps); ``inf`` if eps >= -1."""
    cost = np.asarray(cost, dtype=np.float64)
    eps = np.asarray(elasticity, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        price = np.where(eps < -1.0, cost * eps / (1.0 + eps), np.inf)
    return price


def recommend_price(
    cost: np.ndarray,
    elasticity: np.ndarray,
    current_price: np.ndarray,
    band: PriceBand = PriceBand(),
) -> tuple[np.ndarray, np.ndarray]:
    """Constrained optimum within the band, and a flag for 'no interior optimum' (eps >= -1).

    Profit is unimodal in price for eps < -1, so clipping the Lerner price to the band gives the
    constrained optimum; for eps >= -1 profit increases with price, so it is the band's top.
    """
    current_price = np.asarray(current_price, dtype=np.float64)
    unconstrained = lerner_price(cost, elasticity)
    price = np.clip(unconstrained, band.lower * current_price, band.upper * current_price)
    return price, ~np.isfinite(unconstrained)


def expected_profit(
    price: np.ndarray,
    cost: np.ndarray,
    elasticity: np.ndarray,
    current_price: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Weekly profit (p - c) * V * (p / P0)^eps under the given (true) demand curve."""
    return (price - cost) * volume * (price / current_price) ** elasticity


def baseline_volume(design: MarketDesign) -> np.ndarray:
    """Chain-wide weekly units of each product at its current price (average conditions)."""
    return np.exp(design.log_volume) * np.exp(design.log_traffic).sum()


@dataclass(frozen=True)
class PricingOutcome:
    """True profit of a pricing policy relative to the oracle that knows every elasticity."""

    share_of_oracle: float  # total true profit / oracle profit
    share_by_category: np.ndarray
    price_change: np.ndarray  # recommended / current price, per product
    flagged: np.ndarray  # per product: estimate had no interior optimum


def score_prices(
    design: MarketDesign,
    price: np.ndarray,
    band: PriceBand = PriceBand(),
    flagged: np.ndarray | None = None,
) -> PricingOutcome:
    """True profit of per-product ``price`` relative to the oracle's constrained optimum."""
    cost, p0 = design.unit_cost, design.regular_price
    vol = baseline_volume(design)
    oracle_price, _ = recommend_price(cost, design.elasticity, p0, band)
    profit = expected_profit(price, cost, design.elasticity, p0, vol)
    oracle = expected_profit(oracle_price, cost, design.elasticity, p0, vol)
    by_cat = np.array(
        [
            profit[design.category == c].sum() / oracle[design.category == c].sum()
            for c in range(len(design.category_names))
        ]
    )
    return PricingOutcome(
        share_of_oracle=float(profit.sum() / oracle.sum()),
        share_by_category=by_cat,
        price_change=price / p0,
        flagged=np.zeros(len(p0), dtype=bool) if flagged is None else flagged,
    )


def evaluate_policy(
    design: MarketDesign,
    elasticity_by_category: np.ndarray,
    band: PriceBand = PriceBand(),
) -> PricingOutcome:
    """Price every product from its category's estimated elasticity; score it on the truth."""
    est = np.asarray(elasticity_by_category, dtype=np.float64)[design.category]
    price, flagged = recommend_price(design.unit_cost, est, design.regular_price, band)
    return score_prices(design, price, band, flagged)


def status_quo(design: MarketDesign, band: PriceBand = PriceBand()) -> PricingOutcome:
    """Keep current prices: the benchmark a model-based policy has to beat."""
    return score_prices(design, design.regular_price.copy(), band)
