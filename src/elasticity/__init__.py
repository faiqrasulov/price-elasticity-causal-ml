"""Causal price-elasticity estimation benchmarked on a retail simulator with known ground truth."""

from elasticity.estimators import ESTIMATORS, DMLSettings, Estimate
from elasticity.pricing import PriceBand, evaluate_policy, lerner_price, recommend_price
from elasticity.simulate import SimConfig, make_design, simulate_panel

__all__ = [
    "ESTIMATORS",
    "DMLSettings",
    "Estimate",
    "PriceBand",
    "SimConfig",
    "evaluate_policy",
    "lerner_price",
    "make_design",
    "recommend_price",
    "simulate_panel",
]
__version__ = "0.1.0"
