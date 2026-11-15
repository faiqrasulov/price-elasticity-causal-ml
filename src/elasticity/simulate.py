"""Retail panel simulator with known price elasticities and a confounded pricing policy.

The market (stores, products, true elasticities, regular prices, costs) is fixed by a design
seed; the weekly history (costs, weather, local events, promotions, demand shocks) is redrawn
for every replication. Demand is log-log:

    log q = log Q0_p + log traffic_s + eps_p * (log p - log P0_p)
            + season_c(w) + holiday + local event + heat + display + feature + u + e

and the pricing manager sets

    log p = log P0_p + passthrough * cost shock_pt + kappa * (anticipated demand + lambda * u)
            + log(1 - depth) * promo + noise.

``u`` is a demand signal the manager sees but the analyst never does. With ``lambda = 0`` every
confounder is observed (seasonality, holidays, weather, events, promotions, product and store
identity); with ``lambda > 0`` part of the price variation is driven by an unobserved demand
signal. The wholesale cost shock moves price but not demand, so it is a valid instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

HOLIDAY_WEEKS = (1, 13, 22, 47, 51, 52)  # week-of-year flags (New Year, Easter, ..., Christmas)
HEAT_THRESHOLD = 20.0  # degrees C above which heat lifts demand for some categories
N_REGIONS = 3


@dataclass(frozen=True)
class Category:
    """True parameters of one product category."""

    name: str
    elasticity: float  # mean own-price elasticity of the category's products
    price_level: float  # typical regular shelf price
    markup: float  # regular price / unit cost
    log_volume: float  # log units per store-week at the regular price
    season_amp: float  # amplitude of the yearly demand cycle (log points)
    season_peak: int  # week of year with the seasonal demand peak
    holiday_lift: float  # log-demand lift in holiday weeks
    event_lift: float  # log-demand lift in local-event weeks
    heat_lift: float  # log-demand lift per degree above HEAT_THRESHOLD


CATEGORIES: tuple[Category, ...] = (
    Category("Dairy", -0.6, 1.40, 1.30, np.log(40.0), 0.05, 1, 0.10, 0.05, 0.000),
    Category("Bakery", -1.2, 2.00, 1.45, np.log(35.0), 0.10, 51, 0.25, 0.15, 0.000),
    Category("Snacks", -1.8, 2.60, 2.40, np.log(40.0), 0.25, 51, 0.45, 0.45, 0.010),
    Category("Soft drinks", -2.2, 1.80, 1.65, np.log(45.0), 0.25, 28, 0.25, 0.40, 0.040),
    Category("Household", -2.5, 4.00, 1.75, np.log(25.0), 0.05, 1, 0.05, 0.05, 0.000),
)


@dataclass(frozen=True)
class SimConfig:
    """Size of the panel and parameters of the demand and pricing processes."""

    n_stores: int = 20
    products_per_category: int = 10
    n_weeks: int = 156
    hidden_confounding: float = 0.0  # lambda: weight of the unobserved signal in the price rule
    price_response: float = 0.30  # kappa: log-price change per log point of anticipated demand
    brand_volume: float = 0.25  # loading of brand strength on log volume
    brand_price: float = 0.7  # correlation of brand strength with log regular price
    volume_sd: float = 0.3  # product-level log-volume differences unrelated to brand
    price_dispersion: float = 0.15  # within-category sd of log regular price
    elasticity_sd: float = 0.0  # optional product-level spread of the elasticity within category
    cost_passthrough: float = 0.8
    cost_sd: float = 0.06  # stationary sd of the log wholesale-cost shock
    cost_ar: float = 0.7
    price_noise_sd: float = 0.08  # store-level pricing discretion, unrelated to demand
    demand_noise_sd: float = 0.20
    shock_sd: float = 0.12  # stationary sd of the unobserved demand signal u
    shock_ar: float = 0.6
    promo_rate: float = 0.10
    promo_depth: float = 0.20
    display_rate: float = 0.6  # P(display | promo), store level
    feature_rate: float = 0.5  # P(feature ad | promo), chain level
    display_lift: float = 0.25
    feature_lift: float = 0.15
    event_rate: float = 0.03
    store_traffic_sd: float = 0.30
    temp_noise_sd: float = 2.5

    def with_(self, **kwargs: float | int) -> SimConfig:
        """Return a copy with some fields replaced."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MarketDesign:
    """Fixed market: products, stores and their true parameters."""

    category: np.ndarray  # (P,) category index of each product
    elasticity: np.ndarray  # (P,) true own-price elasticity
    regular_price: np.ndarray  # (P,) P0
    unit_cost: np.ndarray  # (P,) average wholesale cost
    log_volume: np.ndarray  # (P,) log Q0, units per store-week at P0 (before store traffic)
    log_traffic: np.ndarray  # (S,) store traffic index (observed)
    region: np.ndarray  # (S,) weather region of each store
    category_names: tuple[str, ...]

    @property
    def n_products(self) -> int:
        return len(self.category)

    @property
    def n_stores(self) -> int:
        return len(self.log_traffic)

    @property
    def category_elasticity(self) -> np.ndarray:
        """True category elasticity = mean of its products' elasticities."""
        k = len(self.category_names)
        return np.array([self.elasticity[self.category == c].mean() for c in range(k)])

    @property
    def average_elasticity(self) -> float:
        """Target of pooled estimators: plain mean of product elasticities (balanced panel)."""
        return float(self.elasticity.mean())


@dataclass
class Panel:
    """One simulated history: observed data plus the hidden demand signal."""

    data: pd.DataFrame
    design: MarketDesign
    config: SimConfig
    hidden_signal: np.ndarray  # u, aligned with the rows of ``data``


def make_design(config: SimConfig, seed: int = 0) -> MarketDesign:
    """Draw the fixed market (products, stores, true elasticities, prices, costs)."""
    rng = np.random.default_rng(seed)
    n_cat = len(CATEGORIES)
    category = np.repeat(np.arange(n_cat), config.products_per_category)
    n_p = len(category)
    cat_eps = np.array([c.elasticity for c in CATEGORIES])[category]
    noise = rng.standard_normal(n_p)
    # Centre the product deviations within each category so the category truth is exact.
    for c in range(n_cat):
        idx = category == c
        noise[idx] -= noise[idx].mean()
    elasticity = cat_eps + config.elasticity_sd * noise

    brand = rng.standard_normal(n_p)  # unobserved brand strength
    price_noise = rng.standard_normal(n_p)
    r = config.brand_price
    log_p0 = np.log([CATEGORIES[c].price_level for c in category]) + config.price_dispersion * (
        r * brand + np.sqrt(1 - r**2) * price_noise
    )
    markup = np.array([CATEGORIES[c].markup for c in category])
    unit_cost = np.exp(log_p0) / markup
    log_volume = (
        np.array([CATEGORIES[c].log_volume for c in category])
        + config.brand_volume * brand
        + config.volume_sd * rng.standard_normal(n_p)
    )
    log_traffic = config.store_traffic_sd * rng.standard_normal(config.n_stores)
    region = np.arange(config.n_stores) % N_REGIONS
    return MarketDesign(
        category=category,
        elasticity=elasticity,
        regular_price=np.exp(log_p0),
        unit_cost=unit_cost,
        log_volume=log_volume,
        log_traffic=log_traffic,
        region=region,
        category_names=tuple(c.name for c in CATEGORIES),
    )


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], rho: float, sd: float) -> np.ndarray:
    """Stationary AR(1) paths along the last axis."""
    eps = rng.standard_normal(shape)
    out = np.empty(shape)
    out[..., 0] = eps[..., 0]
    scale = np.sqrt(1 - rho**2)
    for t in range(1, shape[-1]):
        out[..., t] = rho * out[..., t - 1] + scale * eps[..., t]
    return sd * out


def simulate_panel(design: MarketDesign, config: SimConfig, seed: int) -> Panel:
    """Simulate one weekly store x product history for a fixed market design."""
    rng = np.random.default_rng(seed)
    n_s, n_p, n_t = design.n_stores, design.n_products, config.n_weeks
    cats = CATEGORIES
    cat = design.category

    week = np.arange(n_t)
    woy = week % 52 + 1
    holiday = np.isin(woy, HOLIDAY_WEEKS).astype(np.float64)

    # Category-level demand drivers, broadcast to (product, week).
    season_amp = np.array([c.season_amp for c in cats])[cat][:, None]
    season_peak = np.array([c.season_peak for c in cats])[cat][:, None]
    season = season_amp * np.cos(2 * np.pi * (woy[None, :] - season_peak) / 52)  # (P, T)
    hol_lift = np.array([c.holiday_lift for c in cats])[cat][:, None] * holiday[None, :]
    event_lift = np.array([c.event_lift for c in cats])[cat]  # (P,)
    heat_lift = np.array([c.heat_lift for c in cats])[cat]  # (P,)

    # Weather by region and week (observed), local events by store and week (observed).
    region_offset = np.array([-3.0, 0.0, 3.0])[:N_REGIONS]
    temp_region = (
        13.0
        + 9.0 * np.sin(2 * np.pi * (woy[None, :] - 16) / 52)
        + region_offset[:, None]
        + _ar1(rng, (N_REGIONS, n_t), 0.5, config.temp_noise_sd)
    )
    temp = temp_region[design.region]  # (S, T)
    heat = np.maximum(temp - HEAT_THRESHOLD, 0.0)
    event = (rng.random((n_s, n_t)) < config.event_rate).astype(np.float64)

    # Promotions: chain-wide price cut + feature ad, store-level display execution.
    promo = (rng.random((n_p, n_t)) < config.promo_rate).astype(np.float64)
    feature = promo * (rng.random((n_p, n_t)) < config.feature_rate)
    display = promo[None] * (rng.random((n_s, n_p, n_t)) < config.display_rate)

    # Wholesale cost shocks (instrument) and the unobserved demand signal.
    cost_shock = _ar1(rng, (n_p, n_t), config.cost_ar, config.cost_sd)
    u = _ar1(rng, (n_s, n_p, n_t), config.shock_ar, config.shock_sd)

    # Demand the manager can anticipate from observable drivers, shape (S, P, T).
    anticipated = (
        season[None]
        + hol_lift[None]
        + event_lift[None, :, None] * event[:, None, :]
        + heat_lift[None, :, None] * heat[:, None, :]
    )

    log_p0 = np.log(design.regular_price)
    log_price = (
        log_p0[None, :, None]
        + config.cost_passthrough * cost_shock[None]
        + config.price_response * (anticipated + config.hidden_confounding * u)
        + np.log1p(-config.promo_depth) * promo[None]
        + config.price_noise_sd * rng.standard_normal((n_s, n_p, n_t))
    )
    log_qty = (
        design.log_volume[None, :, None]
        + design.log_traffic[:, None, None]
        + design.elasticity[None, :, None] * (log_price - log_p0[None, :, None])
        + anticipated
        + config.display_lift * display
        + config.feature_lift * feature[None]
        + u
        + config.demand_noise_sd * rng.standard_normal((n_s, n_p, n_t))
    )

    shape = (n_s, n_p, n_t)
    s_idx, p_idx, t_idx = np.meshgrid(np.arange(n_s), np.arange(n_p), week, indexing="ij")
    log_cost = np.log(design.unit_cost)[:, None] + cost_shock

    def flat(a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, shape).reshape(-1)

    data = pd.DataFrame(
        {
            "store": s_idx.reshape(-1).astype(np.int64),
            "product": p_idx.reshape(-1).astype(np.int64),
            "category": cat[p_idx.reshape(-1)].astype(np.int64),
            "week": t_idx.reshape(-1).astype(np.int64),
            "week_of_year": woy[t_idx.reshape(-1)].astype(np.int64),
            "holiday": holiday[t_idx.reshape(-1)],
            "event": flat(event[:, None, :]),
            "temperature": flat(temp[:, None, :]),
            "promo": flat(promo[None]),
            "display": flat(display),
            "feature": flat(feature[None]),
            "log_traffic": flat(design.log_traffic[:, None, None]),
            "log_cost": flat(log_cost[None]),
            "log_price": log_price.reshape(-1),
            "log_qty": log_qty.reshape(-1),
        }
    )
    return Panel(data=data, design=design, config=config, hidden_signal=u.reshape(-1))
