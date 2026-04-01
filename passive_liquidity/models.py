from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal, Optional


class FillRiskLevel(IntEnum):
    """Discrete fill-risk state from blended activity + book proximity (not a probability)."""

    LOW = 0
    MODERATE = 1
    ELEVATED = 2
    HIGH = 3


@dataclass
class FillRiskContext:
    """
    Recent fill *activity* proxies (short/long) and a derived risk score.

    This is not an estimate of being filled; it drives defensive widening tiers.
    """

    activity_short: float
    activity_long: float
    activity_long_count_only: float
    book_proximity_risk: float
    fill_risk_score: float
    level: FillRiskLevel

    @property
    def fill_rate(self) -> float:
        """Backward-compatible name: long-window count-only activity [0, 1]."""
        return self.activity_long_count_only


@dataclass
class RewardMarketToken:
    """One quotable CLOB asset (outcome) with reward metadata."""

    condition_id: str
    token_id: str
    outcome: str
    question: str
    rewards_max_spread: float
    rewards_min_size: float
    market_id: str
    volume_24hr: float = 0.0
    spread: float = 0.0
    one_day_price_change: float = 0.0
    rate_per_day: float = 0.0


@dataclass
class OrderBookSnapshot:
    best_bid: Optional[float]
    best_ask: Optional[float]
    tick_size: float
    neg_risk: bool
    bids: list[Any] = field(default_factory=list)
    asks: list[Any] = field(default_factory=list)
    raw: Any = None

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return 0.0
        return max(0.0, self.best_ask - self.best_bid)


@dataclass
class RewardRange:
    """Eligible reward band around midpoint (price space, 0–1)."""

    mid: float
    delta: float

    @property
    def bid_floor(self) -> float:
        return self.mid - self.delta

    @property
    def bid_ceiling(self) -> float:
        return self.mid

    @property
    def ask_floor(self) -> float:
        return self.mid

    @property
    def ask_ceiling(self) -> float:
        return self.mid + self.delta


@dataclass
class ScoringStatus:
    any_scoring: bool
    all_scoring: bool
    fraction: float
    order_ids_checked: int
    raw: dict[str, bool] = field(default_factory=dict)


@dataclass
class QuotePlan:
    bid_price: Optional[float]
    ask_price: Optional[float]
    size: float
    post_only: bool = True
    skip_reason: Optional[str] = None


@dataclass
class MarketQuoteState:
    consecutive_not_scoring: int = 0
    last_bid: Optional[float] = None
    last_ask: Optional[float] = None


@dataclass
class AdjustmentDecision:
    """Per-user-order action from AdjustmentEngine."""

    action: Literal["keep", "cancel", "replace"]
    new_price: Optional[float] = None
    reason: str = ""
    # Band vs API scoring (inside band does not imply scoring).
    placement_class: str = ""
    # Populated by AdjustmentEngine for logs (coarse vs fine tick markets).
    band_ticks: Optional[int] = None
    market_mode: str = ""
    tick_distance: Optional[int] = None
