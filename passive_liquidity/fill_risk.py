"""
Blend recent trade activity (count + size, short/long lookback, directional hints)
with distance-to-top-of-book into a fill *risk* score and discrete levels.

Scores are heuristics — not fill probabilities.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.models import FillRiskContext, FillRiskLevel


def _trade_timestamp(t: dict) -> Optional[float]:
    mt = t.get("match_time")
    if mt is None:
        mt = t.get("timestamp") or t.get("last_update")
    if mt is None:
        return None
    try:
        ts = float(mt)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    except (TypeError, ValueError):
        return None


def _trade_notional_usdc(t: dict) -> float:
    for k in ("usdcSize", "usdc_size", "size_usdc"):
        v = t.get(k)
        if v is not None and str(v).strip() != "":
            try:
                return max(0.0, float(v))
            except (TypeError, ValueError):
                break
    try:
        return max(0.0, float(t.get("size") or 0) * float(t.get("price") or 0))
    except (TypeError, ValueError):
        return 0.0


def _trade_side(t: dict) -> Optional[str]:
    s = t.get("side")
    if s is None:
        return None
    return str(s).strip().upper() or None


def directional_weight(order_side: str, trade_side: Optional[str], c: PassiveConfig) -> float:
    """
    Weight trades by how plausibly they indicate pressure against our resting order.

    BUY resting bid: taker SELL hits bids -> align SELL.
    SELL resting ask: taker BUY lifts asks -> align BUY.
    """
    o = order_side.upper()
    if not trade_side:
        return c.fill_dir_unknown_weight
    ts = trade_side.upper()
    if o == "BUY" and ts == "SELL":
        return c.fill_dir_aligned_weight
    if o == "SELL" and ts == "BUY":
        return c.fill_dir_aligned_weight
    if ts == o:
        return c.fill_dir_same_side_weight
    return c.fill_dir_misaligned_weight


def book_proximity_risk(
    order_side: str,
    price: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    ticks_scale: float,
) -> float:
    """
    [0, 1] — higher when the quote is closer to the aggressive side of the book
    (fewer ticks behind best bid / best ask).
    """
    t = max(tick, 1e-9)
    scale = max(0.5, ticks_scale)
    side_u = order_side.upper()
    if side_u == "BUY":
        if best_bid is None:
            return 0.5
        behind = (best_bid - price) / t
        q = max(0.0, behind)
    elif side_u == "SELL":
        if best_ask is None:
            return 0.5
        behind = (price - best_ask) / t
        q = max(0.0, behind)
    else:
        return 0.5
    return 1.0 / (1.0 + q / scale)


def _window_activity(
    trades: list[dict],
    now: float,
    lookback_sec: float,
    count_denom: float,
    size_norm_usdc: float,
    order_side: str,
    c: PassiveConfig,
) -> tuple[float, float, int]:
    """
    Returns (blended_activity [0,1], weighted_usdc_sum, raw_trade_count_in_window).
    """
    cutoff = now - lookback_sec
    w_count = 0.0
    w_usdc = 0.0
    raw_n = 0
    cw = max(1e-6, c.fill_activity_count_weight + c.fill_activity_size_weight)
    wc = c.fill_activity_count_weight / cw
    ws = c.fill_activity_size_weight / cw
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _trade_timestamp(t)
        if ts is None or ts < cutoff:
            continue
        raw_n += 1
        w = directional_weight(order_side, _trade_side(t), c)
        w_count += w
        w_usdc += w * _trade_notional_usdc(t)

    cd = max(1e-6, count_denom)
    sd = max(1e-6, size_norm_usdc)
    count_norm = min(1.0, w_count / cd)
    size_norm = min(1.0, w_usdc / sd)
    blended = wc * count_norm + ws * size_norm
    return blended, w_usdc, raw_n


def count_trades_in_lookback(trades: list[dict], now: float, lookback_sec: float) -> int:
    """Raw trade count with timestamp in (now - lookback_sec, now]."""
    cutoff = now - lookback_sec
    n = 0
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _trade_timestamp(t)
        if ts is None or ts < cutoff:
            continue
        n += 1
    return n


def tape_buy_sell_notional(
    trades: list[dict], now: float, lookback_sec: float
) -> tuple[float, float]:
    """Aggregate USDC notional from taker BUY vs SELL trades in the lookback window."""
    cutoff = now - lookback_sec
    buy_n = 0.0
    sell_n = 0.0
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _trade_timestamp(t)
        if ts is None or ts < cutoff:
            continue
        side = _trade_side(t)
        nu = _trade_notional_usdc(t)
        if side == "BUY":
            buy_n += nu
        elif side == "SELL":
            sell_n += nu
    return buy_n, sell_n


def long_window_count_only_activity(
    trades: list[dict],
    now: float,
    lookback_sec: float,
    count_denom: float,
) -> float:
    """Legacy long-window metric: undirected trade count, normalized."""
    cutoff = now - lookback_sec
    n = 0
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _trade_timestamp(t)
        if ts is None or ts < cutoff:
            continue
        n += 1
    return min(1.0, n / max(1e-6, count_denom))


def classify_fill_risk_level(score: float, c: PassiveConfig) -> FillRiskLevel:
    if score < c.fill_risk_level_1_max:
        return FillRiskLevel.LOW
    if score < c.fill_risk_level_2_max:
        return FillRiskLevel.MODERATE
    if score < c.fill_risk_level_3_max:
        return FillRiskLevel.ELEVATED
    return FillRiskLevel.HIGH


def build_fill_risk_context(
    trades: list[dict],
    *,
    order_side: str,
    price: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    c: PassiveConfig,
    now: Optional[float] = None,
) -> FillRiskContext:
    now = now or time.time()
    short_act, _, _ = _window_activity(
        trades,
        now,
        float(c.fill_short_lookback_sec),
        c.fill_short_count_denom,
        c.fill_short_size_norm_usdc,
        order_side,
        c,
    )
    long_act, _, _ = _window_activity(
        trades,
        now,
        float(c.fill_lookback_sec),
        c.fill_rate_denominator,
        c.fill_long_size_norm_usdc,
        order_side,
        c,
    )
    long_count_only = long_window_count_only_activity(
        trades, now, float(c.fill_lookback_sec), c.fill_rate_denominator
    )
    book_r = book_proximity_risk(
        order_side, price, best_bid, best_ask, tick, c.fill_book_ticks_scale
    )
    blend = max(0.0, min(1.0, c.fill_risk_blend_short))
    base_act = blend * short_act + (1.0 - blend) * long_act
    spike_term = min(1.0, short_act * max(0.1, c.fill_short_spike_boost))
    mix = max(0.0, min(1.0, c.fill_short_spike_mix))
    activity_effective = (1.0 - mix) * base_act + mix * max(base_act, spike_term)
    activity_effective = min(1.0, activity_effective)
    floor = max(0.0, min(1.0, c.fill_activity_book_floor))
    mult = max(0.0, min(1.0, c.fill_activity_book_mult))
    combined = activity_effective * (floor + mult * book_r)
    score = max(0.0, min(1.0, combined))
    level = classify_fill_risk_level(score, c)
    return FillRiskContext(
        activity_short=short_act,
        activity_long=long_act,
        activity_long_count_only=long_count_only,
        book_proximity_risk=book_r,
        fill_risk_score=score,
        level=level,
    )


def widen_ticks_for_level(level: FillRiskLevel, c: PassiveConfig) -> int:
    if level == FillRiskLevel.LOW:
        return 0
    if level == FillRiskLevel.MODERATE:
        return max(1, c.fill_risk_moderate_widen_ticks)
    if level == FillRiskLevel.ELEVATED:
        return max(1, c.adjustment_widen_ticks)
    return max(1, int(math.ceil(c.adjustment_widen_ticks * c.widen_fill_rate_factor)))
