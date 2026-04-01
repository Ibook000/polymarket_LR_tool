"""
Level-2 structural risk (token-level deleveraging).

Priority: evaluated in the main loop *before* Level-1 AdjustmentEngine for each order
when the token is in trigger set and the order is structurally risky — Level 1 never
runs for that order in the same cycle if Level 2 applies a replace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.fill_risk import book_proximity_risk
from passive_liquidity.models import FillRiskContext, FillRiskLevel, RewardRange


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        tick = 0.01
    steps = round(price / tick)
    p = steps * tick
    return max(tick, min(1.0 - tick, p))


def _cap_buy_not_at_best_bid(
    target: float,
    best_bid: Optional[float],
    second_bid: Optional[float],
    tick: float,
) -> float:
    if best_bid is None:
        return target
    cap = second_bid if second_bid is not None else best_bid - tick
    cap = _round_tick(cap, tick)
    out = min(target, cap)
    if out >= best_bid - 1e-12:
        out = _round_tick(best_bid - tick, tick)
    return _round_tick(out, tick)


def _cap_sell_not_at_best_ask(
    target: float,
    best_ask: Optional[float],
    second_ask: Optional[float],
    tick: float,
) -> float:
    if best_ask is None:
        return target
    floor = second_ask if second_ask is not None else best_ask + tick
    floor = _round_tick(floor, tick)
    out = max(target, floor)
    if out <= best_ask + 1e-12:
        out = _round_tick(best_ask + tick, tick)
    return _round_tick(out, tick)


def _trade_ts(t: dict) -> Optional[float]:
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


def microtrend_pressure_against_order(
    order_side: str,
    trades: list[dict],
    now: float,
    lookback_sec: float,
) -> Optional[float]:
    """
    [0,1] share of short-window notional that aggressively trades against our resting side.
    None if there is essentially no print in the window (caller may treat as pass).
    """
    cutoff = now - lookback_sec
    buy_u = 0.0
    sell_u = 0.0
    for t in trades:
        if not isinstance(t, dict):
            continue
        ts = _trade_ts(t)
        if ts is None or ts < cutoff:
            continue
        u = _trade_notional_usdc(t)
        s = _trade_side(t)
        if s == "BUY":
            buy_u += u
        elif s == "SELL":
            sell_u += u
    tot = buy_u + sell_u
    if tot < 1e-9:
        return None
    ou = order_side.upper()
    if ou == "BUY":
        return min(1.0, sell_u / tot)
    if ou == "SELL":
        return min(1.0, buy_u / tot)
    return None


def queue_ticks_from_top(
    side: str,
    price: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
) -> Optional[float]:
    t = max(tick, 1e-9)
    su = side.upper()
    if su == "BUY":
        if best_bid is None:
            return None
        return max(0.0, (best_bid - price) / t)
    if su == "SELL":
        if best_ask is None:
            return None
        return max(0.0, (price - best_ask) / t)
    return None


def is_structural_risky_order(
    *,
    side: str,
    price: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    fill_ctx: FillRiskContext,
    trades: list[dict],
    now: float,
    c: PassiveConfig,
) -> bool:
    """
    Risky only if: tight queue, high book proximity, elevated short activity,
    and directional microtrend pressure against this quote (unless no tape).
    """
    q = queue_ticks_from_top(side, price, best_bid, best_ask, tick)
    if q is None:
        return False
    if q > float(c.struct_max_queue_ticks) + 1e-9:
        return False
    prox = book_proximity_risk(
        side, price, best_bid, best_ask, tick, c.fill_book_ticks_scale
    )
    if prox < float(c.struct_book_proximity_min) - 1e-9:
        return False
    if fill_ctx.activity_short < float(c.struct_short_activity_min) - 1e-9:
        return False

    mp = microtrend_pressure_against_order(
        side, trades, now, float(c.struct_dir_lookback_sec)
    )
    if mp is None:
        return c.struct_directional_allow_no_tape
    return mp >= float(c.struct_directional_min) - 1e-9


def structural_exposure_cut_frac(level: FillRiskLevel, c: PassiveConfig) -> float:
    """Larger cuts at higher aggregate fill-risk on the token."""
    if level == FillRiskLevel.HIGH:
        v = c.struct_cut_frac_high
    elif level == FillRiskLevel.ELEVATED:
        v = c.struct_cut_frac_elevated
    elif level == FillRiskLevel.MODERATE:
        v = c.struct_cut_frac_moderate
    else:
        v = c.struct_cut_frac_low
    return max(0.0, min(0.95, float(v)))


def compute_structural_replace(
    *,
    side: str,
    price: float,
    tick: float,
    reward_range: RewardRange,
    best_bid: Optional[float],
    best_ask: Optional[float],
    second_bid: Optional[float],
    second_ask: Optional[float],
    remaining_size: float,
    exposure_cut_frac: float,
    c: PassiveConfig,
) -> Optional[tuple[float, float]]:
    if remaining_size <= 0:
        return None
    cut = max(0.0, min(0.95, float(exposure_cut_frac)))
    raw_target = remaining_size * (1.0 - cut)
    new_size = max(float(c.struct_min_post_size), raw_target)
    if new_size >= remaining_size - 1e-9:
        return None

    st = max(1, int(c.struct_safety_ticks))
    su = side.upper()
    if su == "BUY":
        target = price - st * tick
        target = max(reward_range.bid_floor, min(reward_range.bid_ceiling, target))
        target = _cap_buy_not_at_best_bid(target, best_bid, second_bid, tick)
        target = max(reward_range.bid_floor, min(reward_range.bid_ceiling, target))
    elif su == "SELL":
        target = price + st * tick
        target = max(reward_range.ask_floor, min(reward_range.ask_ceiling, target))
        target = _cap_sell_not_at_best_ask(target, best_ask, second_ask, tick)
        target = max(reward_range.ask_floor, min(reward_range.ask_ceiling, target))
    else:
        return None

    target = _round_tick(target, tick)
    min_move = max(1, c.adjustment_min_replace_ticks) * tick
    price_moves = abs(target - price) >= min_move - 1e-12
    size_moves = abs(new_size - remaining_size) >= 1e-6
    if not price_moves and not size_moves:
        return None
    return (target, new_size)


@dataclass
class StructuralTokenState:
    last_trigger_monotonic: float = 0.0
    observation_until_monotonic: float = 0.0


def can_trigger_structural_for_token(
    danger_shares: float,
    danger_notional: float,
    token_id: str,
    c: PassiveConfig,
    state: dict[str, StructuralTokenState],
) -> bool:
    if not c.struct_enabled:
        return False
    gate_shares = c.struct_min_danger_exposure > 0 and danger_shares >= float(
        c.struct_min_danger_exposure
    )
    gate_notional = c.struct_min_danger_notional_usdc > 0 and danger_notional >= float(
        c.struct_min_danger_notional_usdc
    )
    if not gate_shares and not gate_notional:
        return False
    now = time.monotonic()
    st = state.setdefault(token_id, StructuralTokenState())
    cd = max(0.0, float(c.struct_cooldown_sec))
    if cd > 0 and (now - st.last_trigger_monotonic) < cd:
        return False
    return True


def mark_structural_cooldown(token_id: str, state: dict[str, StructuralTokenState]) -> None:
    state.setdefault(token_id, StructuralTokenState()).last_trigger_monotonic = time.monotonic()
