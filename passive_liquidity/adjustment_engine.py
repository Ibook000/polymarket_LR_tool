from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.fill_risk import widen_ticks_for_level
from passive_liquidity.models import AdjustmentDecision, FillRiskContext, FillRiskLevel, RewardRange
from passive_liquidity.structural_risk import queue_ticks_from_top

LOG = logging.getLogger(__name__)

# band_ticks <= this => discrete tick-distance rules; ratio-based recenter disabled.
COARSE_BAND_TICKS_MAX = 4


def _round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        tick = 0.01
    steps = round(price / tick)
    p = steps * tick
    return max(tick, min(1.0 - tick, p))


def _inside_reward_eligible_band(side_u: str, price: float, rr: RewardRange) -> bool:
    if side_u == "BUY":
        return rr.bid_floor - 1e-12 <= price <= rr.bid_ceiling + 1e-12
    if side_u == "SELL":
        return rr.ask_floor - 1e-12 <= price <= rr.ask_ceiling + 1e-12
    return False


def _cap_buy_not_at_best_bid(
    target: float,
    best_bid: Optional[float],
    second_bid: Optional[float],
    tick: float,
) -> float:
    """Keep buy limit strictly behind 买一; prefer at or below 买二."""
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
    """Keep sell limit strictly above 卖一; prefer at or above 卖二."""
    if best_ask is None:
        return target
    floor = second_ask if second_ask is not None else best_ask + tick
    floor = _round_tick(floor, tick)
    out = max(target, floor)
    if out <= best_ask + 1e-12:
        out = _round_tick(best_ask + tick, tick)
    return _round_tick(out, tick)


def _near_outer_band_edge(side_u: str, price: float, rr: RewardRange, tick: float, boundary_ticks: int) -> bool:
    """True if price sits within `boundary_ticks` of the band edge farthest from mid."""
    prox = max(1, boundary_ticks) * tick
    if side_u == "BUY":
        return price <= rr.bid_floor + prox + 1e-12
    if side_u == "SELL":
        return price >= rr.ask_ceiling - prox - 1e-12
    return False


class AdjustmentEngine:
    """
    Decide minimal changes to *existing* user orders: keep, cancel, or replace price.
    Does not invent new orders.
    """

    def __init__(self, config: PassiveConfig):
        self._c = config

    def decide(
        self,
        *,
        side: str,
        price: float,
        mid: float,
        tick: float,
        reward_range: RewardRange,
        scoring: bool,
        inventory: float,
        fill_risk: FillRiskContext,
        non_scoring_streak: int = 0,
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
        book_second_bid: Optional[float] = None,
        book_second_ask: Optional[float] = None,
        structural_observation_mode: bool = False,
        last_mid: Optional[float] = None,
    ) -> AdjustmentDecision:
        c = self._c
        side_u = side.upper()
        rr = reward_range
        delta = max(rr.delta, 1e-9)
        inside_band_now = _inside_reward_eligible_band(side_u, price, rr)
        t_eff = max(float(tick), 1e-12)
        band_ticks = max(1, int(math.floor(delta / t_eff)))
        coarse_market = band_ticks <= COARSE_BAND_TICKS_MAX
        tick_distance = int(round(abs(price - mid) / t_eff))
        mode_str = "coarse" if coarse_market else "fine"

        def _out(d: AdjustmentDecision) -> AdjustmentDecision:
            return replace(
                d,
                band_ticks=band_ticks,
                market_mode=mode_str,
                tick_distance=tick_distance,
            )

        def _placement_for_widen() -> str:
            if scoring and inside_band_now:
                return "inside_band_and_scoring"
            if inside_band_now:
                return "inside_band_but_not_effective"
            return "outside_band"

        min_move = max(1, c.adjustment_min_replace_ticks) * tick

        def replace_if_material(new_p: float, reason: str, pc: str = "") -> AdjustmentDecision:
            np = _round_tick(new_p, tick)
            if side_u == "BUY":
                np = _cap_buy_not_at_best_bid(np, best_bid, book_second_bid, tick)
                np = max(rr.bid_floor, min(rr.bid_ceiling, np))
            elif side_u == "SELL":
                np = _cap_sell_not_at_best_ask(np, best_ask, book_second_ask, tick)
                np = max(rr.ask_floor, min(rr.ask_ceiling, np))
            np = _round_tick(np, tick)
            if abs(np - price) < min_move - 1e-12:
                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason=f"no_op_small_delta:{reason}",
                        placement_class=pc,
                    )
                )
            return _out(
                AdjustmentDecision("replace", new_price=np, reason=reason, placement_class=pc)
            )

        # --- inventory / exposure ---
        if side_u == "BUY" and inventory >= c.max_position:
            return _out(
                AdjustmentDecision(
                    "cancel",
                    reason="inventory_at_max_long_no_more_bids",
                    placement_class="outside_band"
                    if not inside_band_now
                    else "inside_band_but_not_effective",
                )
            )
        if side_u == "SELL" and inventory <= -c.max_position:
            return _out(
                AdjustmentDecision(
                    "cancel",
                    reason="inventory_at_max_short_no_more_asks",
                    placement_class="outside_band"
                    if not inside_band_now
                    else "inside_band_but_not_effective",
                )
            )

        # --- far outside reward band (risky / useless for rewards) ---
        if side_u == "BUY":
            if price > mid + tick:
                return _out(
                    AdjustmentDecision("cancel", reason="buy_above_mid", placement_class="outside_band")
                )
            if price < rr.bid_floor - 2 * tick:
                return _out(
                    AdjustmentDecision(
                        "cancel", reason="buy_far_below_reward_band", placement_class="outside_band"
                    )
                )
        else:
            if price < mid - tick:
                return _out(
                    AdjustmentDecision("cancel", reason="sell_below_mid", placement_class="outside_band")
                )
            if price > rr.ask_ceiling + 2 * tick:
                return _out(
                    AdjustmentDecision(
                        "cancel", reason="sell_far_above_reward_band", placement_class="outside_band"
                    )
                )

        # --- fill activity + book proximity -> multi-level widen (away from mid, in band) ---
        widen_ticks = widen_ticks_for_level(fill_risk.level, c)
        if coarse_market:
            widen_ticks = min(widen_ticks, 1)
        if widen_ticks > 0:
            widen = max(1, widen_ticks) * tick
            pcw = _placement_for_widen()
            lvl = fill_risk.level.name.lower()
            if side_u == "BUY":
                target = price - widen
                target = max(rr.bid_floor, min(rr.bid_ceiling, target))
                return replace_if_material(
                    target,
                    f"widen_buy_fill_risk_{lvl}",
                    pcw,
                )
            target = price + widen
            target = max(rr.ask_floor, min(rr.ask_ceiling, target))
            return replace_if_material(
                target,
                f"widen_sell_fill_risk_{lvl}",
                pcw,
            )

        # --- in-band coarse tick market: discrete distance (no ratio recenter) ---
        _rec_pc = "recenter_for_scoring"

        def _recenter_material(new_p: float, reason: str) -> AdjustmentDecision:
            np = _round_tick(new_p, tick)
            if side_u == "BUY":
                np = _cap_buy_not_at_best_bid(np, best_bid, book_second_bid, tick)
                np = max(rr.bid_floor, min(rr.bid_ceiling, np))
            elif side_u == "SELL":
                np = _cap_sell_not_at_best_ask(np, best_ask, book_second_ask, tick)
                np = max(rr.ask_floor, min(rr.ask_ceiling, np))
            np = _round_tick(np, tick)
            if abs(np - price) < min_move - 1e-12:
                return _out(
                    AdjustmentDecision(
                        "keep", reason="no_op_small_delta", placement_class=_rec_pc
                    )
                )
            return _out(
                AdjustmentDecision(
                    "replace", new_price=np, reason=reason, placement_class=_rec_pc
                )
            )

        if coarse_market and inside_band_now:
            toward = mid - price
            streak_need = max(1, int(c.recenter_nudge_streak))
            q_top = queue_ticks_from_top(side_u, price, best_bid, best_ask, tick)
            min_top = float(c.recenter_min_ticks_from_top)
            ok_top = q_top is not None and q_top >= min_top - 1e-12
            low_clear = (
                fill_risk.level == FillRiskLevel.LOW and not structural_observation_mode
            )

            if tick_distance == 0:
                LOG.info(
                    "[Decision] mode=coarse band_ticks=%d tick_distance=0 action=keep reason=coarse_tick_at_mid",
                    band_ticks,
                )
                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason="coarse_tick_at_mid",
                        placement_class=_rec_pc,
                    )
                )
            if tick_distance == 1:
                LOG.info(
                    "[Decision] mode=coarse band_ticks=%d tick_distance=1 action=keep reason=coarse_tick_stable_zone",
                    band_ticks,
                )
                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason="coarse_tick_stable_zone",
                        placement_class=_rec_pc,
                    )
                )
            if tick_distance >= 2:
                if (
                    c.recenter_enabled
                    and low_clear
                    and ok_top
                    and non_scoring_streak >= streak_need
                    and abs(toward) >= 1e-12
                ):
                    if toward > 0:
                        raw = min(price + tick, mid - tick)
                    else:
                        raw = max(price - tick, mid + tick)
                    dec = _recenter_material(raw, "coarse_tick_move_inward")
                    LOG.info(
                        "[Decision] mode=coarse band_ticks=%d tick_distance=%d action=%s reason=%s",
                        band_ticks,
                        tick_distance,
                        dec.action,
                        dec.reason,
                    )
                    return dec
                reason_wait = "coarse_tick_far_wait_streak"
                if not low_clear:
                    reason_wait = "coarse_tick_far_blocked_risk_or_observation"
                elif not ok_top:
                    reason_wait = "coarse_tick_far_blocked_top_of_book"
                elif not c.recenter_enabled:
                    reason_wait = "coarse_tick_far_recenter_disabled"
                LOG.info(
                    "[Decision] mode=coarse band_ticks=%d tick_distance=%d action=keep reason=%s",
                    band_ticks,
                    tick_distance,
                    reason_wait,
                )
                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason=reason_wait,
                        placement_class=_rec_pc,
                    )
                )

        # --- fine tick: distance-based re-center (ratio tiers; after widen; not structural) ---
        if (
            not coarse_market
            and c.recenter_enabled
            and inside_band_now
            and fill_risk.level == FillRiskLevel.LOW
            and not structural_observation_mode
        ):
            q_top = queue_ticks_from_top(side_u, price, best_bid, best_ask, tick)
            min_top = float(c.recenter_min_ticks_from_top)
            if q_top is not None and q_top >= min_top - 1e-12:
                mid_move_need = float(c.recenter_mid_move_frac) * delta
                mid_ok = (
                    c.recenter_mid_move_frac <= 0
                    or last_mid is None
                    or abs(mid - last_mid) >= mid_move_need - 1e-12
                )
                if mid_ok:
                    dr = abs(price - mid) / delta
                    toward = mid - price
                    streak_need = max(1, int(c.recenter_nudge_streak))
                    step_t = max(1, int(c.recenter_step_ticks)) * tick
                    max_move = max(1, int(c.recenter_max_step_ticks)) * tick
                    ratio = max(0.01, min(0.49, float(c.recenter_target_ratio)))

                    if dr < 0.2 - 1e-12:
                        return _out(
                            AdjustmentDecision(
                                "keep",
                                reason="too_close_to_mid_unsafe",
                                placement_class=_rec_pc,
                            )
                        )
                    if dr < 0.5 - 1e-12:
                        if non_scoring_streak >= streak_need and abs(toward) >= 1e-12:
                            if toward > 0:
                                raw = min(price + tick, mid - tick)
                            else:
                                raw = max(price - tick, mid + tick)
                            return _recenter_material(raw, "slow_nudge_for_scoring")
                        return _out(
                            AdjustmentDecision(
                                "keep",
                                reason="stable_zone_no_adjust",
                                placement_class=_rec_pc,
                            )
                        )
                    if dr < 0.7 - 1e-12:
                        if abs(toward) < 1e-12:
                            return _out(
                                AdjustmentDecision(
                                    "keep",
                                    reason="stable_zone_no_adjust",
                                    placement_class=_rec_pc,
                                )
                            )
                        if toward > 0:
                            ideal = min(price + step_t, mid - tick)
                        else:
                            ideal = max(price - step_t, mid + tick)
                        step_move = min(abs(ideal - price), max_move)
                        direction = 1 if toward > 0 else -1
                        raw = price + direction * step_move
                        return _recenter_material(raw, "moderate_recentering")
                    if abs(toward) < 1e-12:
                        return _out(
                            AdjustmentDecision(
                                "keep",
                                reason="stable_zone_no_adjust",
                                placement_class=_rec_pc,
                            )
                        )
                    mag = ratio * delta
                    if toward > 0:
                        anchor = mid - mag
                        ideal = max(price, min(anchor, mid - tick))
                    else:
                        anchor = mid + mag
                        ideal = min(price, max(anchor, mid + tick))
                    if (toward > 0 and ideal <= price + 1e-12) or (
                        toward < 0 and ideal >= price - 1e-12
                    ):
                        return _out(
                            AdjustmentDecision(
                                "keep",
                                reason="no_op_small_delta",
                                placement_class=_rec_pc,
                            )
                        )
                    step_move = min(abs(ideal - price), max_move)
                    direction = 1 if ideal > price else -1
                    raw = price + direction * step_move
                    return _recenter_material(raw, "aggressive_recentering")

        # --- not scoring (mid-chasing / explore only when fill risk is LOW and not in struct observation) ---
        if not scoring:
            mid_chase_allowed = (
                fill_risk.level == FillRiskLevel.LOW and not structural_observation_mode
            )
            nudge = max(1, c.adjustment_nudge_ticks) * tick
            need_streak = max(1, c.adjustment_non_scoring_streak_nudge)
            low_q_thr = max(1, c.inside_band_low_quality_streak)

            if not inside_band_now:
                if not mid_chase_allowed:
                    return _out(
                        AdjustmentDecision(
                            "keep",
                            reason="no_mid_chase_fill_risk_or_struct_observation",
                            placement_class="outside_band",
                        )
                    )
                step = tick if coarse_market else nudge
                if side_u == "BUY":
                    target = min(mid, price + step)
                    target = max(rr.bid_floor, min(rr.bid_ceiling, target))
                    rsn = (
                        "coarse_tick_nudge_into_band_1"
                        if coarse_market
                        else "nudge_buy_into_band_not_scoring"
                    )
                    return replace_if_material(target, rsn, "outside_band")
                target = max(mid, price - step)
                target = max(rr.ask_floor, min(rr.ask_ceiling, target))
                rsn = (
                    "coarse_tick_nudge_into_band_1"
                    if coarse_market
                    else "nudge_ask_into_band_not_scoring"
                )
                return replace_if_material(target, rsn, "outside_band")

            # Inside band but API says not scoring — band eligibility does not guarantee scoring.
            low_quality = non_scoring_streak >= low_q_thr

            if low_quality:
                if coarse_market:
                    return _out(
                        AdjustmentDecision(
                            "keep",
                            reason="coarse_tick_low_quality_no_explore",
                            placement_class="inside_band_but_not_effective",
                        )
                    )
                inv_ok = (
                    side_u == "BUY"
                    and inventory < c.max_position * c.low_quality_explore_inv_frac
                ) or (
                    side_u == "SELL"
                    and inventory > -c.max_position * c.low_quality_explore_inv_frac
                )
                fill_ok = (
                    fill_risk.fill_risk_score <= c.low_quality_explore_max_risk_score
                    and fill_risk.level <= FillRiskLevel.MODERATE
                )
                interval = c.low_quality_explore_interval_cycles
                phase = non_scoring_streak - low_q_thr
                explore_slot = (
                    interval > 0
                    and phase >= 0
                    and phase % interval == 0
                    and inv_ok
                    and fill_ok
                    and mid_chase_allowed
                )

                if explore_slot:
                    explore = tick
                    if side_u == "BUY":
                        target = min(mid, price + explore)
                        target = max(rr.bid_floor, min(rr.bid_ceiling, target))
                        return replace_if_material(
                            target,
                            "low_quality_1tick_explore_buy",
                            "inside_band_but_not_effective",
                        )
                    target = max(mid, price - explore)
                    target = max(rr.ask_floor, min(rr.ask_ceiling, target))
                    return replace_if_material(
                        target,
                        "low_quality_1tick_explore_sell",
                        "inside_band_but_not_effective",
                    )

                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason="inside_band_low_quality_passive_no_chase",
                        placement_class="inside_band_but_not_effective",
                    )
                )

            near_outer = _near_outer_band_edge(
                side_u, price, rr, tick, c.adjustment_non_scoring_boundary_ticks
            )
            if coarse_market:
                return _out(
                    AdjustmentDecision(
                        "keep",
                        reason="coarse_tick_no_outer_edge_nudge",
                        placement_class="inside_band_but_not_effective",
                    )
                )
            if near_outer and non_scoring_streak >= need_streak and mid_chase_allowed:
                if side_u == "BUY":
                    target = min(mid, price + nudge)
                    target = max(rr.bid_floor, min(rr.bid_ceiling, target))
                    return replace_if_material(
                        target,
                        "nudge_buy_from_outer_edge_streak_not_scoring",
                        "inside_band_but_not_effective",
                    )
                target = max(mid, price - nudge)
                target = max(rr.ask_floor, min(rr.ask_ceiling, target))
                return replace_if_material(
                    target,
                    "nudge_ask_from_outer_edge_streak_not_scoring",
                    "inside_band_but_not_effective",
                )

            return _out(
                AdjustmentDecision(
                    "keep",
                    reason="not_scoring_inside_band_wait",
                    placement_class="inside_band_but_not_effective",
                )
            )

        pc_ok = "inside_band_and_scoring" if inside_band_now else "scoring_outside_band"
        if inside_band_now and not coarse_market:
            dr_end = abs(price - mid) / delta
            if dr_end >= 0.2 - 1e-12:
                pc_blk = (
                    "inside_band_and_scoring"
                    if scoring
                    else "inside_band_but_not_effective"
                )
                if fill_risk.level != FillRiskLevel.LOW:
                    return _out(
                        AdjustmentDecision(
                            "keep",
                            reason="recenter_blocked_by_risk",
                            placement_class=pc_blk,
                        )
                    )
                q_end = queue_ticks_from_top(
                    side_u, price, best_bid, best_ask, tick
                )
                if q_end is not None and q_end < float(
                    c.recenter_min_ticks_from_top
                ) - 1e-12:
                    return _out(
                        AdjustmentDecision(
                            "keep",
                            reason="recenter_blocked_by_top_of_book",
                            placement_class=pc_blk,
                        )
                    )
        final = AdjustmentDecision(
            "keep", reason="scoring_ok_low_risk", placement_class=pc_ok
        )
        LOG.info(
            "[Decision] mode=%s band_ticks=%d tick_distance=%d action=%s reason=%s",
            mode_str,
            band_ticks,
            tick_distance,
            final.action,
            final.reason,
        )
        return _out(final)
