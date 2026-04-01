"""
Simplified placement policy: coarse tick (book-based level pick) or fine tick (band ratio).

All legacy risk / scoring / structural paths are disabled; this module is the sole price logic.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, Optional

from passive_liquidity.models import AdjustmentDecision
from passive_liquidity.orderbook_fetcher import _level_price

LOG = logging.getLogger(__name__)

TickRegime = Literal["coarse", "fine", "unsupported"]


def classify_tick_regime(tick: float) -> TickRegime:
    """API may return 0.01 / 0.001 or 1 / 0.1 depending on scaling."""
    t = float(tick)
    if math.isclose(t, 1.0, rel_tol=0.0, abs_tol=1e-6) or math.isclose(
        t, 0.01, rel_tol=0.0, abs_tol=1e-9
    ):
        return "coarse"
    if math.isclose(t, 0.1, rel_tol=0.0, abs_tol=1e-6) or math.isclose(
        t, 0.001, rel_tol=0.0, abs_tol=1e-12
    ):
        return "fine"
    return "unsupported"


def _round_tick(price: float, tick: float) -> float:
    t = max(float(tick), 1e-12)
    steps = round(price / t)
    p = steps * t
    return max(t, min(1.0 - t, p))


def _level_size(level: Any) -> float:
    if level is None:
        return 0.0
    s = getattr(level, "size", None)
    if s is None and isinstance(level, dict):
        s = (
            level.get("size")
            or level.get("amount")
            or level.get("quantity")
            or level.get("shares")
        )
    try:
        return max(0.0, float(s or 0))
    except (TypeError, ValueError):
        return 0.0


def _coarse_reward_scan_range(
    side_u: str, mid: float, delta: float, tick: float
) -> tuple[float, float, float, int]:
    """
    Scan the reward half-band aligned to whole ticks (CLOB δ may be fractional, e.g. 0.045):

    - band = floor(δ / tick) * tick  → e.g. δ=0.045, tick=0.01 → 4 ticks → 0.04 → BUY [0.245, mid]
    - BUY: [mid − band, mid], SELL: [mid, mid + band]

    If δ is missing vs tick, fall back to 5 ticks.
    Returns (lo, hi, band_used, n_ticks).
    """
    t = max(float(tick), 1e-12)
    d = max(float(delta), 0.0)
    if d >= t - 1e-15:
        n_ticks = max(1, int(math.floor(d / t + 1e-9)))
    else:
        n_ticks = 5
    band = n_ticks * t
    band = max(band, 1e-12)
    if side_u == "BUY":
        lo, hi = mid - band, mid
    else:
        lo, hi = mid, mid + band
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi, band, n_ticks


def _book_prices_in_range(
    side_u: str,
    levels: list[Any],
    lo: float,
    hi: float,
    tick: float,
) -> list[float]:
    """Distinct prices with positive size on the given side, snapped to tick, inside [lo, hi]."""
    out: set[float] = set()
    for lv in levels or []:
        raw = _level_price(lv)
        if raw is None:
            continue
        p = _round_tick(float(raw), tick)
        if p < lo - 1e-12 or p > hi + 1e-12:
            continue
        if _level_size(lv) <= 0:
            continue
        out.add(p)
    return sorted(out)


@dataclass(frozen=True)
class EligibleBandDepthStats:
    """Order-book depth inside the same reward half-band used for pricing (coarse vs fine)."""

    scan_lo: float
    scan_hi: float
    tick_regime: str
    price_sizes: tuple[tuple[float, float], ...]
    total_in_band: float
    closer_to_mid_than_order: float
    pct_closer_of_band: Optional[float]


def _eligible_band_lo_hi(
    side_u: str, mid: float, delta: float, tick: float, regime: TickRegime
) -> tuple[float, float]:
    if regime == "coarse":
        lo, hi, _, _ = _coarse_reward_scan_range(side_u, mid, delta, tick)
        return lo, hi
    band = max(float(delta), 1e-12)
    if side_u == "BUY":
        lo, hi = mid - band, mid
    else:
        lo, hi = mid, mid + band
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def aggregate_depth_in_band(
    *,
    side: str,
    mid: float,
    delta: float,
    tick: float,
    bids: list[Any],
    asks: list[Any],
) -> tuple[float, float, str, list[tuple[float, float]], float]:
    """
    Returns (lo, hi, regime, sorted (price, size) list, total_size).
    BUY sums bids in band; SELL sums asks in band. Sizes merged per tick-rounded price.
    """
    side_u = side.upper()
    regime = classify_tick_regime(tick)
    lo, hi = _eligible_band_lo_hi(side_u, mid, delta, tick, regime)
    levels = bids if side_u == "BUY" else asks
    t = max(float(tick), 1e-12)
    agg: dict[float, float] = defaultdict(float)
    for lv in levels or []:
        raw = _level_price(lv)
        if raw is None:
            continue
        p = _round_tick(float(raw), t)
        if p < lo - 1e-12 or p > hi + 1e-12:
            continue
        sz = _level_size(lv)
        if sz <= 0:
            continue
        agg[p] += sz
    ordered = sorted(agg.items(), key=lambda x: x[0])
    total = sum(s for _, s in ordered)
    return lo, hi, regime, ordered, total


def compute_eligible_band_depth_stats(
    *,
    side: str,
    order_price: float,
    mid: float,
    delta: float,
    tick: float,
    bids: list[Any],
    asks: list[Any],
) -> EligibleBandDepthStats:
    lo, hi, regime, price_sizes, total = aggregate_depth_in_band(
        side=side,
        mid=mid,
        delta=delta,
        tick=tick,
        bids=bids,
        asks=asks,
    )
    t = max(float(tick), 1e-12)
    op = _round_tick(float(order_price), t)
    closer = 0.0
    for p, s in price_sizes:
        if abs(mid - p) + 1e-12 < abs(mid - op):
            closer += s
    pct = (closer / total * 100.0) if total > 1e-12 else None
    return EligibleBandDepthStats(
        scan_lo=lo,
        scan_hi=hi,
        tick_regime=regime,
        price_sizes=tuple(price_sizes),
        total_in_band=total,
        closer_to_mid_than_order=closer,
        pct_closer_of_band=pct,
    )


def format_eligible_band_depth_summary_zh(
    stats: EligibleBandDepthStats,
    *,
    max_levels: int = 10,
) -> str:
    """Extra lines for periodic Telegram band summary (Chinese)."""
    lines = [
        f"  深度统计区间[{stats.scan_lo:.4f},{stats.scan_hi:.4f}] regime={stats.tick_regime}",
    ]
    if not stats.price_sizes:
        lines.append("  带内各价深度: 无（该侧无正深度）")
    else:
        shown = stats.price_sizes[:max_levels]
        seg = " ".join(f"{p:.4f}:{s:g}" for p, s in shown)
        if len(stats.price_sizes) > max_levels:
            seg += f" …共{len(stats.price_sizes)}档"
        lines.append(
            f"  带内各价深度: {seg} | 带内合计 {stats.total_in_band:g}"
        )
    if stats.total_in_band <= 1e-12:
        lines.append("  较本单更靠 mid 侧深度: —（带内合计为 0）")
    elif stats.pct_closer_of_band is not None:
        lines.append(
            f"  较本单更靠 mid 侧深度: {stats.closer_to_mid_than_order:g} "
            f"（占带内合计 {stats.pct_closer_of_band:.1f}%）"
        )
    else:
        lines.append(
            f"  较本单更靠 mid 侧深度: {stats.closer_to_mid_than_order:g}"
        )
    return "\n".join(lines)


def _pick_coarse_target(
    side_u: str, mid: float, candidates: list[float]
) -> tuple[Optional[float], str]:
    """
    candidates sorted ascending by price.
    Rank by distance from mid: BUY below mid -> lower price = farther.
    """
    n = len(candidates)
    if n <= 0:
        return None, "coarse_tick_abandon_due_to_too_few_levels"
    if n <= 2:
        return None, "coarse_tick_abandon_due_to_too_few_levels"

    by_dist_asc = sorted(candidates, key=lambda p: abs(p - mid))
    if n == 3:
        chosen = by_dist_asc[1]
        return chosen, "coarse_tick_choose_middle_of_3"

    by_dist_desc = sorted(candidates, key=lambda p: abs(p - mid), reverse=True)
    # second farthest: index 1 when n >= 2
    idx = 1 if n >= 2 else 0
    chosen = by_dist_desc[idx]
    if n == 4:
        return chosen, "coarse_tick_choose_third_from_mid_of_4"
    return chosen, "coarse_tick_choose_second_farthest_default"


def _min_replace_delta(tick: float, min_replace_ticks: int) -> float:
    return max(1, int(min_replace_ticks)) * float(tick)


def decide_simple_price(
    *,
    side: str,
    price: float,
    mid: float,
    tick: float,
    delta: float,
    bids: list[Any],
    asks: list[Any],
    min_replace_ticks: int = 1,
) -> tuple[AdjustmentDecision, dict[str, Any]]:
    """
    Single pricing rule: coarse (tick ~0.01) or fine (~0.001); unsupported -> keep.
    """
    side_u = side.upper()
    regime = classify_tick_regime(tick)
    meta: dict[str, Any] = {
        "tick_size": tick,
        "tick_regime": regime,
        "mid": mid,
        "side": side_u,
    }

    if regime == "unsupported":
        meta["candidate_prices"] = []
        meta["candidate_count"] = 0
        meta["chosen_target_price"] = None
        meta["reason_code"] = "unsupported_tick_keep"
        LOG.info(
            "simple_price tick=%s regime=unsupported -> keep (no coarse/fine rule)",
            tick,
        )
        return (
            AdjustmentDecision("keep", reason="unsupported_tick_keep"),
            meta,
        )

    if regime == "coarse":
        lo, hi, band_used, band_ticks = _coarse_reward_scan_range(
            side_u, mid, delta, tick
        )
        levels = bids if side_u == "BUY" else asks
        cand = _book_prices_in_range(side_u, levels, lo, hi, tick)
        meta["candidate_prices"] = list(cand)
        meta["candidate_count"] = len(cand)
        meta["coarse_range_lo_hi"] = (lo, hi)
        meta["coarse_reward_band_delta"] = band_used
        meta["coarse_band_ticks"] = band_ticks

        target, rcode = _pick_coarse_target(side_u, mid, cand)
        meta["chosen_target_price"] = target
        meta["reason_code"] = rcode

        LOG.info(
            "simple_price coarse tick=%s mid=%.4f side=%s api_delta=%.4f "
            "band_ticks=%d band_used=%.4f scan=[%.4f,%.4f] book_levels=%d "
            "candidates=%s n=%d chosen=%s reason=%s",
            tick,
            mid,
            side_u,
            float(delta),
            band_ticks,
            band_used,
            lo,
            hi,
            len(levels or []),
            cand,
            len(cand),
            target,
            rcode,
        )

        if target is None:
            return (
                AdjustmentDecision("cancel", reason=rcode),
                meta,
            )

        tp = _round_tick(float(target), tick)
        min_d = _min_replace_delta(tick, min_replace_ticks)
        if abs(tp - price) < min_d - 1e-12:
            meta["reason_code"] = "coarse_tick_keep_already_at_target"
            return (
                AdjustmentDecision(
                    "keep",
                    reason="coarse_tick_keep_already_at_target",
                ),
                meta,
            )
        return (
            AdjustmentDecision("replace", new_price=tp, reason=rcode),
            meta,
        )

    # fine regime
    band = max(float(delta), 1e-12)
    dr = abs(float(price) - float(mid)) / band
    meta["distance_ratio"] = dr
    meta["candidate_prices"] = []
    meta["candidate_count"] = 0
    t = float(tick)

    ideal = (
        mid - 0.5 * band
        if side_u == "BUY"
        else mid + 0.5 * band
    )
    ideal = _round_tick(ideal, t)
    meta["chosen_target_price"] = ideal

    if 0.4 - 1e-12 <= dr <= 0.6 + 1e-12:
        meta["reason_code"] = "fine_tick_keep_in_target_band"
        LOG.info(
            "simple_price fine tick=%s mid=%.4f price=%.4f dr=%.4f -> keep in [0.4,0.6] band",
            tick,
            mid,
            price,
            dr,
        )
        return (
            AdjustmentDecision(
                "keep",
                reason="fine_tick_keep_in_target_band",
            ),
            meta,
        )

    if dr < 0.4 - 1e-12:
        rcode = "fine_tick_move_outward_to_half_band"
    else:
        rcode = "fine_tick_move_inward_to_half_band"

    meta["reason_code"] = rcode
    min_d = _min_replace_delta(t, min_replace_ticks)
    if abs(ideal - price) < min_d - 1e-12:
        meta["reason_code"] = f"{rcode}_noop_small_delta"
        LOG.info(
            "simple_price fine tick=%s dr=%.4f target=%.4f current=%.4f -> keep small delta",
            tick,
            dr,
            ideal,
            price,
        )
        return (
            AdjustmentDecision("keep", reason=meta["reason_code"]),
            meta,
        )

    LOG.info(
        "simple_price fine tick=%s mid=%.4f price=%.4f dr=%.4f -> replace to %.4f (%s)",
        tick,
        mid,
        price,
        dr,
        ideal,
        rcode,
    )
    return (
        AdjustmentDecision("replace", new_price=ideal, reason=rcode),
        meta,
    )
