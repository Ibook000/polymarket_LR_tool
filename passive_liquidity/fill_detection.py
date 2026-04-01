"""
Best-effort fill detection for monitored (whitelist, non-manual) open orders.

Compares snapshots across main-loop iterations; corroborates vanished orders with recent trades
that reference the order id (maker/taker fields).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.order_manager import _market, _oid, _price, _remaining_size, _side, _token_id

LOG = logging.getLogger(__name__)


def _order_status(o: dict) -> str:
    s = o.get("status")
    if s is None:
        return ""
    return str(s).strip().upper()


def _matched_size(o: dict) -> float:
    try:
        return max(0.0, float(o.get("size_matched") or 0))
    except (TypeError, ValueError):
        return 0.0


def order_original_size(o: dict) -> float:
    if o.get("original_size") is not None and str(o.get("original_size")).strip() != "":
        try:
            return max(0.0, float(o["original_size"]))
        except (TypeError, ValueError):
            pass
    return _remaining_size(o) + _matched_size(o)


def cumulative_filled_size(o: dict) -> float:
    m = _matched_size(o)
    if m > 1e-12:
        return m
    return max(0.0, order_original_size(o) - _remaining_size(o))


@dataclass
class MonitoredOrderSnapshot:
    order_id: str
    token_id: str
    condition_id: str
    side: str
    price: float
    original_size: float
    remaining_size: float
    matched_size: float
    cumulative_filled: float
    status: str
    order: dict

    @staticmethod
    def from_order(o: dict) -> Optional["MonitoredOrderSnapshot"]:
        oid = _oid(o)
        token_id = _token_id(o)
        condition_id = _market(o)
        if not oid or not token_id:
            return None
        rem = _remaining_size(o)
        orig = order_original_size(o)
        matched = _matched_size(o)
        cum = cumulative_filled_size(o)
        return MonitoredOrderSnapshot(
            order_id=oid,
            token_id=token_id,
            condition_id=condition_id,
            side=_side(o),
            price=_price(o),
            original_size=orig,
            remaining_size=rem,
            matched_size=matched,
            cumulative_filled=cum,
            status=_order_status(o),
            order=dict(o),
        )


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


def _trade_size(t: dict) -> float:
    try:
        return max(0.0, float(t.get("size") or 0))
    except (TypeError, ValueError):
        return 0.0


def _trade_price(t: dict) -> Optional[float]:
    try:
        if t.get("price") is None:
            return None
        return float(t["price"])
    except (TypeError, ValueError):
        return None


def _trade_asset_id(t: dict) -> str:
    return str(t.get("asset_id") or t.get("token_id") or "")


def trade_references_order_id(t: dict, order_id: str) -> bool:
    oid = str(order_id)
    for k in ("taker_order_id", "maker_order_id", "order_id", "orderID", "id"):
        v = t.get(k)
        if v is not None and str(v) == oid:
            return True
    for mo in t.get("maker_orders") or []:
        if not isinstance(mo, dict):
            continue
        for k in ("order_id", "orderID", "id"):
            v = mo.get(k)
            if v is not None and str(v) == oid:
                return True
    return False


def infer_fill_from_trades_for_order(
    *,
    order_id: str,
    token_id: str,
    trades: list[dict],
    now: float,
    lookback_sec: float,
    max_size: float,
) -> tuple[float, Optional[float]]:
    """
    Sum trade size for rows in the lookback window that reference this order_id.
    """
    cutoff = now - lookback_sec
    contributors: list[tuple[float, float]] = []
    for t in trades:
        if not isinstance(t, dict):
            continue
        aid = _trade_asset_id(t)
        if aid and str(aid) != str(token_id):
            continue
        ts = _trade_ts(t)
        if ts is None or ts < cutoff:
            continue
        if not trade_references_order_id(t, order_id):
            continue
        sz = _trade_size(t)
        if sz <= 0:
            continue
        px = _trade_price(t)
        contributors.append((sz, px if px is not None else 0.0))
    if not contributors:
        return 0.0, None
    total = 0.0
    num = 0.0
    den = 0.0
    for sz, px in contributors:
        take = min(sz, max(0.0, max_size - total))
        if take <= 0:
            break
        total += take
        num += take * px
        den += take
        if total >= max_size - 1e-9:
            break
    avg_px = (num / den) if den > 1e-12 else None
    return total, avg_px


class FillNotificationTracker:
    """
    Tracks previous open-order snapshots and cumulative filled amounts already notified.
    """

    def __init__(self) -> None:
        self._prev: dict[str, MonitoredOrderSnapshot] = {}
        self._notified_cumulative: dict[str, float] = {}

    def prev_token_ids(self) -> set[str]:
        return {s.token_id for s in self._prev.values() if s.token_id}

    def clear(self) -> None:
        self._prev.clear()
        self._notified_cumulative.clear()

    def process_loop(
        self,
        *,
        eligible_orders: list[dict],
        scoring_map: dict[str, bool],
        trades_by_token: dict[str, list],
        manual_token_ids: set[str],
        config: PassiveConfig,
        now: float,
        get_inventory: Callable[[str, str], float],
        send_fill_telegram: Callable[..., None],
    ) -> None:
        lookback = max(30.0, float(config.fill_infer_trade_lookback_sec))
        allow_manual = config.telegram_fill_manual_tokens

        for oid in list(self._prev.keys()):
            snap = self._prev[oid]
            if snap.token_id in manual_token_ids and not allow_manual:
                del self._prev[oid]
                self._notified_cumulative.pop(oid, None)

        current_snaps: dict[str, MonitoredOrderSnapshot] = {}
        for o in eligible_orders:
            if not isinstance(o, dict):
                continue
            tid = _token_id(o)
            if tid in manual_token_ids and not allow_manual:
                continue
            s = MonitoredOrderSnapshot.from_order(o)
            if s:
                current_snaps[s.order_id] = s

        current_oids = set(current_snaps.keys())

        for oid in list(self._prev.keys()):
            if oid in current_oids:
                continue
            prev = self._prev.pop(oid)
            prev_notified = self._notified_cumulative.pop(oid, 0.0)
            if prev.token_id in manual_token_ids and not allow_manual:
                continue

            trades = list(trades_by_token.get(prev.token_id) or [])
            inferred, inf_px = infer_fill_from_trades_for_order(
                order_id=oid,
                token_id=prev.token_id,
                trades=trades,
                now=now,
                lookback_sec=lookback,
                max_size=max(prev.remaining_size, 0.0) + 1e-6,
            )
            if inferred <= 1e-9:
                LOG.debug(
                    "Order vanished without trade rows referencing order id (cancel/replace?): %s",
                    oid[:20],
                )
                continue

            fill_amt = min(inferred, prev.remaining_size) if prev.remaining_size > 1e-9 else inferred
            observed_cum = min(
                prev.original_size,
                prev.cumulative_filled + fill_amt,
            )
            delta_notify = observed_cum - prev_notified
            if delta_notify <= 1e-9:
                continue

            is_full = prev.remaining_size <= fill_amt + 1e-9 or observed_cum >= prev.original_size - 1e-9
            self._maybe_send(
                config=config,
                snap=prev,
                filled_increment=delta_notify,
                fill_price=inf_px,
                remaining_after=0.0,
                is_full=is_full,
                scoring_map=scoring_map,
                get_inventory=get_inventory,
                send_fill_telegram=send_fill_telegram,
                dedupe_total_filled=observed_cum,
            )

        for oid, cur in current_snaps.items():
            prev = self._prev.get(oid)

            if prev is None:
                self._prev[oid] = cur
                self._notified_cumulative[oid] = cur.cumulative_filled
                continue

            prev_notified = self._notified_cumulative.get(oid, 0.0)
            rem_drop = max(0.0, prev.remaining_size - cur.remaining_size)
            cum_gain = max(0.0, cur.cumulative_filled - prev.cumulative_filled)
            inc = max(rem_drop, cum_gain)
            if inc <= 1e-9:
                self._prev[oid] = cur
                continue

            observed_cum = min(
                cur.original_size,
                max(cur.cumulative_filled, prev.cumulative_filled + inc),
            )
            delta_notify = observed_cum - prev_notified
            if delta_notify <= 1e-9:
                self._prev[oid] = cur
                continue

            is_full = cur.remaining_size <= 1e-9
            fill_px: Optional[float] = cur.price
            self._maybe_send(
                config=config,
                snap=cur,
                filled_increment=delta_notify,
                fill_price=fill_px,
                remaining_after=cur.remaining_size,
                is_full=is_full,
                scoring_map=scoring_map,
                get_inventory=get_inventory,
                send_fill_telegram=send_fill_telegram,
                dedupe_total_filled=observed_cum,
            )
            self._notified_cumulative[oid] = observed_cum
            self._prev[oid] = cur

    def _maybe_send(
        self,
        *,
        config: PassiveConfig,
        snap: MonitoredOrderSnapshot,
        filled_increment: float,
        fill_price: Optional[float],
        remaining_after: float,
        is_full: bool,
        scoring_map: dict[str, bool],
        get_inventory: Callable[[str, str], float],
        send_fill_telegram: Callable[..., None],
        dedupe_total_filled: float,
    ) -> None:
        if not config.telegram_notify_fill:
            return
        if is_full and not config.telegram_notify_full_fill:
            return
        if not is_full and not config.telegram_notify_partial_fill:
            return
        scoring = bool(scoring_map.get(snap.order_id, False))
        inv = 0.0
        try:
            inv = float(get_inventory(snap.condition_id, snap.token_id))
        except Exception as e:
            LOG.debug("inventory for fill notify skipped: %s", e)
        try:
            send_fill_telegram(
                order=snap.order,
                order_id=snap.order_id,
                token_id=snap.token_id,
                condition_id=snap.condition_id,
                side=snap.side,
                order_price=snap.price,
                filled_size=filled_increment,
                remaining_size=remaining_after,
                is_full=is_full,
                fill_price=fill_price,
                scoring=scoring,
                inventory=inv,
                dedupe_total_filled=dedupe_total_filled,
            )
        except Exception as e:
            LOG.warning("fill telegram callback failed: %s", e)
