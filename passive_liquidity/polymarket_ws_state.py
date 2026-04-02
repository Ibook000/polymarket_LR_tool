"""
Thread-safe in-memory state for Polymarket CLOB user/market WebSocket feeds.

WebSocket callbacks only mutate this hub; the main loop reads and reconciles with REST.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from passive_liquidity.models import OrderBookSnapshot
from passive_liquidity.orderbook_fetcher import (
    _best_ask_from_levels,
    _best_bid_from_levels,
    second_best_ask_from_levels,
    second_best_bid_from_levels,
)

LOG = logging.getLogger(__name__)


def _parse_ws_ts(raw: Any) -> float:
    if raw is None or raw == "":
        return time.time()
    try:
        t = float(raw)
        if t > 1e12:
            t /= 1000.0
        return t
    except (TypeError, ValueError):
        return time.time()


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class UserOrderWsRecord:
    order_id: str
    asset_id: str = ""
    market: str = ""
    side: str = ""
    price: float = 0.0
    original_size: float = 0.0
    size_matched: float = 0.0
    remaining_size: float = 0.0
    order_type: str = ""  # PLACEMENT / UPDATE / CANCELLATION
    last_event_ts: float = 0.0
    last_trade_status: str = ""
    last_trade_ts: float = 0.0


@dataclass
class MarketTokenWsRecord:
    asset_id: str
    bids: list[dict[str, Any]] = field(default_factory=list)
    asks: list[dict[str, Any]] = field(default_factory=list)
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    second_best_bid: Optional[float] = None
    second_best_ask: Optional[float] = None
    tick_size: float = 0.01
    last_trade_price: Optional[float] = None
    last_trade_side: str = ""
    last_update_ts: float = 0.0
    neg_risk: bool = False


class PolymarketWsHub:
    def __init__(self, *, stale_sec: float) -> None:
        self._stale_sec = max(5.0, float(stale_sec))
        self._lock = threading.Lock()
        self._user_orders: dict[str, UserOrderWsRecord] = {}
        self._user_connected = False
        self._user_last_event_ts = 0.0
        self._user_last_connect_mono = 0.0
        self._user_last_disconnect_mono = 0.0
        self._user_sub_sent_ok = False
        self._user_last_error: str = ""

        self._market_tokens: dict[str, MarketTokenWsRecord] = {}
        self._market_connected = False
        self._market_last_event_ts = 0.0
        self._market_last_connect_mono = 0.0
        self._market_last_disconnect_mono = 0.0
        self._market_sub_sent_ok = False
        self._market_last_error: str = ""

        # Recent trades for monitoring (token_id -> deque of dicts compatible with fill_risk)
        self._activity: dict[str, deque[dict[str, Any]]] = {}
        self._activity_max = 800

    # --- user channel writes (from WS thread) ---

    def user_set_connected(self, ok: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._user_connected = ok
            if ok:
                self._user_last_connect_mono = now
            else:
                self._user_last_disconnect_mono = now

    def user_mark_subscription_ok(self, ok: bool) -> None:
        with self._lock:
            self._user_sub_sent_ok = ok

    def user_set_error(self, msg: str) -> None:
        with self._lock:
            self._user_last_error = (msg or "")[:500]

    def user_touch_event(self) -> None:
        with self._lock:
            self._user_last_event_ts = time.time()

    def user_apply_order_message(self, msg: dict[str, Any]) -> None:
        oid = str(msg.get("id") or "")
        if not oid:
            return
        aid = str(msg.get("asset_id") or "")
        mkt = str(msg.get("market") or "")
        typ = str(msg.get("type") or "").upper()
        ts = _parse_ws_ts(msg.get("timestamp"))
        orig = _f(msg.get("original_size"))
        matched = _f(msg.get("size_matched"))
        rem = _f(msg.get("size")) if msg.get("size") is not None else max(0.0, orig - matched)
        side = str(msg.get("side") or "").upper()
        price = _f(msg.get("price"))
        with self._lock:
            self._user_last_event_ts = time.time()
            rec = self._user_orders.get(oid)
            if rec is None:
                rec = UserOrderWsRecord(order_id=oid)
                self._user_orders[oid] = rec
            rec.asset_id = aid or rec.asset_id
            rec.market = mkt or rec.market
            rec.side = side or rec.side
            rec.price = price or rec.price
            rec.original_size = max(orig, rec.original_size)
            rec.size_matched = max(matched, rec.size_matched)
            rec.remaining_size = rem
            rec.order_type = typ
            rec.last_event_ts = ts

    def user_apply_trade_message(self, msg: dict[str, Any]) -> None:
        ts = _parse_ws_ts(
            msg.get("timestamp") or msg.get("matchtime") or msg.get("last_update")
        )
        status = str(msg.get("status") or "").upper()
        aid = str(msg.get("asset_id") or "")
        side = str(msg.get("side") or "").upper()
        price = _f(msg.get("price"))
        size = _f(msg.get("size"))
        taker_oid = str(msg.get("taker_order_id") or "")
        with self._lock:
            self._user_last_event_ts = time.time()
            if aid:
                self._append_activity(
                    aid,
                    {
                        "timestamp": int(ts * 1000) if ts > 1e9 else ts,
                        "match_time": ts,
                        "size": size,
                        "price": price,
                        "side": side,
                        "asset_id": aid,
                    },
                )
            for mo in msg.get("maker_orders") or []:
                if not isinstance(mo, dict):
                    continue
                m_oid = str(mo.get("order_id") or "")
                if not m_oid:
                    continue
                m_aid = str(mo.get("asset_id") or aid)
                m_amt = _f(mo.get("matched_amount"))
                m_px = _f(mo.get("price"))
                rec = self._user_orders.get(m_oid)
                if rec is None:
                    rec = UserOrderWsRecord(order_id=m_oid, asset_id=m_aid)
                    self._user_orders[m_oid] = rec
                else:
                    rec.asset_id = m_aid or rec.asset_id
                if m_amt > 0:
                    prev_sm = rec.size_matched
                    rec.size_matched = prev_sm + m_amt
                    if rec.original_size > 1e-9:
                        rec.size_matched = min(rec.size_matched, rec.original_size)
                rec.last_trade_status = status
                rec.last_trade_ts = ts
                if m_aid:
                    self._append_activity(
                        m_aid,
                        {
                            "timestamp": int(ts * 1000) if ts > 1e9 else ts,
                            "match_time": ts,
                            "size": m_amt,
                            "price": m_px,
                            "side": side,
                            "asset_id": m_aid,
                            "maker_order_id": m_oid,
                        },
                    )
            if taker_oid:
                rec_t = self._user_orders.get(taker_oid)
                if rec_t is None:
                    rec_t = UserOrderWsRecord(order_id=taker_oid, asset_id=aid)
                    self._user_orders[taker_oid] = rec_t
                rec_t.last_trade_status = status
                rec_t.last_trade_ts = ts

    def _append_activity(self, token_id: str, row: dict[str, Any]) -> None:
        if not token_id:
            return
        dq = self._activity.get(token_id)
        if dq is None:
            dq = deque(maxlen=self._activity_max)
            self._activity[token_id] = dq
        dq.append(row)

    # --- market channel writes ---

    def market_set_connected(self, ok: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._market_connected = ok
            if ok:
                self._market_last_connect_mono = now
            else:
                self._market_last_disconnect_mono = now

    def market_mark_subscription_ok(self, ok: bool) -> None:
        with self._lock:
            self._market_sub_sent_ok = ok

    def market_set_error(self, msg: str) -> None:
        with self._lock:
            self._market_last_error = (msg or "")[:500]

    def market_touch_event(self) -> None:
        with self._lock:
            self._market_last_event_ts = time.time()

    def market_apply_book(self, msg: dict[str, Any]) -> None:
        aid = str(msg.get("asset_id") or "")
        if not aid:
            return
        bids = msg.get("bids") or []
        asks = msg.get("asks") or []
        if not isinstance(bids, list):
            bids = []
        if not isinstance(asks, list):
            asks = []
        ts = _parse_ws_ts(msg.get("timestamp"))
        with self._lock:
            self._market_last_event_ts = time.time()
            rec = self._market_tokens.get(aid)
            if rec is None:
                rec = MarketTokenWsRecord(asset_id=aid)
                self._market_tokens[aid] = rec
            rec.bids = [dict(x) for x in bids if isinstance(x, dict)]
            rec.asks = [dict(x) for x in asks if isinstance(x, dict)]
            rec.best_bid = _best_bid_from_levels(rec.bids)
            rec.best_ask = _best_ask_from_levels(rec.asks)
            rec.second_best_bid = second_best_bid_from_levels(rec.bids)
            rec.second_best_ask = second_best_ask_from_levels(rec.asks)
            rec.last_update_ts = ts

    def market_apply_best_bid_ask(self, msg: dict[str, Any]) -> None:
        aid = str(msg.get("asset_id") or "")
        if not aid:
            return
        bb = msg.get("best_bid")
        ba = msg.get("best_ask")
        ts = _parse_ws_ts(msg.get("timestamp"))
        with self._lock:
            self._market_last_event_ts = time.time()
            rec = self._market_tokens.get(aid)
            if rec is None:
                rec = MarketTokenWsRecord(asset_id=aid)
                self._market_tokens[aid] = rec
            if bb is not None and str(bb).strip() != "":
                rec.best_bid = _f(bb)
            if ba is not None and str(ba).strip() != "":
                rec.best_ask = _f(ba)
            rec.last_update_ts = ts

    def market_apply_tick_size_change(self, msg: dict[str, Any]) -> None:
        aid = str(msg.get("asset_id") or "")
        if not aid:
            return
        nt = msg.get("new_tick_size")
        ts = _parse_ws_ts(msg.get("timestamp"))
        with self._lock:
            self._market_last_event_ts = time.time()
            rec = self._market_tokens.get(aid)
            if rec is None:
                rec = MarketTokenWsRecord(asset_id=aid)
                self._market_tokens[aid] = rec
            if nt is not None and str(nt).strip() != "":
                rec.tick_size = max(_f(nt), 1e-6)
            rec.last_update_ts = ts

    def market_apply_last_trade_price(self, msg: dict[str, Any]) -> None:
        aid = str(msg.get("asset_id") or "")
        if not aid:
            return
        px = _f(msg.get("price"))
        sz = _f(msg.get("size"))
        side = str(msg.get("side") or "").upper()
        ts = _parse_ws_ts(msg.get("timestamp"))
        with self._lock:
            self._market_last_event_ts = time.time()
            rec = self._market_tokens.get(aid)
            if rec is None:
                rec = MarketTokenWsRecord(asset_id=aid)
                self._market_tokens[aid] = rec
            rec.last_trade_price = px
            rec.last_trade_side = side
            rec.last_update_ts = ts
            self._append_activity(
                aid,
                {
                    "timestamp": int(ts * 1000) if ts > 1e9 else ts,
                    "match_time": ts,
                    "size": sz,
                    "price": px,
                    "side": side,
                    "asset_id": aid,
                },
            )

    def market_apply_price_change(self, msg: dict[str, Any]) -> None:
        ts = _parse_ws_ts(msg.get("timestamp"))
        for ch in msg.get("price_changes") or []:
            if not isinstance(ch, dict):
                continue
            aid = str(ch.get("asset_id") or "")
            if not aid:
                continue
            bb = ch.get("best_bid")
            ba = ch.get("best_ask")
            with self._lock:
                self._market_last_event_ts = time.time()
                rec = self._market_tokens.get(aid)
                if rec is None:
                    rec = MarketTokenWsRecord(asset_id=aid)
                    self._market_tokens[aid] = rec
                if bb is not None and str(bb).strip() != "":
                    rec.best_bid = _f(bb)
                if ba is not None and str(ba).strip() != "":
                    rec.best_ask = _f(ba)
                rec.last_update_ts = ts

    # --- reads for main loop ---

    def user_channel_healthy(self) -> bool:
        with self._lock:
            if not self._user_connected or not self._user_sub_sent_ok:
                return False
            return (time.time() - self._user_last_event_ts) < self._stale_sec * 3

    def user_connected_flag(self) -> bool:
        with self._lock:
            return self._user_connected

    def user_last_event_ts(self) -> float:
        with self._lock:
            return float(self._user_last_event_ts)

    def user_stale(self) -> bool:
        with self._lock:
            if not self._user_connected:
                return True
            return (time.time() - self._user_last_event_ts) > self._stale_sec * 3

    def market_channel_healthy(self, token_id: str) -> bool:
        with self._lock:
            if not self._market_connected or not self._market_sub_sent_ok:
                return False
            rec = self._market_tokens.get(token_id)
            if rec is None:
                return False
            return (time.time() - rec.last_update_ts) <= self._stale_sec

    def market_stale(self, token_id: str) -> bool:
        with self._lock:
            if not self._market_connected:
                return True
            rec = self._market_tokens.get(token_id)
            if rec is None:
                return True
            return (time.time() - rec.last_update_ts) > self._stale_sec

    def market_connected_flag(self) -> bool:
        with self._lock:
            return self._market_connected

    def get_user_size_matched(self, order_id: str) -> Optional[float]:
        with self._lock:
            rec = self._user_orders.get(str(order_id))
            if rec is None:
                return None
            if rec.size_matched <= 1e-12:
                return None
            return float(rec.size_matched)

    def reconcile_user_orders_with_rest(self, open_orders: list[dict]) -> None:
        """Raise WS cumulative matched to at least REST snapshot (repair lag/drift)."""
        from passive_liquidity.order_manager import _oid

        def _rest_cum_filled(o: dict) -> float:
            try:
                m = max(0.0, float(o.get("size_matched") or 0))
            except (TypeError, ValueError):
                m = 0.0
            if m > 1e-12:
                return m
            try:
                rem = max(0.0, float(o.get("remaining_size") or o.get("size") or 0))
            except (TypeError, ValueError):
                rem = 0.0
            try:
                orig = max(0.0, float(o.get("original_size") or 0))
            except (TypeError, ValueError):
                orig = 0.0
            if orig > 1e-12:
                return max(0.0, orig - rem)
            return 0.0

        with self._lock:
            for o in open_orders:
                if not isinstance(o, dict):
                    continue
                oid = _oid(o)
                if not oid:
                    continue
                rest_cum = float(_rest_cum_filled(o))
                rec = self._user_orders.get(oid)
                if rec is None:
                    rec = UserOrderWsRecord(order_id=oid)
                    self._user_orders[oid] = rec
                rec.size_matched = max(rec.size_matched, rest_cum)

    def prune_user_orders_not_in(self, keep_ids: set[str]) -> None:
        with self._lock:
            dead = [k for k in self._user_orders if k not in keep_ids]
            for k in dead:
                del self._user_orders[k]

    def orderbook_from_ws(self, token_id: str) -> Optional[OrderBookSnapshot]:
        with self._lock:
            rec = self._market_tokens.get(token_id)
            if rec is None:
                return None
            if (
                rec.best_bid is None
                and rec.best_ask is None
                and not rec.bids
                and not rec.asks
            ):
                return None
            bids = rec.bids
            asks = rec.asks
            bb = rec.best_bid if rec.best_bid is not None else _best_bid_from_levels(bids)
            ba = rec.best_ask if rec.best_ask is not None else _best_ask_from_levels(asks)
            tick = float(rec.tick_size or 0.01)
            return OrderBookSnapshot(
                best_bid=bb,
                best_ask=ba,
                tick_size=tick,
                neg_risk=rec.neg_risk,
                bids=bids,
                asks=asks,
                raw={"source": "ws_market"},
            )

    def get_market_tick_size(self, token_id: str) -> Optional[float]:
        with self._lock:
            rec = self._market_tokens.get(token_id)
            if rec is None or rec.tick_size <= 0:
                return None
            return float(rec.tick_size)

    def activity_trades(
        self, token_id: str, *, now: float, lookback_sec: float
    ) -> list[dict[str, Any]]:
        cutoff = now - max(1.0, lookback_sec)
        with self._lock:
            dq = self._activity.get(token_id)
            if not dq:
                return []
            out: list[dict[str, Any]] = []
            for row in dq:
                ts = row.get("match_time")
                if ts is None:
                    continue
                try:
                    t = float(ts)
                    if t > 1e12:
                        t /= 1000.0
                except (TypeError, ValueError):
                    continue
                if t >= cutoff:
                    out.append(dict(row))
            return out

    def connection_debug(self) -> dict[str, Any]:
        with self._lock:
            return {
                "user_connected": self._user_connected,
                "user_sub_ok": self._user_sub_sent_ok,
                "user_last_event_ts": self._user_last_event_ts,
                "user_error": self._user_last_error,
                "market_connected": self._market_connected,
                "market_sub_ok": self._market_sub_sent_ok,
                "market_last_event_ts": self._market_last_event_ts,
                "market_error": self._market_last_error,
            }
