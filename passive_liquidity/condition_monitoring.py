"""
Read-only market condition monitoring for logging and Telegram alerts.

Must not be imported from order placement / cancel / replace paths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.fill_risk import (
    build_fill_risk_context,
    count_trades_in_lookback,
    tape_buy_sell_notional,
)


@dataclass(frozen=True)
class FillMonitorSnapshot:
    fill_rate: float
    short_window_trades: int
    long_window_trades: int
    fill_risk_score: float
    adverse_share: float
    direction_en: str
    direction_zh: str


@dataclass
class _SendRecord:
    mono: float
    fingerprint: str
    metrics: dict[str, float]


def tape_direction_for_order(
    order_side: str, buy_n: float, sell_n: float
) -> tuple[str, str, float]:
    """
    For a resting BUY, adverse tape is SELL (hits bids). For SELL, adverse is BUY.
    Returns (direction_en, direction_zh, adverse_share in [0,1]).
    """
    total = buy_n + sell_n
    if total < 1e-12:
        return "no tape", "近期无成交方向信息", 0.5
    buy_share = buy_n / total
    sell_share = sell_n / total
    o = order_side.upper()
    if o == "BUY":
        adverse = sell_share
        if sell_share >= 0.55:
            return "aggressive selling", "卖盘成交为主（偏 aggressive selling）", adverse
        if buy_share >= 0.55:
            return "aggressive buying", "买盘成交为主（偏 aggressive buying）", adverse
        return "balanced", "买卖成交较均衡", adverse
    if o == "SELL":
        adverse = buy_share
        if buy_share >= 0.55:
            return "aggressive buying", "买盘成交为主（偏 aggressive buying）", adverse
        if sell_share >= 0.55:
            return "aggressive selling", "卖盘成交为主（偏 aggressive selling）", adverse
        return "balanced", "买卖成交较均衡", adverse
    return "unknown side", "订单方向未知", 0.5


def build_fill_monitor_snapshot(
    trades: list[dict],
    *,
    order_side: str,
    price: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    c: PassiveConfig,
    now: Optional[float] = None,
) -> FillMonitorSnapshot:
    now = now or time.time()
    short_sec = max(1.0, float(c.monitor_short_trade_lookback_sec))
    long_sec = max(1.0, float(c.fill_lookback_sec))
    ctx = build_fill_risk_context(
        trades,
        order_side=order_side,
        price=price,
        best_bid=best_bid,
        best_ask=best_ask,
        tick=tick,
        c=c,
        now=now,
    )
    fill_rate = float(ctx.activity_long_count_only)
    short_n = count_trades_in_lookback(trades, now, short_sec)
    long_n = count_trades_in_lookback(trades, now, long_sec)
    buy_n, sell_n = tape_buy_sell_notional(trades, now, short_sec)
    dir_en, dir_zh, adverse = tape_direction_for_order(order_side, buy_n, sell_n)
    return FillMonitorSnapshot(
        fill_rate=fill_rate,
        short_window_trades=short_n,
        long_window_trades=long_n,
        fill_risk_score=float(ctx.fill_risk_score),
        adverse_share=float(adverse),
        direction_en=dir_en,
        direction_zh=dir_zh,
    )


def fill_alert_condition(
    snap: FillMonitorSnapshot,
    c: PassiveConfig,
) -> tuple[bool, list[str]]:
    """Returns (triggered, list of reason tags for logging)."""
    reasons: list[str] = []
    if snap.fill_rate > c.alert_fill_rate_threshold:
        reasons.append("fill_rate")
    if snap.short_window_trades >= c.alert_short_trades_threshold:
        reasons.append("short_trades")
    if snap.fill_risk_score > c.alert_fill_risk_score_threshold:
        reasons.append("fill_risk_score")
    if snap.adverse_share >= c.alert_direction_imbalance_min:
        reasons.append("direction_imbalance")
    return (len(reasons) > 0, reasons)


def fill_alert_fingerprint(snap: FillMonitorSnapshot) -> str:
    return (
        f"{snap.fill_rate:.3f}|{snap.short_window_trades}|{snap.long_window_trades}|"
        f"{snap.fill_risk_score:.3f}|{snap.adverse_share:.3f}"
    )


def fill_metrics_dict(snap: FillMonitorSnapshot) -> dict[str, float]:
    return {
        "fill_rate": snap.fill_rate,
        "short_trades": float(snap.short_window_trades),
        "long_trades": float(snap.long_window_trades),
        "fill_risk_score": snap.fill_risk_score,
        "adverse_share": snap.adverse_share,
    }


def _significant_fill_worsening(
    prev: dict[str, float], cur: dict[str, float], c: PassiveConfig
) -> bool:
    return (
        cur["fill_risk_score"]
        >= prev["fill_risk_score"] + c.alert_significant_fill_risk_delta
        or cur["short_trades"]
        >= prev["short_trades"] + float(c.alert_significant_short_trades_delta)
        or cur["fill_rate"] >= prev["fill_rate"] + c.alert_significant_fill_rate_delta
        or cur["adverse_share"]
        >= prev["adverse_share"] + c.alert_significant_adverse_share_delta
    )


def depth_metrics_dict(total: float, closer: float, ratio: float) -> dict[str, float]:
    return {
        "total_depth": total,
        "closer_depth": closer,
        "depth_ratio": ratio,
    }


def depth_alert_fingerprint(
    band_lo: float, band_hi: float, total: float, closer: float, ratio: float
) -> str:
    return f"{band_lo:.4f}|{band_hi:.4f}|{total:.1f}|{closer:.1f}|{ratio:.4f}"


def _significant_depth_worsening(
    prev: dict[str, float], cur: dict[str, float], c: PassiveConfig
) -> bool:
    return cur["depth_ratio"] >= prev["depth_ratio"] + c.alert_significant_depth_ratio_delta


class PassiveMonitorAlertGate:
    """
    Cooldown + dedupe for passive monitor Telegram alerts only.
    """

    def __init__(self, c: PassiveConfig) -> None:
        self._c = c
        self._fill_prev_condition: dict[str, bool] = {}
        self._fill_last_send: dict[str, _SendRecord] = {}
        self._depth_prev_condition: dict[str, bool] = {}
        self._depth_last_send: dict[str, _SendRecord] = {}

    def should_send_fill_alert(
        self,
        key: str,
        *,
        now_mono: float,
        triggered: bool,
        fingerprint: str,
        metrics: dict[str, float],
    ) -> bool:
        if not self._c.alert_monitoring_enabled:
            return False
        if not triggered:
            self._fill_prev_condition[key] = False
            return False
        prev_on = self._fill_prev_condition.get(key, False)
        newly = not prev_on
        self._fill_prev_condition[key] = True

        rec = self._fill_last_send.get(key)
        cd = max(0.0, float(self._c.alert_cooldown_sec))

        if rec is None:
            return True
        elapsed = now_mono - rec.mono
        if newly:
            return True
        if _significant_fill_worsening(rec.metrics, metrics, self._c):
            return True
        if elapsed >= cd:
            return True
        return False

    def record_fill_sent(
        self,
        key: str,
        *,
        now_mono: float,
        fingerprint: str,
        metrics: dict[str, float],
    ) -> None:
        self._fill_last_send[key] = _SendRecord(
            mono=now_mono, fingerprint=fingerprint, metrics=dict(metrics)
        )

    def should_send_depth_alert(
        self,
        key: str,
        *,
        now_mono: float,
        triggered: bool,
        fingerprint: str,
        metrics: dict[str, float],
    ) -> bool:
        if not self._c.alert_monitoring_enabled:
            return False
        if not triggered:
            self._depth_prev_condition[key] = False
            return False
        prev_on = self._depth_prev_condition.get(key, False)
        newly = not prev_on
        self._depth_prev_condition[key] = True

        rec = self._depth_last_send.get(key)
        cd = max(0.0, float(self._c.alert_cooldown_sec))

        if rec is None:
            return True
        elapsed = now_mono - rec.mono
        if newly:
            return True
        if _significant_depth_worsening(rec.metrics, metrics, self._c):
            return True
        if elapsed >= cd:
            return True
        return False

    def record_depth_sent(
        self,
        key: str,
        *,
        now_mono: float,
        fingerprint: str,
        metrics: dict[str, float],
    ) -> None:
        self._depth_last_send[key] = _SendRecord(
            mono=now_mono, fingerprint=fingerprint, metrics=dict(metrics)
        )

    def reset_cycle_flags_when_idle(self) -> None:
        """Optional: call when there are no orders to clear stale 'condition true' memory."""
        self._fill_prev_condition.clear()
        self._depth_prev_condition.clear()
