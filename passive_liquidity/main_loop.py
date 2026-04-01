from __future__ import annotations

import logging
import time
from typing import Any, Optional

from passive_liquidity.account_portfolio import (
    fetch_collateral_snapshot,
    half_hour_slot_key,
    read_optional_deposit_env,
    resolve_deposit_reference,
    seconds_until_next_half_hour_boundary,
)
from passive_liquidity.bridge_deposits import fetch_bridge_polygon_usdc_deposits
from passive_liquidity.polygon_deposits import fetch_polygon_usdc_deposit_summary
from passive_liquidity.cancel_reason_zh import cancel_category_zh
from passive_liquidity.clob_factory import build_trading_client, funder_address
from passive_liquidity.condition_monitoring import (
    PassiveMonitorAlertGate,
    build_fill_monitor_snapshot,
    depth_alert_fingerprint,
    depth_metrics_dict,
    fill_alert_condition,
    fill_alert_fingerprint,
    fill_metrics_dict,
)
from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.fill_detection import FillNotificationTracker
from passive_liquidity.logger_setup import setup_logging
from passive_liquidity.market_display import MarketDisplayResolver
from passive_liquidity.order_manager import (
    OrderManager,
    _market,
    _oid,
    _price,
    _remaining_size,
    _side,
    _token_id,
)
from passive_liquidity.models import OrderBookSnapshot
from passive_liquidity.orderbook_fetcher import OrderBookFetcher
from passive_liquidity.reward_monitor import RewardMonitor
from passive_liquidity.risk_manager import RiskManager
from passive_liquidity.simple_price_policy import (
    compute_eligible_band_depth_stats,
    decide_simple_price,
    format_eligible_band_depth_summary_zh,
)
from passive_liquidity.telegram_notifier import (
    OrderEventFormat,
    TelegramNotifier,
    build_telegram_notifier_from_env,
    polymarket_api_error_zh_hint,
    scoring_status_text,
    stable_fingerprint,
)

LOG = logging.getLogger("main_loop")


def _order_display_meta(order: dict) -> tuple[str, str]:
    title = str(
        order.get("question")
        or order.get("market_question")
        or order.get("title")
        or ""
    ).strip()
    if not title:
        slug = order.get("market_slug") or order.get("slug") or ""
        title = str(slug).strip() if slug else ""
    if not title:
        mid = str(order.get("market") or order.get("condition_id") or "").strip()
        title = (mid[:48] + "…") if len(mid) > 48 else mid if mid else "(未知盘口)"
    outcome = str(order.get("outcome") or order.get("outcome_name") or "").strip()
    return title, outcome


def _order_has_human_market_copy(order: dict) -> bool:
    """True if CLOB already returned a slug or question (no Gamma lookup needed)."""
    if str(order.get("question") or order.get("market_question") or order.get("title") or "").strip():
        return True
    if str(order.get("market_slug") or order.get("slug") or "").strip():
        return True
    return False


def _resolve_order_display(
    resolver: Optional[MarketDisplayResolver],
    order: dict,
    condition_id: str,
    token_id: str,
) -> tuple[str, str]:
    title, outcome = _order_display_meta(order)
    if resolver is None or _order_has_human_market_copy(order):
        return title, outcome
    gq, go = resolver.lookup(condition_id, token_id)
    if gq:
        title = gq
    if go:
        outcome = go
    return title, outcome


def _telegram_order_event(
    tg: TelegramNotifier,
    event_key: str,
    order: dict,
    *,
    condition_id: str,
    token_id: str,
    display_resolver: Optional[MarketDisplayResolver],
    side: str,
    inventory: float,
    scoring_status_text_s: str,
    old_price: Optional[float],
    new_price: Optional[float],
    size: Optional[float],
    reason: str,
) -> None:
    if not tg.enabled:
        return
    title, outcome = _resolve_order_display(display_resolver, order, condition_id, token_id)
    ev = OrderEventFormat(
        account_label=tg.account_label,
        market_title=title,
        outcome=outcome,
        token_id=token_id,
        side=side,
        old_price=old_price,
        new_price=new_price,
        size=size,
        scoring_status_text=scoring_status_text_s,
        inventory=inventory if abs(inventory) > 1e-8 else None,
        reason=reason,
    )
    text = tg.format_order_event_message(ev)
    fp = stable_fingerprint(text)
    LOG.info("Telegram order event key=%s", event_key)
    tg.send_message(text, event_key=event_key, payload_hash=fp)


def _resolve_initial_frozen_whitelist(
    client,
    order_manager: OrderManager,
    env_whitelist: frozenset[str],
) -> tuple[frozenset[str], str, Optional[int]]:
    """
    Build the process-lifetime whitelist.

    - If PASSIVE_TOKEN_WHITELIST (env) is non-empty: use it (frozen).
    - Else: unique token_ids from a one-time fetch of open orders at startup (frozen).
    Returns (whitelist, source_label, open_order_count when seeded from orders else None).
    """
    if env_whitelist:
        return (frozenset(env_whitelist), "PASSIVE_TOKEN_WHITELIST", None)
    seed_orders = order_manager.fetch_all_open_orders(client)
    tokens = frozenset(
        _token_id(o)
        for o in seed_orders
        if isinstance(o, dict) and _oid(o) and _token_id(o) and _market(o)
    )
    return (tokens, "open_orders_at_startup", len(seed_orders))


def main() -> None:
    setup_logging()
    config = PassiveConfig.from_env()
    telegram = build_telegram_notifier_from_env()
    if telegram.enabled:
        LOG.info("Telegram notifications enabled (account=%s)", telegram.account_label)
    else:
        LOG.info("Telegram notifications disabled or misconfigured")

    print(
        "白名单监控：启动时固定白名单（见下方）。运行期间出现的新 token_id 一律忽略。\n"
        "· 若设置了 PASSIVE_TOKEN_WHITELIST：以其为准。\n"
        "· 否则：从启动当刻的未成交单中提取唯一 token_id 作为白名单。\n"
        "调价仅按简化规则（粗 tick 盘口档位 / 细 tick 带宽比例）；不会新建订单。\n"
        "若该 outcome 已有仓位（库存非零），则不对该盘口挂单做任何处理（不撤单、不改价、不参与监控逻辑）。\n",
        flush=True,
    )

    client = build_trading_client(config.clob_host, config.chain_id)

    from py_clob_client.client import ClobClient

    ro_client = ClobClient(config.clob_host, chain_id=config.chain_id)

    funder = funder_address()
    book_fetcher = OrderBookFetcher(ro_client)
    reward_monitor = RewardMonitor(config)
    risk = RiskManager(config, funder)
    order_manager = OrderManager()
    market_display = MarketDisplayResolver(config.gamma_api_host)

    frozen_whitelist, wl_source, seed_order_n = _resolve_initial_frozen_whitelist(
        client, order_manager, config.token_whitelist
    )

    seed_part = (
        f"open_orders_seen={seed_order_n}"
        if seed_order_n is not None
        else "open_orders_seen=n/a (whitelist from env)"
    )
    LOG.info(
        "=== INITIAL WHITELIST (frozen for this process) === source=%s %s "
        "unique_token_count=%d inv_threshold=%.6f ===",
        wl_source,
        seed_part,
        len(frozen_whitelist),
        config.inventory_manual_threshold,
    )
    if frozen_whitelist:
        for tid in sorted(frozen_whitelist):
            LOG.info("WHITELIST token_id=%s", tid)
    else:
        LOG.warning(
            "Initial whitelist is EMPTY — no token_ids will be managed until you restart "
            "after placing orders (or set PASSIVE_TOKEN_WHITELIST)."
        )

    seed_note = (
        f" 启动时未成交单数={seed_order_n}"
        if seed_order_n is not None
        else "（白名单来自环境变量，未按挂单推断）"
    )
    print(
        f"【启动白名单】来源={wl_source} {seed_note} 唯一 token 数={len(frozen_whitelist)} "
        f"库存门槛={config.inventory_manual_threshold}\n",
        flush=True,
    )
    if frozen_whitelist:
        for tid in sorted(frozen_whitelist):
            print(f"  · {tid}", flush=True)
        print("", flush=True)
    else:
        print("  （空）之后新挂的单不会自动加入白名单；需重启或配置 PASSIVE_TOKEN_WHITELIST。\n", flush=True)

    if telegram.enabled:
        telegram.notify_whitelist_init(
            source=wl_source,
            token_ids=sorted(frozen_whitelist),
            open_order_count=seed_order_n,
        )

    deposited_baseline = 0.0
    deposit_source_zh = ""
    env_dep = read_optional_deposit_env()
    polygon_summary = None
    try:
        polygon_summary = fetch_polygon_usdc_deposit_summary(funder)
    except Exception as e:
        LOG.debug("On-chain deposit fetch skipped: %s", e)

    bridge_summary = None
    try:
        bridge_summary = fetch_bridge_polygon_usdc_deposits(funder)
    except Exception as e:
        LOG.debug("Bridge deposit fetch skipped: %s", e)

    try:
        bal_orders = order_manager.fetch_all_open_orders(client)
    except Exception as e:
        LOG.warning("Startup balance snapshot: open orders unavailable: %s", e)
        bal_orders = []
    snap0 = fetch_collateral_snapshot(client, bal_orders)
    startup_total = float(snap0.total_balance_usdc) if snap0 else 0.0
    deposited_baseline, deposit_source_zh, deposit_approximate = resolve_deposit_reference(
        polygon_summary=polygon_summary,
        env_override=env_dep,
        bridge_summary=bridge_summary,
        startup_total_balance=startup_total,
    )
    if deposited_baseline is not None:
        LOG.info(
            "Deposit reference: %.4f USDC — %s",
            deposited_baseline,
            deposit_source_zh,
        )
    else:
        LOG.info("Deposit reference: not configured — %s", deposit_source_zh)
    LOG.debug(
        "deposit baseline: deposit_reference_source=%s deposited_reference=%.6f "
        "approximate=%s",
        deposit_source_zh,
        deposited_baseline,
        deposit_approximate,
    )

    if snap0:
        pnl0 = (
            None
            if deposited_baseline is None
            else snap0.total_balance_usdc - deposited_baseline
        )
        LOG.debug(
            "account snapshot (startup): api_account_total=%.6f api_collateral=%.6f "
            "locked_amount_in_open_orders=%.6f computed_available_balance=%.6f "
            "deposited_reference=%s computed_pnl=%s",
            snap0.total_balance_usdc,
            snap0.api_collateral_usdc,
            snap0.locked_open_buy_usdc,
            snap0.available_balance_usdc,
            f"{deposited_baseline:.6f}" if deposited_baseline is not None else "None",
            f"{pnl0:.6f}" if pnl0 is not None else "None",
        )
        extra_note = ""
        if polygon_summary is not None and polygon_summary.approximate:
            extra_note = (polygon_summary.note_zh or "").strip()
        elif deposited_baseline is None:
            extra_note = (
                "说明: 未自动取得累计入账参考。当前账户总额≠历史累计充值，"
                "请勿混用。请在 .env 设置 TELEGRAM_TOTAL_DEPOSITED_USDC，"
                "或配置 POLYGONSCAN_API_KEY；也可尝试 Bridge API 是否包含您的入账。"
            )
        elif "Bridge API" in deposit_source_zh:
            extra_note = "说明: 入账参考来自 Polymarket Bridge API 已完成的 Polygon USDC 入账。"
        elif "临时参考：启动时账户总额" in deposit_source_zh:
            extra_note = (
                "说明: 已开启 PASSIVE_USE_STARTUP_TOTAL_AS_DEPOSIT_REF；"
                "该参考通常不等于真实累计充值，仅作临时盈亏口径。"
            )
        if telegram.enabled:
            telegram.notify_account_startup(
                deposited_reference_usdc=deposited_baseline,
                total_account_usdc=snap0.total_balance_usdc,
                available_balance_usdc=snap0.available_balance_usdc,
                locked_open_buy_usdc=snap0.locked_open_buy_usdc,
                pnl_usdc=pnl0,
                extra_note_zh=extra_note,
            )
    elif telegram.enabled:
        LOG.warning(
            "Telegram: startup account snapshot skipped (collateral API failed); "
            "入账参考=%.4f（%s）",
            deposited_baseline,
            deposit_source_zh,
        )

    next_summary_at = time.time() + seconds_until_next_half_hour_boundary()
    last_summary_slot: Optional[str] = None

    LOG.info(
        "Whitelist monitoring started; interval=%.1fs post_only=%s",
        config.loop_interval,
        config.monitoring_post_only,
    )

    error_streak = 0
    fill_tracker = FillNotificationTracker()
    monitor_alert_gate = PassiveMonitorAlertGate(config)
    next_band_summary_at = time.time() + max(
        1.0, float(config.telegram_band_summary_interval_sec)
    )

    while True:
        try:
            orders = order_manager.fetch_all_open_orders(client)
            now = time.time()
            band_summary_rows: list[dict[str, Any]] = []
            band_summary_eligible_n = 0
            if telegram.enabled and now >= next_summary_at:
                slot = half_hour_slot_key(now)
                if slot != last_summary_slot:
                    last_summary_slot = slot
                    snap_p = fetch_collateral_snapshot(client, orders)
                    if snap_p:
                        pnl_p = (
                            None
                            if deposited_baseline is None
                            else snap_p.total_balance_usdc - deposited_baseline
                        )
                        LOG.debug(
                            "account summary (periodic): api_account_total=%.6f "
                            "api_collateral=%.6f locked_amount_in_open_orders=%.6f "
                            "computed_available_balance=%.6f deposit_reference_source=%s "
                            "deposited_reference=%s computed_pnl=%s",
                            snap_p.total_balance_usdc,
                            snap_p.api_collateral_usdc,
                            snap_p.locked_open_buy_usdc,
                            snap_p.available_balance_usdc,
                            deposit_source_zh,
                            f"{deposited_baseline:.6f}"
                            if deposited_baseline is not None
                            else "None",
                            f"{pnl_p:.6f}" if pnl_p is not None else "None",
                        )
                        LOG.info("Telegram periodic account summary slot=%s", slot)
                        time_label = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(now)
                        )
                        telegram.notify_periodic_account_summary(
                            slot_key=slot,
                            time_label=time_label,
                            total_account_usdc=snap_p.total_balance_usdc,
                            available_balance_usdc=snap_p.available_balance_usdc,
                            deposited_reference_usdc=deposited_baseline,
                            pnl_usdc=pnl_p,
                        )
                    else:
                        LOG.warning("Telegram periodic summary skipped (no collateral data)")
                next_summary_at = time.time() + seconds_until_next_half_hour_boundary()

            if not orders:
                fill_tracker.clear()
                monitor_alert_gate.reset_cycle_flags_when_idle()
                LOG.info("No open orders for this API key; idle.")
            else:
                eligible_orders: list[dict] = []
                inv_by_token: dict[str, float] = {}
                position_skip_logged: set[str] = set()

                for o in orders:
                    oid = _oid(o)
                    token_id = _token_id(o)
                    condition_id = _market(o)
                    if not oid or not token_id or not condition_id:
                        LOG.warning("Skip order with missing id/market/asset: %s", o)
                        continue

                    if token_id not in frozen_whitelist:
                        continue

                    if token_id not in inv_by_token:
                        inv_by_token[token_id] = risk.get_inventory(
                            condition_id, token_id
                        )
                    inv = inv_by_token[token_id]
                    if abs(inv) > 1e-8:
                        if token_id not in position_skip_logged:
                            position_skip_logged.add(token_id)
                            LOG.info(
                                "SKIP_POSITION token_id=%s condition_id=%s inventory=%.6f — "
                                "已有仓位，本盘口挂单不处理",
                                token_id[:28],
                                condition_id[:20],
                                inv,
                            )
                        continue

                    eligible_orders.append(o)

                if frozen_whitelist and orders and not eligible_orders:
                    LOG.debug(
                        "No eligible orders this cycle (%d open): all non-whitelist or holding position",
                        len(orders),
                    )

                ids = [_oid(o) for o in eligible_orders if _oid(o)]
                scoring_map = reward_monitor.batch_order_scoring(client, ids)
                book_cache: dict[str, OrderBookSnapshot] = {}
                tokens_for_trades = {
                    _token_id(o) for o in eligible_orders if _token_id(o)
                }
                tokens_for_trades |= fill_tracker.prev_token_ids()
                trades_by_token: dict[str, list] = {}
                for _tid in tokens_for_trades:
                    trades_by_token[_tid] = risk.fetch_trades_for_token(
                        client, _tid
                    )

                def _send_fill_telegram(**kw: Any) -> None:
                    if not telegram.enabled:
                        return
                    order = kw["order"]
                    tid = kw["token_id"]
                    cid = kw["condition_id"]
                    title, outcome = _resolve_order_display(
                        market_display, order, cid, tid
                    )
                    ft_zh = "全部成交" if kw["is_full"] else "部分成交"
                    text = telegram.format_order_fill_message(
                        account_label=telegram.account_label,
                        market_title=title,
                        outcome=outcome,
                        side=str(kw["side"]),
                        order_price=float(kw["order_price"]),
                        filled_size=float(kw["filled_size"]),
                        remaining_size=float(kw["remaining_size"]),
                        fill_type_zh=ft_zh,
                        scoring_status_text_s=scoring_status_text(kw["scoring"]),
                        fill_price=kw.get("fill_price"),
                        inventory=float(kw["inventory"]),
                    )
                    fp = stable_fingerprint(
                        kw["order_id"],
                        f"{float(kw['dedupe_total_filled']):.8f}",
                    )
                    oid_key = str(kw["order_id"])[:48].replace(":", "_")
                    cum_tag = int(round(float(kw["dedupe_total_filled"]) * 1_000_000))
                    LOG.info("Telegram fill notify order=%s", kw["order_id"][:18])
                    telegram.send_message(
                        text,
                        event_key=f"fill:order:{oid_key}:{cum_tag}",
                        payload_hash=fp,
                    )

                fill_tracker.process_loop(
                    eligible_orders=eligible_orders,
                    scoring_map=scoring_map,
                    trades_by_token=trades_by_token,
                    manual_token_ids=set(),
                    config=config,
                    now=now,
                    get_inventory=lambda c_id, t_id: risk.get_inventory(c_id, t_id),
                    send_fill_telegram=_send_fill_telegram,
                )

                cycle_rows: list[dict[str, Any]] = []
                for o in eligible_orders:
                    oid = _oid(o)
                    token_id = _token_id(o)
                    condition_id = _market(o)
                    if not oid or not token_id or not condition_id:
                        continue

                    if token_id not in book_cache:
                        book_cache[token_id] = book_fetcher.get_orderbook(token_id)
                    book = book_cache[token_id]
                    mid = book.mid
                    if mid is None:
                        mid = book_fetcher.mid_price(token_id)
                    if mid is None:
                        LOG.warning("No mid for token %s; skip order %s", token_id[:24], oid[:16])
                        continue

                    tick = float(book.tick_size or 0.01)
                    rewards_spread = reward_monitor.get_rewards_max_spread_for_market(condition_id)
                    reward_range = reward_monitor.get_reward_range(mid, rewards_spread)
                    delta = max(reward_range.delta, 1e-9)

                    scoring = bool(scoring_map.get(oid, False))
                    inventory = risk.get_inventory(condition_id, token_id)
                    side = _side(o)
                    price = _price(o)
                    sz = _remaining_size(o)

                    cycle_rows.append(
                        {
                            "o": o,
                            "oid": oid,
                            "token_id": token_id,
                            "condition_id": condition_id,
                            "book": book,
                            "mid": mid,
                            "tick": tick,
                            "delta": delta,
                            "scoring": scoring,
                            "inventory": inventory,
                            "side": side,
                            "price": price,
                            "size": sz,
                        }
                    )

                band_summary_rows = cycle_rows
                band_summary_eligible_n = len(eligible_orders)

                fill_monitor_keys_done: set[str] = set()
                for row in cycle_rows:
                    o = row["o"]
                    oid = row["oid"]
                    token_id = row["token_id"]
                    condition_id = row["condition_id"]
                    book = row["book"]
                    mid = row["mid"]
                    tick = row["tick"]
                    delta = row["delta"]
                    scoring = row["scoring"]
                    inventory = row["inventory"]
                    side = row["side"]
                    price = row["price"]
                    sz_snap = row["size"]

                    fill_mkey = f"{token_id}:{str(side).upper()}"
                    if fill_mkey not in fill_monitor_keys_done:
                        fill_monitor_keys_done.add(fill_mkey)
                        trades_fm = trades_by_token.get(token_id) or []
                        snap_fm = build_fill_monitor_snapshot(
                            trades_fm,
                            order_side=str(side),
                            price=float(price),
                            best_bid=book.best_bid,
                            best_ask=book.best_ask,
                            tick=float(tick),
                            c=config,
                            now=now,
                        )
                        trig_fm, reasons_fm = fill_alert_condition(snap_fm, config)
                        LOG.info(
                            "MONITOR fill token=%s side=%s fill_rate=%.4f short_trades=%d "
                            "long_trades=%d fill_risk_score=%.4f trade_dir_en=%s trade_dir_zh=%s "
                            "adverse_share=%.4f alert=%s reasons=%s",
                            token_id[:28],
                            str(side).upper(),
                            snap_fm.fill_rate,
                            snap_fm.short_window_trades,
                            snap_fm.long_window_trades,
                            snap_fm.fill_risk_score,
                            snap_fm.direction_en,
                            snap_fm.direction_zh,
                            snap_fm.adverse_share,
                            trig_fm,
                            ",".join(reasons_fm) if reasons_fm else "",
                        )
                        if telegram.enabled and config.alert_monitoring_enabled:
                            mt_fm, oc_fm = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            fm_m = fill_metrics_dict(snap_fm)
                            fm_fp = fill_alert_fingerprint(snap_fm)
                            send_fm = monitor_alert_gate.should_send_fill_alert(
                                fill_mkey,
                                now_mono=time.monotonic(),
                                triggered=trig_fm,
                                fingerprint=fm_fp,
                                metrics=fm_m,
                            )
                            if send_fm:
                                telegram.notify_passive_fill_risk_alert(
                                    market_title=mt_fm,
                                    outcome=oc_fm,
                                    token_id=token_id,
                                    side=str(side),
                                    fill_rate=snap_fm.fill_rate,
                                    short_trades=snap_fm.short_window_trades,
                                    long_trades=snap_fm.long_window_trades,
                                    fill_risk_score=snap_fm.fill_risk_score,
                                    direction_en=snap_fm.direction_en,
                                    reasons=reasons_fm,
                                )
                                monitor_alert_gate.record_fill_sent(
                                    fill_mkey,
                                    now_mono=time.monotonic(),
                                    fingerprint=fm_fp,
                                    metrics=fm_m,
                                )

                    try:
                        dst = compute_eligible_band_depth_stats(
                            side=str(side),
                            order_price=float(price),
                            mid=float(mid),
                            delta=float(delta),
                            tick=float(tick),
                            bids=book.bids,
                            asks=book.asks,
                        )
                        tot_d = float(dst.total_in_band)
                        clo_d = float(dst.closer_to_mid_than_order)
                        ratio_d = (clo_d / tot_d) if tot_d > 1e-12 else 0.0
                        trig_d = bool(
                            tot_d > 1e-12
                            and ratio_d > float(config.alert_depth_ratio_threshold)
                        )
                        LOG.info(
                            "MONITOR depth oid=%s token=%s band=[%.4f,%.4f] total_depth=%.4f "
                            "closer_to_mid=%.4f depth_ratio=%.4f alert=%s",
                            oid[:18],
                            token_id[:28],
                            dst.scan_lo,
                            dst.scan_hi,
                            tot_d,
                            clo_d,
                            ratio_d,
                            trig_d,
                        )
                        if telegram.enabled and config.alert_monitoring_enabled:
                            mt_d, oc_d = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            dm_m = depth_metrics_dict(tot_d, clo_d, ratio_d)
                            dm_fp = depth_alert_fingerprint(
                                dst.scan_lo, dst.scan_hi, tot_d, clo_d, ratio_d
                            )
                            dk = str(oid)
                            send_d = monitor_alert_gate.should_send_depth_alert(
                                dk,
                                now_mono=time.monotonic(),
                                triggered=trig_d,
                                fingerprint=dm_fp,
                                metrics=dm_m,
                            )
                            if send_d:
                                telegram.notify_passive_depth_risk_alert(
                                    market_title=mt_d,
                                    outcome=oc_d,
                                    token_id=token_id,
                                    order_id_short=oid[:24],
                                    band_lo=dst.scan_lo,
                                    band_hi=dst.scan_hi,
                                    total_depth=tot_d,
                                    closer_depth=clo_d,
                                    depth_ratio=ratio_d,
                                )
                                monitor_alert_gate.record_depth_sent(
                                    dk,
                                    now_mono=time.monotonic(),
                                    fingerprint=dm_fp,
                                    metrics=dm_m,
                                )
                    except Exception as e:
                        LOG.debug("MONITOR depth stats skipped: %s", e)

                    decision, meta = decide_simple_price(
                        side=side,
                        price=price,
                        mid=mid,
                        tick=tick,
                        delta=delta,
                        bids=book.bids,
                        asks=book.asks,
                        min_replace_ticks=config.adjustment_min_replace_ticks,
                    )

                    def _on_replace_post_retry(attempt: int, err: str) -> None:
                        if not telegram.enabled:
                            return
                        if attempt != 1 and attempt % 5 != 0:
                            return
                        np = (
                            float(decision.new_price)
                            if decision.new_price is not None
                            else None
                        )
                        mt, oc = _resolve_order_display(
                            market_display, o, condition_id, token_id
                        )
                        hint = polymarket_api_error_zh_hint(err)
                        lines = [
                            f"订单 id: {oid[:28]}…",
                            f"token_id: {token_id}",
                            f"市场: {mt}",
                            f"方向: {oc or '—'}",
                            f"买卖: {side} 目标价: {np} 份额: {sz_snap}",
                            f"状态: 撤单已成功，第 {attempt} 次提交新单仍失败（程序在重试）",
                            "",
                            hint,
                            "",
                            "接口原文（截断）:",
                            err[:1200],
                        ]
                        telegram.notify_operational_warning_zh(
                            title_zh="改价后重新挂单失败",
                            lines=lines,
                            event_key=f"warn:replace_post:{oid}:{attempt}",
                        )

                    apply_result = order_manager.apply_decision(
                        client,
                        o,
                        decision,
                        post_only=config.monitoring_post_only,
                        delay_after_cancel_sec=config.replace_delay_after_cancel_sec,
                        replace_post_retry_interval_sec=config.replace_post_retry_interval_sec,
                        replace_post_max_retries=config.replace_post_max_retries,
                        on_replace_post_retry=_on_replace_post_retry
                        if telegram.enabled
                        else None,
                    )

                    dist = abs(mid - price)
                    dist_norm = dist / max(delta, 1e-12)
                    LOG.info(
                        "simple_order=%s %s @ %.4f | mid=%.4f dist_norm=%.3f×band | "
                        "tick_size=%s regime=%s candidates=%s candidate_count=%s "
                        "chosen_target=%s | apply=%s reason_code=%s",
                        oid[:18],
                        side,
                        price,
                        mid,
                        dist_norm,
                        meta.get("tick_size"),
                        meta.get("tick_regime"),
                        meta.get("candidate_prices"),
                        meta.get("candidate_count"),
                        meta.get("chosen_target_price"),
                        apply_result.outcome,
                        meta.get("reason_code"),
                    )

                    st = scoring_status_text(scoring)
                    if apply_result.outcome == "replaced_ok":
                        _telegram_order_event(
                            telegram,
                            f"order:{oid}:replaced",
                            o,
                            condition_id=condition_id,
                            token_id=token_id,
                            display_resolver=market_display,
                            side=side,
                            inventory=inventory,
                            scoring_status_text_s=st,
                            old_price=apply_result.old_price,
                            new_price=apply_result.new_price,
                            size=apply_result.size,
                            reason=apply_result.decision_reason,
                        )
                    elif apply_result.outcome == "canceled_ok":
                        if telegram.enabled:
                            mt, oc = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            if decision.reason == "coarse_tick_abandon_due_to_too_few_levels":
                                _cand = meta.get("candidate_prices")
                                _prices = (
                                    [float(x) for x in _cand]
                                    if isinstance(_cand, list)
                                    else []
                                )
                                telegram.notify_coarse_tick_abandon(
                                    market_title=mt,
                                    outcome=oc,
                                    token_id=token_id,
                                    n_candidates=int(meta.get("candidate_count") or 0),
                                    reason_code=decision.reason,
                                    candidate_prices=_prices,
                                    mid=float(mid),
                                    coarse_range_lo_hi=meta.get("coarse_range_lo_hi"),
                                    tick_size=float(meta["tick_size"])
                                    if meta.get("tick_size") is not None
                                    else None,
                                    reward_band_delta=float(meta["coarse_reward_band_delta"])
                                    if meta.get("coarse_reward_band_delta")
                                    is not None
                                    else None,
                                )
                            else:
                                cat_zh, det_zh, raw_r = cancel_category_zh(
                                    apply_result.decision_reason
                                )
                                telegram.notify_order_cancelled_chinese(
                                    order_id_short=oid[:24],
                                    market_title=mt,
                                    outcome=oc,
                                    price=float(apply_result.old_price or 0.0),
                                    size=float(apply_result.size or 0.0),
                                    category_zh=cat_zh,
                                    detail_zh=det_zh,
                                    raw_reason=raw_r,
                                )
                    elif apply_result.outcome in (
                        "replace_failed",
                        "replace_cancel_failed",
                    ):
                        if telegram.enabled:
                            mt, oc = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            raw_err = apply_result.error_detail or ""
                            title = (
                                "改价新单重试耗尽仍失败"
                                if apply_result.outcome == "replace_failed"
                                else "改价时撤单失败"
                            )
                            hint = polymarket_api_error_zh_hint(raw_err)
                            lines = [
                                f"订单 id: {oid[:28]}…",
                                f"token_id: {token_id}",
                                f"市场: {mt}",
                                f"方向: {oc or '—'}",
                                f"买卖: {side}",
                                f"目标价: {apply_result.new_price}",
                                f"份额: {apply_result.size}",
                                f"调价原因码: {apply_result.decision_reason}",
                                "",
                                hint,
                                "",
                                "接口原文:",
                                raw_err[:1200],
                            ]
                            telegram.notify_operational_warning_zh(
                                title_zh=f"【警告】{title}",
                                lines=lines,
                                event_key=f"warn:{apply_result.outcome}:{oid}",
                            )
                    elif apply_result.outcome == "canceled_fail":
                        if telegram.enabled:
                            mt, oc = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            raw_err = apply_result.error_detail or ""
                            hint = polymarket_api_error_zh_hint(raw_err)
                            lines = [
                                f"订单 id: {oid[:28]}…",
                                f"市场: {mt}",
                                f"方向: {oc or '—'}",
                                f"买卖: {side}",
                                f"计划撤单原因: {apply_result.decision_reason}",
                                "",
                                hint,
                                "",
                                "接口原文:",
                                raw_err[:1200],
                            ]
                            telegram.notify_operational_warning_zh(
                                title_zh="【警告】撤单失败",
                                lines=lines,
                                event_key=f"warn:canceled_fail:{oid}",
                            )
                    elif apply_result.outcome in (
                        "noop_missing_id",
                        "noop_unknown_action",
                        "replace_skip_bad_order",
                        "replace_skip_size",
                    ):
                        if telegram.enabled:
                            mt, oc = _resolve_order_display(
                                market_display, o, condition_id, token_id
                            )
                            labels = {
                                "noop_missing_id": "订单缺少 id，跳过处理",
                                "noop_unknown_action": "未知操作类型，跳过",
                                "replace_skip_bad_order": "改价跳过（订单缺 token 或 side）",
                                "replace_skip_size": "改价跳过（剩余份额为 0）",
                            }
                            telegram.notify_operational_warning_zh(
                                title_zh=f"【警告】{labels.get(apply_result.outcome, apply_result.outcome)}",
                                lines=[
                                    f"订单 id: {(oid or '')[:28]}…",
                                    f"市场: {mt}",
                                    f"方向: {oc or '—'}",
                                    f"outcome: {apply_result.outcome}",
                                ],
                                event_key=f"warn:{apply_result.outcome}:{oid or 'na'}",
                            )

            if (
                telegram.enabled
                and config.telegram_band_summary_enabled
                and config.telegram_band_summary_interval_sec > 0
                and now >= next_band_summary_at
            ):
                iv = max(1.0, float(config.telegram_band_summary_interval_sec))
                detail_lines: list[str] = []
                for row in sorted(
                    band_summary_rows, key=lambda r: str(r.get("oid") or "")
                ):
                    dist = abs(float(row["mid"]) - float(row["price"]))
                    dlt = max(float(row["delta"]), 1e-12)
                    ratio = dist / dlt
                    title, outcome = _resolve_order_display(
                        market_display,
                        row["o"],
                        row["condition_id"],
                        row["token_id"],
                    )
                    st = scoring_status_text(row["scoring"])
                    title_short = (title[:44] + "…") if len(title) > 45 else title
                    depth_block = ""
                    book = row.get("book")
                    if book is not None:
                        try:
                            dst = compute_eligible_band_depth_stats(
                                side=str(row["side"]),
                                order_price=float(row["price"]),
                                mid=float(row["mid"]),
                                delta=float(row["delta"]),
                                tick=float(row["tick"]),
                                bids=book.bids,
                                asks=book.asks,
                            )
                            depth_block = "\n" + format_eligible_band_depth_summary_zh(
                                dst
                            )
                        except Exception as e:
                            LOG.debug("band summary depth stats: %s", e)
                            depth_block = "\n  （带内深度统计失败）"
                    detail_lines.append(
                        f"· {title_short}\n"
                        f"  方向:{outcome or '—'} {row['side']} 挂单价={row['price']:.4f} mid={row['mid']:.4f}\n"
                        f"  |价−mid|/δ={ratio:.1%}（小数 {ratio:.3f}）计分:{st} 订单={str(row['oid'])[:18]}…"
                        f"{depth_block}"
                    )
                if not detail_lines and band_summary_eligible_n > 0:
                    detail_lines = [
                        "（有 eligible 挂单但本轮回无法取得 mid，无明细）"
                    ]
                elif not detail_lines and orders:
                    detail_lines = [
                        "（有开仓单但无 eligible：白名单外或该 outcome 有仓位已跳过）"
                    ]
                elif not detail_lines:
                    detail_lines = ["（当前无开仓单）"]
                bucket = int(now // iv)
                time_label = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(now)
                )
                telegram.notify_order_band_summary(
                    time_label=time_label,
                    interval_sec=iv,
                    lines=detail_lines,
                    time_bucket=bucket,
                )
                next_band_summary_at = now + iv
                LOG.info(
                    "Telegram band summary sent bucket=%s rows=%d",
                    bucket,
                    len(band_summary_rows),
                )

            error_streak = 0
        except KeyboardInterrupt:
            LOG.info("Interrupted; exiting.")
            break
        except Exception as e:
            error_streak += 1
            LOG.exception(
                "Main loop error (%d%s): %s",
                error_streak,
                f"/{config.max_api_errors_before_cancel_all}"
                if config.max_api_errors_before_cancel_all > 0
                else "",
                e,
            )
            if (
                config.max_api_errors_before_cancel_all > 0
                and error_streak >= config.max_api_errors_before_cancel_all
            ):
                try:
                    client.cancel_all()
                    LOG.critical("cancel_all() after repeated API failures.")
                except Exception:
                    LOG.exception("cancel_all failed")
                error_streak = 0

        time.sleep(config.loop_interval)


if __name__ == "__main__":
    main()
