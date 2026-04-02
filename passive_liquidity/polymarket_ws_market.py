"""
Background Polymarket CLOB market-channel WebSocket (book, trades, tick).

Updates PolymarketWsHub only; does not execute trading logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable

import websockets

from passive_liquidity.polymarket_ws_state import PolymarketWsHub

LOG = logging.getLogger(__name__)

MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL_SEC = 10.0
BACKOFF_INITIAL = 2.0
BACKOFF_MAX = 60.0


def _parse_messages(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _dispatch_market(hub: PolymarketWsHub, msg: dict) -> None:
    et = str(msg.get("event_type") or "").lower()
    if et == "book":
        hub.market_apply_book(msg)
    elif et == "price_change":
        hub.market_apply_price_change(msg)
    elif et == "tick_size_change":
        hub.market_apply_tick_size_change(msg)
    elif et == "last_trade_price":
        hub.market_apply_last_trade_price(msg)
    elif et == "best_bid_ask":
        hub.market_apply_best_bid_ask(msg)


class PolymarketMarketWsThread(threading.Thread):
    def __init__(
        self,
        hub: PolymarketWsHub,
        *,
        get_asset_ids: Callable[[], list[str]],
        name: str = "polymarket-ws-market",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._hub = hub
        self._get_asset_ids = get_asset_ids
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            asyncio.run(self._runner())
        except Exception:
            LOG.exception("market ws asyncio.run exited")

    async def _runner(self) -> None:
        backoff = BACKOFF_INITIAL
        while not self._stop.is_set():
            assets = [str(x) for x in self._get_asset_ids() if str(x).strip()]
            if not assets:
                LOG.info("market ws: no asset ids yet; sleep 5s")
                await asyncio.sleep(5.0)
                continue
            sub = {
                "type": "market",
                "assets_ids": assets,
                "custom_feature_enabled": True,
            }
            try:
                LOG.info(
                    "market ws connecting assets=%d url=%s",
                    len(assets),
                    MARKET_WS_URL,
                )
                async with websockets.connect(
                    MARKET_WS_URL,
                    ping_interval=None,
                    close_timeout=5,
                ) as ws:
                    self._hub.market_set_error("")
                    await ws.send(json.dumps(sub))
                    self._hub.market_mark_subscription_ok(True)
                    self._hub.market_set_connected(True)
                    LOG.info(
                        "market ws connected subscription_sent assets=%d",
                        len(assets),
                    )
                    backoff = BACKOFF_INITIAL
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            if self._stop.is_set():
                                break
                            if isinstance(raw, bytes):
                                raw = raw.decode("utf-8", errors="replace")
                            if not isinstance(raw, str):
                                continue
                            s = raw.strip()
                            if s.upper() == "PONG":
                                continue
                            if s.upper() == "PING":
                                await ws.send("PONG")
                                continue
                            for msg in _parse_messages(raw):
                                _dispatch_market(self._hub, msg)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._hub.market_mark_subscription_ok(False)
                self._hub.market_set_connected(False)
                self._hub.market_set_error(str(e))
                LOG.warning("market ws error: %s", e)
            finally:
                self._hub.market_set_connected(False)
                LOG.warning(
                    "market ws disconnected; reconnect in %.1fs (stop=%s)",
                    backoff,
                    self._stop.is_set(),
                )
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(BACKOFF_MAX, backoff * 1.5)

    async def _ping_loop(self, ws) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(PING_INTERVAL_SEC)
                await ws.send("PING")
        except asyncio.CancelledError:
            return
        except Exception as e:
            LOG.debug("market ws ping failed: %s", e)
