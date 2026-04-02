"""
Background Polymarket CLOB user-channel WebSocket (orders + trades).

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

USER_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
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


def _handle_user_payload(hub: PolymarketWsHub, msg: dict) -> None:
    et = str(msg.get("event_type") or "").lower()
    top = str(msg.get("type") or "").upper()
    if et == "trade" or top == "TRADE":
        hub.user_apply_trade_message(msg)
        return
    if et == "order" or top in ("PLACEMENT", "UPDATE", "CANCELLATION"):
        hub.user_apply_order_message(msg)
        return


class PolymarketUserWsThread(threading.Thread):
    """
    Daemon thread running asyncio user WebSocket with reconnect.
    """

    def __init__(
        self,
        hub: PolymarketWsHub,
        *,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        get_markets: Callable[[], list[str]],
        name: str = "polymarket-ws-user",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._hub = hub
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._get_markets = get_markets
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            asyncio.run(self._runner())
        except Exception:
            LOG.exception("user ws asyncio.run exited")

    async def _runner(self) -> None:
        backoff = BACKOFF_INITIAL
        while not self._stop.is_set():
            markets = [str(m) for m in self._get_markets() if str(m).strip()]
            if not markets:
                await asyncio.sleep(3.0)
                continue
            sub = {
                "type": "user",
                "auth": {
                    "apiKey": self._api_key,
                    "secret": self._api_secret,
                    "passphrase": self._api_passphrase,
                },
                "markets": markets,
            }
            try:
                LOG.info(
                    "user ws connecting markets=%d url=%s",
                    len(markets),
                    USER_WS_URL,
                )
                async with websockets.connect(
                    USER_WS_URL,
                    ping_interval=None,
                    close_timeout=5,
                ) as ws:
                    self._hub.user_set_error("")
                    await ws.send(json.dumps(sub))
                    self._hub.user_mark_subscription_ok(True)
                    self._hub.user_set_connected(True)
                    LOG.info(
                        "user ws connected subscription_sent markets=%d",
                        len(markets),
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
                                _handle_user_payload(self._hub, msg)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._hub.user_mark_subscription_ok(False)
                self._hub.user_set_connected(False)
                self._hub.user_set_error(str(e))
                LOG.warning("user ws error: %s", e)
            finally:
                self._hub.user_set_connected(False)
                LOG.warning(
                    "user ws disconnected; reconnect in %.1fs (stop=%s)",
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
            LOG.debug("user ws ping failed: %s", e)
