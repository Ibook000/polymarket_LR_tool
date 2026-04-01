from __future__ import annotations

import logging
from typing import Any, Optional

from passive_liquidity.models import OrderBookSnapshot

LOG = logging.getLogger(__name__)


def _level_price(level: Any) -> Optional[float]:
    if level is None:
        return None
    p = getattr(level, "price", None)
    if p is None and isinstance(level, dict):
        p = level.get("price")
    if p is None or p == "":
        return None
    return float(p)


def _best_bid_from_levels(bids: list[Any]) -> Optional[float]:
    """Highest buy price; API does not guarantee bids[0] is the best bid."""
    prices = [_level_price(b) for b in bids]
    prices = [x for x in prices if x is not None]
    return max(prices) if prices else None


def _best_ask_from_levels(asks: list[Any]) -> Optional[float]:
    """Lowest sell price; API does not guarantee asks[0] is the best ask."""
    prices = [_level_price(a) for a in asks]
    prices = [x for x in prices if x is not None]
    return min(prices) if prices else None


def second_best_bid_from_levels(bids: list[Any]) -> Optional[float]:
    """Second-highest distinct bid (买二); None if fewer than two price levels."""
    prices = sorted({_level_price(b) for b in bids if _level_price(b) is not None}, reverse=True)
    return prices[1] if len(prices) >= 2 else None


def second_best_ask_from_levels(asks: list[Any]) -> Optional[float]:
    """Second-lowest distinct ask (卖二); None if fewer than two price levels."""
    prices = sorted({_level_price(a) for a in asks if _level_price(a) is not None})
    return prices[1] if len(prices) >= 2 else None


class OrderBookFetcher:
    def __init__(self, clob_ro: Any):
        """
        clob_ro: ClobClient with at least get_order_book (can be read-only / L2).
        """
        self._client = clob_ro

    def get_orderbook(self, token_id: str) -> OrderBookSnapshot:
        book = self._client.get_order_book(token_id)
        bids = getattr(book, "bids", None) or []
        asks = getattr(book, "asks", None) or []
        bb = _best_bid_from_levels(bids)
        ba = _best_ask_from_levels(asks)
        tick = float(getattr(book, "tick_size", None) or "0.01")
        nr = bool(getattr(book, "neg_risk", False))
        snap = OrderBookSnapshot(
            best_bid=bb,
            best_ask=ba,
            tick_size=tick,
            neg_risk=nr,
            bids=bids,
            asks=asks,
            raw=book,
        )
        return snap

    def mid_price(self, token_id: str) -> Optional[float]:
        book = self.get_orderbook(token_id)
        if book.mid is not None:
            return book.mid
        mp = self._client.get_midpoint(token_id)
        if isinstance(mp, dict):
            raw = mp.get("mid")
            if raw is None or raw == "":
                LOG.warning("get_midpoint missing mid for token_id=%s…", token_id[:24])
                return None
            return float(raw)
        if mp is None:
            return None
        return float(mp)
