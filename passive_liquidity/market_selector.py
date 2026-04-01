"""
Reward-market catalog helper (list / paginate markets with incentives).

Not used by the default monitoring main loop in ``main_loop.py``; kept for
optional tooling or custom scripts.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.http_utils import http_json
from passive_liquidity.models import RewardMarketToken

LOG = logging.getLogger(__name__)


class MarketSelector:
    """Discover reward-active markets from CLOB `/rewards/markets/multi`."""

    def __init__(self, config: PassiveConfig):
        self._config = config

    def _rows_to_legs(self, rows: list[dict[str, Any]]) -> list[RewardMarketToken]:
        flattened: list[RewardMarketToken] = []
        for m in rows:
            rate = 0.0
            cfgs = m.get("rewards_config") or []
            if cfgs:
                rate = float(cfgs[0].get("rate_per_day") or 0)
            tokens = m.get("tokens") or []
            if not self._config.quote_all_outcome_tokens and tokens:
                tokens = tokens[:1]
            for t in tokens:
                flattened.append(
                    RewardMarketToken(
                        condition_id=str(m["condition_id"]),
                        token_id=str(t["token_id"]),
                        outcome=str(t.get("outcome") or ""),
                        question=str(m.get("question") or ""),
                        rewards_max_spread=float(m.get("rewards_max_spread") or 0),
                        rewards_min_size=float(m.get("rewards_min_size") or 0),
                        market_id=str(m.get("market_id") or ""),
                        volume_24hr=float(m.get("volume_24hr") or 0),
                        spread=float(m.get("spread") or 0),
                        one_day_price_change=float(m.get("one_day_price_change") or 0),
                        rate_per_day=rate,
                    )
                )
        flattened.sort(key=lambda x: x.rate_per_day, reverse=True)
        return flattened

    def list_all_quotable_legs(self) -> list[RewardMarketToken]:
        """
        Paginate through all reward markets until the API reports no more pages.
        Each row is one quotable outcome (token); binary markets may appear twice if
        quote_all_outcome_tokens is enabled.
        """
        base = f"{self._config.clob_host}/rewards/markets/multi"
        cursor: str | None = None
        rows: list[dict[str, Any]] = []
        while True:
            q: dict[str, Any] = {
                "page_size": 500,
                "order_by": "rate_per_day",
                "position": "DESC",
            }
            if cursor:
                q["next_cursor"] = cursor
            url = f"{base}?{urlencode(q)}"
            page = http_json("GET", url)
            chunk = page.get("data") or []
            rows.extend(chunk)
            cursor = page.get("next_cursor")
            if not cursor or cursor == "LTE=":
                break

        legs = self._rows_to_legs(rows)
        LOG.info("MarketSelector: loaded %d reward markets -> %d quotable legs", len(rows), len(legs))
        return legs

    def get_reward_markets(self) -> list[RewardMarketToken]:
        """Legacy: capped list for automated selection (not used by interactive main loop)."""
        base = f"{self._config.clob_host}/rewards/markets/multi"
        cursor: str | None = None
        rows: list[dict[str, Any]] = []
        while True:
            q: dict[str, Any] = {"page_size": min(500, max(1, self._config.max_markets * 2))}
            if cursor:
                q["next_cursor"] = cursor
            q["order_by"] = "rate_per_day"
            q["position"] = "DESC"
            url = f"{base}?{urlencode(q)}"
            page = http_json("GET", url)
            chunk = page.get("data") or []
            rows.extend(chunk)
            cursor = page.get("next_cursor")
            if not cursor or cursor == "LTE=" or len(rows) >= self._config.max_markets * 4:
                break

        flattened = self._rows_to_legs(rows)
        out = flattened[: self._config.max_quote_tasks]
        LOG.info("MarketSelector: %d quotable outcome legs (cap=%d)", len(out), self._config.max_quote_tasks)
        return out
