"""
Resolve CLOB `condition_id` / `token_id` to human-readable market titles via Gamma API.

Polymarket stores market copy in the language used on the site (typically English);
this is not machine-translated to Chinese.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from passive_liquidity.http_utils import http_json

LOG = logging.getLogger(__name__)


def _parse_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except json.JSONDecodeError:
            pass
    return []


def _outcome_for_token(market: dict, token_id: str) -> str:
    tokens = _parse_str_list(market.get("clobTokenIds"))
    outcomes = _parse_str_list(market.get("outcomes"))
    if not token_id or not tokens or not outcomes or len(tokens) != len(outcomes):
        return ""
    try:
        i = tokens.index(str(token_id))
        return outcomes[i].strip()
    except ValueError:
        return ""


class MarketDisplayResolver:
    """
    Cache Gamma lookups so Telegram / logs show question text instead of 0x… hashes.
    """

    def __init__(self, gamma_api_host: str) -> None:
        self._host = gamma_api_host.rstrip("/")
        self._by_token: dict[str, tuple[str, str]] = {}
        self._by_condition: dict[str, str] = {}
        self._miss_token: set[str] = set()

    def lookup(self, condition_id: str, token_id: str) -> tuple[str, str]:
        """
        Returns (question, outcome_label). Either field may be empty if unknown.
        """
        tid = str(token_id or "").strip()
        cid = str(condition_id or "").strip()

        if tid and tid in self._by_token:
            return self._by_token[tid]

        if tid and tid not in self._miss_token:
            rows = self._fetch_markets(f"{self._host}/markets?clob_token_ids={tid}")
            if rows:
                m = rows[0]
                q = str(m.get("question") or "").strip()
                oc = _outcome_for_token(m, tid)
                self._by_token[tid] = (q, oc)
                if q and cid and cid not in self._by_condition:
                    self._by_condition[cid] = q
                return (q, oc)
            self._miss_token.add(tid)

        if cid and cid in self._by_condition:
            return (self._by_condition[cid], "")

        if cid:
            rows = self._fetch_markets(f"{self._host}/markets?condition_ids={cid}")
            if rows:
                m = rows[0]
                q = str(m.get("question") or "").strip()
                if q:
                    self._by_condition[cid] = q
                oc = _outcome_for_token(m, tid) if tid else ""
                if tid and q:
                    self._by_token[tid] = (q, oc)
                return (q, oc)

        return ("", "")

    def _fetch_markets(self, url: str) -> list[dict]:
        try:
            data = http_json("GET", url)
        except Exception as e:
            LOG.debug("Gamma market lookup failed %s: %s", url[:80], e)
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
