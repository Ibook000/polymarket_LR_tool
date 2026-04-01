"""
Polygon on-chain USDC deposit totals for the Polymarket funder address (Polygonscan API).

When POLYGON_USDC_DEPOSIT_FROM_ALLOWLIST is unset, any inbound USDC (per contract)
to the funder is counted — approximate. With an allowlist, only transfers from those
addresses count as deposits (exact mode for Polymarket bridge/vault senders).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

from passive_liquidity.http_utils import http_json

LOG = logging.getLogger(__name__)

POLYGONSCAN_API = "https://api.polygonscan.com/api"

# USDC.e (PoS bridged) on Polygon — common for older Polymarket flows
DEFAULT_USDC_CONTRACT = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


@dataclass(frozen=True)
class PolygonDepositSummary:
    total_usdc: float
    deposit_count: int
    latest_deposit_unix: Optional[int]
    approximate: bool
    note_zh: str


def _parse_allowlist(raw: str) -> Optional[frozenset[str]]:
    if not raw.strip():
        return None
    return frozenset(x.strip().lower() for x in raw.split(",") if x.strip())


def fetch_polygon_usdc_deposit_summary(funder_address: str) -> Optional[PolygonDepositSummary]:
    """
    Sum qualifying ERC-20 transfers into funder_address via Polygonscan tokentx.

    Returns None if API key missing or API error (caller should fall back to env / startup).
    """
    api_key = os.environ.get("POLYGONSCAN_API_KEY", "").strip()
    if not api_key:
        LOG.debug("POLYGONSCAN_API_KEY unset; skip on-chain deposit fetch")
        return None

    funder = (funder_address or "").strip()
    if not funder:
        return None

    contract = (
        os.environ.get("POLYGON_USDC_CONTRACT", "").strip() or DEFAULT_USDC_CONTRACT
    )
    allow = _parse_allowlist(os.environ.get("POLYGON_USDC_DEPOSIT_FROM_ALLOWLIST", ""))
    approximate = allow is None
    if approximate:
        note_zh = (
            "链上统计为近似值：已将指向本 funder 地址的 USDC 入账转账累加，"
            "未按 Polymarket 专属金库/桥地址过滤。"
        )
    else:
        note_zh = (
            "链上统计：仅计入来自 POLYGON_USDC_DEPOSIT_FROM_ALLOWLIST 所列地址的 USDC 转入。"
        )

    # hash -> list of (amount_usdc, unix_ts)
    by_hash: dict[str, list[tuple[float, int]]] = {}
    page = 0
    max_pages = int(os.environ.get("POLYGONSCAN_TOKENTX_MAX_PAGES", "25") or "25")
    offset = min(10_000, int(os.environ.get("POLYGONSCAN_TOKENTX_OFFSET", "1000") or "1000"))

    while page < max_pages:
        page += 1
        qs = urlencode(
            {
                "module": "account",
                "action": "tokentx",
                "address": funder,
                "contractaddress": contract,
                "page": str(page),
                "offset": str(offset),
                "sort": "asc",
                "apikey": api_key,
            }
        )
        url = f"{POLYGONSCAN_API}?{qs}"
        try:
            raw = http_json("GET", url, timeout=60.0)
        except Exception as e:
            LOG.warning("Polygonscan tokentx request failed: %s", e)
            return None

        if not isinstance(raw, dict):
            LOG.warning("Polygonscan unexpected response type: %s", type(raw))
            return None

        result = raw.get("result")
        if isinstance(result, str):
            LOG.warning("Polygonscan error: %s", result[:400])
            return None
        if not isinstance(result, list):
            LOG.warning("Polygonscan unexpected result type: %s", type(result).__name__)
            return None

        if len(result) == 0:
            break

        status = str(raw.get("status") or "")
        if status != "1":
            LOG.warning(
                "Polygonscan tokentx status=%s message=%s (page=%s)",
                status,
                raw.get("message"),
                page,
            )
            return None

        for row in result:
            if not isinstance(row, dict):
                continue
            to_a = str(row.get("to") or "").strip().lower()
            if to_a != funder.lower():
                continue
            from_a = str(row.get("from") or "").strip().lower()
            if allow is not None and from_a not in allow:
                continue
            try:
                dec = int(row.get("tokenDecimal") or 6)
            except (TypeError, ValueError):
                dec = 6
            try:
                v = row.get("value")
                val_raw = int(v) if v is not None else 0
            except (TypeError, ValueError):
                try:
                    val_raw = int(float(v))
                except (TypeError, ValueError):
                    continue
            amt = val_raw / float(10**dec)
            if amt <= 0:
                continue
            h = str(row.get("hash") or "").strip().lower()
            if not h:
                continue
            try:
                ts = int(row.get("timeStamp") or 0)
            except (TypeError, ValueError):
                ts = 0
            by_hash.setdefault(h, []).append((amt, ts))

        if len(result) < offset:
            break

    if not by_hash:
        return PolygonDepositSummary(
            total_usdc=0.0,
            deposit_count=0,
            latest_deposit_unix=None,
            approximate=approximate,
            note_zh=note_zh,
        )

    total = 0.0
    latest: Optional[int] = None
    for txs in by_hash.values():
        for amt, ts in txs:
            total += amt
            if ts > 0 and (latest is None or ts > latest):
                latest = ts

    n_dep = len(by_hash)
    LOG.info(
        "On-chain USDC deposits: total=%.4f USDC count=%d latest_ts=%s approximate=%s",
        total,
        n_dep,
        latest,
        approximate,
    )
    return PolygonDepositSummary(
        total_usdc=total,
        deposit_count=n_dep,
        latest_deposit_unix=latest,
        approximate=approximate,
        note_zh=note_zh,
    )
