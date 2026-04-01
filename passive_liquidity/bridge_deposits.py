"""
Polymarket Bridge API — completed deposits targeting Polygon USDC.

GET https://bridge.polymarket.com/status/{address}

The `address` is often the deposit flow address from bridge responses; for many EVM
funder/proxy wallets it still returns useful rows. On failure, callers fall back
to other deposit reference sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from passive_liquidity.http_utils import http_json

LOG = logging.getLogger(__name__)

BRIDGE_STATUS_URL = "https://bridge.polymarket.com/status/{address}"

# Polygon PoS; bridge schema uses string chain ids
POLYGON_CHAIN_IDS = frozenset({"137", "0x89"})

# Destination USDC on Polygon (USDC.e common)
DEFAULT_POLYGON_USDC_LOWER = frozenset(
    {
        "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # USDC.e
        "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",  # native USDC
    }
)


@dataclass(frozen=True)
class BridgeDepositSummary:
    total_usdc: float
    deposit_count: int
    latest_created_ms: Optional[int]


def _polygon_usdc_contracts_lower() -> frozenset[str]:
    import os

    raw = os.environ.get("BRIDGE_POLYGON_USDC_CONTRACTS", "").strip()
    if not raw:
        return DEFAULT_POLYGON_USDC_LOWER
    out = set(DEFAULT_POLYGON_USDC_LOWER)
    for p in raw.split(","):
        x = p.strip().lower()
        if x.startswith("0x") and len(x) == 42:
            out.add(x)
    return frozenset(out)


def fetch_bridge_polygon_usdc_deposits(address: str) -> Optional[BridgeDepositSummary]:
    """
    Sum COMPLETED bridge transactions that target Polygon USDC.

    Returns None on HTTP/payload errors. Returns zero totals only when the API
    succeeded and had no matching rows (caller may still prefer another source).
    """
    addr = (address or "").strip()
    if not addr:
        return None
    url = BRIDGE_STATUS_URL.format(address=addr)
    try:
        raw = http_json("GET", url, timeout=45.0)
    except Exception as e:
        LOG.debug("Bridge /status request failed for %s…: %s", addr[:12], e)
        return None

    if not isinstance(raw, dict):
        return None
    if raw.get("error"):
        LOG.debug("Bridge /status error: %s", raw.get("error"))
        return None

    txs = raw.get("transactions")
    if not isinstance(txs, list):
        return None

    contracts = _polygon_usdc_contracts_lower()
    by_tx: dict[str, float] = {}
    latest_ms: Optional[int] = None

    for t in txs:
        if not isinstance(t, dict):
            continue
        status = str(t.get("status") or "").upper()
        if status != "COMPLETED":
            continue
        chain = str(t.get("toChainId") or "").strip()
        if chain not in POLYGON_CHAIN_IDS:
            continue
        tok = str(t.get("toTokenAddress") or "").strip().lower()
        if tok not in contracts:
            continue
        raw_amt = t.get("fromAmountBaseUnit")
        try:
            amt_base = int(str(raw_amt).strip())
        except (TypeError, ValueError):
            continue
        # Bridge payloads use 6 decimals for USDC-sized amounts in examples
        amt_usdc = amt_base / 1_000_000.0
        if amt_usdc <= 0:
            continue
        th = str(t.get("txHash") or t.get("transactionHash") or "").strip().lower()
        key = th if th else f"row:{id(t)}"
        by_tx[key] = by_tx.get(key, 0.0) + amt_usdc
        ctm = t.get("createdTimeMs")
        try:
            ms = int(float(ctm)) if ctm is not None else 0
        except (TypeError, ValueError):
            ms = 0
        if ms > 0 and (latest_ms is None or ms > latest_ms):
            latest_ms = ms

    total = sum(by_tx.values())
    n = len(by_tx)
    if n == 0:
        LOG.debug(
            "Bridge /status: %d tx rows, none matched Polygon+USDC+COMPLETED for %s…",
            len(txs),
            addr[:12],
        )
        return None

    LOG.info(
        "Bridge deposit summary: %.4f USDC from %d completed tx (latest_ms=%s)",
        total,
        n,
        latest_ms,
    )
    return BridgeDepositSummary(
        total_usdc=total, deposit_count=n, latest_created_ms=latest_ms
    )
