from __future__ import annotations

import logging
import time
from typing import Any, Optional

from passive_liquidity.config_manager import PassiveConfig
from passive_liquidity.fill_risk import build_fill_risk_context, long_window_count_only_activity
from passive_liquidity.http_utils import http_json
from passive_liquidity.models import FillRiskContext

LOG = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: PassiveConfig, user_address: str):
        self._config = config
        self._user = user_address

    def get_inventory(self, condition_id: str, token_id: str) -> float:
        url = (
            f"{self._config.data_api_host}/positions?"
            f"user={self._user}&market={condition_id}&limit=500"
        )
        try:
            rows = http_json("GET", url)
        except Exception as e:
            LOG.warning("positions API failed: %s", e)
            return 0.0
        if not isinstance(rows, list):
            return 0.0
        for p in rows:
            aid = str(p.get("asset") or "")
            if aid == str(token_id):
                return float(p.get("size") or 0)
        return 0.0

    def fetch_trades_for_token(self, client: Any, token_id: str) -> list[dict]:
        from py_clob_client.clob_types import TradeParams

        try:
            raw = client.get_trades(TradeParams(asset_id=token_id))
        except Exception as e:
            LOG.warning("get_trades failed: %s", e)
            return []
        return [t for t in raw if isinstance(t, dict)]

    def get_recent_fill_rate(self, client: Any, token_id: str) -> float:
        """
        Legacy: long lookback, trade count only, no direction (activity proxy [0, 1]).
        Prefer build_fill_risk_context for new logic.
        """
        trades = self.fetch_trades_for_token(client, token_id)
        return long_window_count_only_activity(
            trades,
            time.time(),
            float(self._config.fill_lookback_sec),
            self._config.fill_rate_denominator,
        )

    def build_fill_risk_context(
        self,
        client: Any,
        token_id: str,
        *,
        order_side: str,
        price: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        tick: float,
        trades: Optional[list[dict]] = None,
    ) -> FillRiskContext:
        tlist = trades if trades is not None else self.fetch_trades_for_token(client, token_id)
        return build_fill_risk_context(
            tlist,
            order_side=order_side,
            price=price,
            best_bid=best_bid,
            best_ask=best_ask,
            tick=tick,
            c=self._config,
        )

    def volatility_high(self, abs_one_day_change: float) -> bool:
        return abs_one_day_change >= self._config.volatility_abs_change_threshold
