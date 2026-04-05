"""Regression: custom coarse N → BUY targets (N=1 mid, N=2 one tick below, …)."""

from __future__ import annotations

import unittest

from passive_liquidity.simple_price_policy import (
    CustomPricingSettings,
    _decide_custom_coarse,
    _ticks_from_mid_into_band,
)


class TestCustomCoarseOffset(unittest.TestCase):
    def test_ticks_from_mid_grid_stable(self) -> None:
        self.assertEqual(_ticks_from_mid_into_band("BUY", 0.16, 0.15, 0.01), 1)
        self.assertEqual(_ticks_from_mid_into_band("BUY", 0.16, 0.16, 0.01), 0)

    def test_n1_targets_aligned_mid_buy(self) -> None:
        settings = CustomPricingSettings(
            coarse_tick_offset_from_mid=1,
            coarse_allow_top_of_book=True,
            coarse_min_candidate_levels=1,
            fine_safe_band_min=0.4,
            fine_safe_band_max=0.6,
            fine_target_band_ratio=0.5,
        )
        meta: dict = {}
        d, m = _decide_custom_coarse(
            side_u="BUY",
            price=0.20,
            mid=0.16,
            tick=0.01,
            delta=0.05,
            bids=[],
            asks=[],
            min_replace_ticks=1,
            settings=settings,
            best_bid=0.10,
            best_ask=0.20,
            meta=meta,
        )
        self.assertEqual(d.action, "replace")
        self.assertAlmostEqual(float(d.new_price or 0), 0.16, places=6)
        self.assertEqual(m.get("custom_coarse_tick_offset_effective"), 0)

    def test_n_through_n4_buy_ladder(self) -> None:
        for n, want in [(4, 0.13), (3, 0.14), (2, 0.15), (1, 0.16)]:
            settings = CustomPricingSettings(
                coarse_tick_offset_from_mid=n,
                coarse_allow_top_of_book=True,
                coarse_min_candidate_levels=1,
                fine_safe_band_min=0.4,
                fine_safe_band_max=0.6,
                fine_target_band_ratio=0.5,
            )
            d, m = _decide_custom_coarse(
                side_u="BUY",
                price=0.20,
                mid=0.16,
                tick=0.01,
                delta=0.05,
                bids=[],
                asks=[],
                min_replace_ticks=1,
                settings=settings,
                best_bid=0.10,
                best_ask=0.20,
                meta={},
            )
            self.assertEqual(
                d.action,
                "replace",
                msg=f"N={n} reason={m.get('reason_code')}",
            )
            self.assertAlmostEqual(float(d.new_price or 0), want, places=6, msg=f"N={n}")


if __name__ == "__main__":
    unittest.main()
