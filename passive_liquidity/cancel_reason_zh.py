"""
Map AdjustmentEngine cancel `reason` strings to Chinese labels for Telegram.
"""

from __future__ import annotations

from typing import Tuple

# (撤单类别标题, 可读说明；保留英文 reason 供排查)
_REASON_TABLE: dict[str, Tuple[str, str]] = {
    "inventory_at_max_long_no_more_bids": (
        "因库存 / 人工风控撤单",
        "多头库存已达上限，撤销买单",
    ),
    "inventory_at_max_short_no_more_asks": (
        "因库存 / 人工风控撤单",
        "空头库存已达上限，撤销卖单",
    ),
    "buy_above_mid": (
        "因价格无效或偏离激励带撤单（相对中间价）",
        "买价高于中间价，不符合策略，已撤单",
    ),
    "sell_below_mid": (
        "因价格无效或偏离激励带撤单（相对中间价）",
        "卖价低于中间价，不符合策略，已撤单",
    ),
    "buy_far_below_reward_band": (
        "因价格无效或偏离激励带撤单",
        "买价远低于激励带，已撤单",
    ),
    "sell_far_above_reward_band": (
        "因价格无效或偏离激励带撤单",
        "卖价远高于激励带，已撤单",
    ),
}


def cancel_category_zh(decision_reason: str) -> Tuple[str, str, str]:
    """
    Returns (类别, 说明, 原始 reason).

    类别对齐需求：
    - 跟随中间价调价：引擎对纯撤单较少用；若将来有 mid 跟踪撤单可在此扩展。
    - 防御性撤单（成交风险）：引擎多用 replace；撤单路径可在此扩展 widen_* 等。
    - 库存 / 人工：库存上限撤单。
    - 价格无效或越带：mid / 激励带相关撤单。
    - 其他：兜底。
    """
    key = (decision_reason or "").strip()
    if key in _REASON_TABLE:
        cat, desc = _REASON_TABLE[key]
        return (cat, desc, key)
    if key.startswith("widen_") or "fill_pressure" in key or "fill" in key.lower():
        return (
            "因成交风险防御性撤单",
            "与成交密度或防御性风控相关的撤单（或系统扩展原因）",
            key,
        )
    if "manual" in key.lower() or "inventory" in key.lower():
        return (
            "因库存 / 人工风控撤单",
            "库存或手动相关撤单",
            key,
        )
    if "mid" in key.lower() or "band" in key.lower() or "nudge" in key.lower():
        return (
            "因跟随中间价 / 激励带调价撤单",
            "价格、中间价或激励带相关撤单",
            key,
        )
    return (
        "其他系统原因撤单",
        "未分类的策略或系统撤单",
        key,
    )
