"""
交易成本计算模块

与 momentum_rotation 保持一致的成本模型：
  佣金 + 滑点 + 冲击成本
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    COMMISSION_RATE,
    SLIPPAGE,
    TAX_RATE,
    MIN_COMMISSION,
    IMPACT_COST_COEF,
)


def compute_one_way_cost(
    trade_amount: float,
    amount_ma20: float,
) -> tuple[float, float, float, float]:
    """
    计算单次交易（买入或卖出）的总成本。

    Returns
    -------
    commission, slippage_cost, impact_cost, total
    """
    commission = max(trade_amount * COMMISSION_RATE, MIN_COMMISSION)
    slippage_cost = trade_amount * SLIPPAGE

    impact_cost = 0.0
    if amount_ma20 > 0 and trade_amount > 0:
        amount_ratio = trade_amount / amount_ma20
        impact_cost = trade_amount * (amount_ratio * IMPACT_COST_COEF)

    total = commission + slippage_cost + impact_cost
    return commission, slippage_cost, impact_cost, total


def compute_total_friction_cost(
    from_symbol: str,
    to_symbol: str,
    sell_amount: float,
    buy_amount: float,
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> tuple[float, dict]:
    """双边摩擦总成本。"""
    from_amount_ma20 = etf_data[from_symbol].loc[date_idx, "amount_ma20"]
    to_amount_ma20 = etf_data[to_symbol].loc[date_idx, "amount_ma20"]

    _, _, _, sell_cost = compute_one_way_cost(sell_amount, from_amount_ma20)
    _, _, _, buy_cost = compute_one_way_cost(buy_amount, to_amount_ma20)

    total_cost = sell_cost + buy_cost
    details = {
        "sell_cost": round(sell_cost, 2),
        "buy_cost": round(buy_cost, 2),
        "total_friction": round(total_cost, 2),
    }
    return total_cost, details
