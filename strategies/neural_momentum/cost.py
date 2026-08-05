"""
摩擦成本校验模块

职责：
  1. 计算单只 ETF 的交易成本（佣金+滑点+冲击成本）
  2. 计算双边摩擦总成本（卖出旧标的 + 买入新标的）
  3. 校验预期超额收益是否大于摩擦成本

对齐策略原文 3.3 节：
  - 单边成本 = 交易佣金 + 交易滑点 + 冲击成本
  - 冲击成本 = 调仓金额 / 标的近20日日均成交额 × 冲击系数
  - 双边成本 = 旧标的单边成本 + 新标的单边成本
  - 触发调仓条件：预期超额收益 > 双边总摩擦成本
"""

from typing import Dict, Optional, Tuple

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
    price: float,
) -> Tuple[float, float, float, float]:
    """
    计算单次交易（买入或卖出）的总成本。

    Parameters
    ----------
    trade_amount : float
        交易金额（元）
    amount_ma20 : float
        标的近20日日均成交额（元）
    price : float
        交易价格

    Returns
    -------
    commission : float
        佣金（默认万二，无最低5元限制）
    slippage_cost : float
        滑点成本（默认万分之一）
    impact_cost : float
        冲击成本（按流动性动态计算）
    total : float
        总成本（三者之和）
    """
    # 1. 佣金
    commission = max(trade_amount * COMMISSION_RATE, MIN_COMMISSION)

    # 2. 滑点（价差成本）
    slippage_cost = trade_amount * SLIPPAGE

    # 3. 冲击成本（与流动性成反比）
    if amount_ma20 > 0 and trade_amount > 0:
        # 调仓金额占日均成交额比例
        amount_ratio = trade_amount / amount_ma20
        impact_cost = trade_amount * (amount_ratio * IMPACT_COST_COEF)
    else:
        impact_cost = 0.0

    total = commission + slippage_cost + impact_cost
    return commission, slippage_cost, impact_cost, total


def compute_total_friction_cost(
    from_symbol: str,
    to_symbol: str,
    sell_amount: float,
    buy_amount: float,
    from_price: float,
    to_price: float,
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
) -> Tuple[float, Dict]:
    """
    计算双边摩擦总成本。

    Parameters
    ----------
    from_symbol : str
        当前持仓 ETF（卖出方）
    to_symbol : str
        目标 ETF（买入方）
    sell_amount : float
        卖出金额
    buy_amount : float
        买入金额
    from_price : float
        卖出价格
    to_price : float
        买入价格
    etf_data : dict
        {symbol: DataFrame}
    date_idx : int
        当前日期索引

    Returns
    -------
    total_cost : float
        双边总摩擦成本（元）
    details : dict
        成本明细，用于日志
    """
    # 获取20日均成交额
    from_amount_ma20 = etf_data[from_symbol].loc[date_idx, "amount_ma20"]
    to_amount_ma20 = etf_data[to_symbol].loc[date_idx, "amount_ma20"]

    # 卖出旧标的的成本
    _, _, _, sell_cost = compute_one_way_cost(sell_amount, from_amount_ma20, from_price)

    # 买入新标的的成本
    _, _, _, buy_cost = compute_one_way_cost(buy_amount, to_amount_ma20, to_price)

    total_cost = sell_cost + buy_cost

    details = {
        "sell_cost": round(sell_cost, 2),
        "buy_cost": round(buy_cost, 2),
        "total_friction": round(total_cost, 2),
        "sell_amount_ma20": round(from_amount_ma20, 2),
        "buy_amount_ma20": round(to_amount_ma20, 2),
    }

    return total_cost, details


def friction_cost_ratio(total_friction: float, total_trade_amount: float) -> float:
    """
    摩擦成本占交易金额的比例（用于显示和校验）。
    """
    if total_trade_amount <= 0:
        return 0.0
    return total_friction / total_trade_amount
