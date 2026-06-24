"""
风控模块

职责：
  1. 浮动止盈（Trailing Stop-Profit）：盈利达阈值的回撤保护
  2. ATR 动态止损：基于波动率的自适应止损
  3. 极端回撤强制风控：兜底机制

对齐策略原文 3.5 节：
  - 浮动止盈：持仓收益≥10%后启动，从最高点回撤≥5%时止盈
  - ATR止损：收盘价 < 持仓成本 - 2×ATR 时止损
  - 极端回撤：账户从最高点回撤≥15%时强制清仓

所有风控信号优先级高于调仓信号。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    STOP_PROFIT_THRESHOLD,
    STOP_PROFIT_DRAWBACK,
    ATR_MULTIPLIER,
    MAX_DRAWDOWN,
)


class RiskState:
    """风控状态追踪器。

    在回测引擎的逐日循环中维护，记录每笔持仓的入场价与最高价。
    """

    def __init__(self):
        # 当前持仓的成本价
        self.entry_price: float = 0.0
        # 持仓期间的最高价（用于浮动止盈）
        self.peak_price: float = 0.0
        # 浮动止盈是否已激活（盈利达到阈值后为True）
        self.trailing_active: bool = False

        # 账户级别
        self.peak_total_value: float = 0.0

    def on_open_position(self, entry_price: float):
        """开仓/买入时重置风控状态。"""
        self.entry_price = entry_price
        self.peak_price = entry_price
        self.trailing_active = False

    def update_peak(self, current_high: float):
        """更新持仓期间最高价。"""
        if current_high > self.peak_price:
            self.peak_price = current_high

    def update_peak_total_value(self, total_value: float):
        """更新账户最高总资产。"""
        if total_value > self.peak_total_value:
            self.peak_total_value = total_value


# ── 风控检查函数 ──


def check_stop_profit(
    risk_state: RiskState,
    current_low: float,
    floating_return: float,
) -> Tuple[bool, str]:
    """
    浮动止盈检查。

    1. 如果浮动收益率 >= STOP_PROFIT_THRESHOLD（默认10%），激活浮动止盈
    2. 激活后，如果日内最低价从最高点回撤 >= STOP_PROFIT_DRAWBACK（5%），触发止盈

    使用 daily low 而非 close 做回撤检测，更贴近实盘：日内价格一旦
    触及止盈线即执行，而非等到收盘才判断。

    Returns
    -------
    triggered : bool
        是否触发止盈
    reason : str
        触发原因（用于日志）
    """
    # 更新最高价（在调用前外部已更新）

    # 检查是否应激活浮动止盈
    if not risk_state.trailing_active:
        if floating_return >= STOP_PROFIT_THRESHOLD:
            risk_state.trailing_active = True
            return False, "浮动止盈已激活"

    # 激活后检查回撤（用 low 价格：日内最低点回撤达阈值即触发）
    if risk_state.trailing_active and risk_state.peak_price > 0:
        drawdown = (risk_state.peak_price - current_low) / risk_state.peak_price
        if drawdown >= STOP_PROFIT_DRAWBACK:
            return True, (
                f"浮动止盈触发：从最高价{risk_state.peak_price:.4f}"
                f"回撤{drawdown:.2%}≥{STOP_PROFIT_DRAWBACK:.0%}"
            )

    return False, ""


def check_stop_loss(
    entry_price: float,
    current_low: float,
    current_atr: float,
) -> Tuple[bool, str]:
    """
    ATR 动态止损检查。

    止损线 = 持仓成本价 - ATR_MULTIPLIER × 当日ATR值
    日内最低价跌破止损线时触发止损。

    使用 daily low 而非 close：实盘中若日内价格曾触及止损线，
    止损单即被激活（即使收盘收回），回测应反映这一情况。

    Returns
    -------
    triggered : bool
    reason : str
    """
    if entry_price <= 0 or current_atr <= 0:
        return False, ""

    stop_line = entry_price - ATR_MULTIPLIER * current_atr
    if current_low < stop_line:
        return True, (
            f"ATR止损触发：最低价{current_low:.4f} < "
            f"止损线{stop_line:.4f}（成本{entry_price:.4f} - "
            f"{ATR_MULTIPLIER}×ATR{current_atr:.4f}）"
        )

    return False, ""


def check_extreme_drawdown(
    risk_state: RiskState,
    current_total_value: float,
) -> Tuple[bool, str]:
    """
    极端回撤检查。

    账户总资产从最高点回撤 >= MAX_DRAWDOWN（15%）时强制清仓。

    Returns
    -------
    triggered : bool
    reason : str
    """
    if risk_state.peak_total_value <= 0:
        return False, ""

    drawdown_ratio = (
        risk_state.peak_total_value - current_total_value
    ) / risk_state.peak_total_value

    if drawdown_ratio >= MAX_DRAWDOWN:
        return True, (
            f"极端回撤触发：总资产从最高{risk_state.peak_total_value:.2f}"
            f"回撤{drawdown_ratio:.2%}≥{MAX_DRAWDOWN:.0%}"
        )

    return False, ""


def run_all_risk_checks(
    risk_state: RiskState,
    total_value: float,
    has_position: bool,
    hold_symbol: Optional[str],
    current_high: float,
    current_low: float,
    current_close: float,
    current_atr: float,
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    mode: str = "B",
) -> Tuple[str, str]:
    """
    全量风控检查（按优先级：极端回撤 > 止损 > 止盈）。

    止损和止盈均使用 daily low 做触发判断，更贴近实盘：
    日内价格一旦触及条件即触发，而非等到收盘。

    mode 控制检查范围:
      "B" = 全开（止损 + 止盈 + 极端回撤）
      "C" = 仅极端回撤（兜底风控）
      "A" 应在调用层就已跳过，不会到达此函数

    Parameters
    ----------
    risk_state : RiskState
        风控状态
    total_value : float
        当前总资产
    has_position : bool
        是否持有仓位
    hold_symbol : str or None
        持仓ETF代码
    current_high : float
        当日最高价（用于更新峰值）
    current_low : float
        当日最低价（用于止损/止盈检测）
    current_close : float
        当日收盘价（用于判断是否达到收益阈值）
    current_atr : float
        当日ATR
    etf_data : dict
        ETF数据
    date_idx : int
        当前日期索引
    mode : str
        "B"=全开, "C"=仅极端回撤

    Returns
    -------
    action : str
        "none" / "stop_loss" / "stop_profit" / "extreme_drawdown"
    reason : str
        触发原因描述
    """
    # 1. 极端回撤（所有风控模式都检查）
    triggered, reason = check_extreme_drawdown(risk_state, total_value)
    if triggered:
        return "extreme_drawdown", reason

    # 2. 止损 + 止盈（仅 B 模式检查）
    if mode == "B" and has_position and hold_symbol is not None:
        triggered, reason = check_stop_loss(
            risk_state.entry_price, current_low, current_atr
        )
        if triggered:
            return "stop_loss", reason

        # 止盈检查
        #    激活阈值用 close（避免日内毛刺误激活）
        #    回撤检测用 low（更早触发，贴近实盘）
        floating_return = (current_close - risk_state.entry_price) / risk_state.entry_price
        triggered, reason = check_stop_profit(risk_state, current_low, floating_return)
        if triggered:
            return "stop_profit", reason

    return "none", ""
