"""
风控模块

与 composite_momentum / ADX 保持一致的风控体系：
  1. 浮动止盈
  2. ATR 动态止损
  3. 极端回撤强制清仓

RSI 策略使用 Mode A（纯信号），此模块保留以保持架构一致性。
"""

from typing import Optional

import numpy as np

from .config import (
    STOP_PROFIT_THRESHOLD,
    STOP_PROFIT_DRAWBACK,
    ATR_MULTIPLIER,
    MAX_DRAWDOWN,
)


class RiskState:
    """风控状态追踪器。"""

    def __init__(self):
        self.entry_price: float = 0.0
        self.peak_price: float = 0.0
        self.trailing_active: bool = False
        self.peak_total_value: float = 0.0

    def on_open_position(self, entry_price: float):
        self.entry_price = entry_price
        self.peak_price = entry_price
        self.trailing_active = False

    def update_peak(self, current_high: float):
        if current_high > self.peak_price:
            self.peak_price = current_high

    def update_peak_total_value(self, total_value: float):
        if total_value > self.peak_total_value:
            self.peak_total_value = total_value


def check_stop_profit(
    risk_state: RiskState,
    current_low: float,
    floating_return: float,
) -> tuple[bool, str]:
    """浮动止盈检查。"""
    if not risk_state.trailing_active:
        if floating_return >= STOP_PROFIT_THRESHOLD:
            risk_state.trailing_active = True
            return False, "浮动止盈已激活"

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
) -> tuple[bool, str]:
    """ATR 动态止损。"""
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
) -> tuple[bool, str]:
    """极端回撤检查。"""
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
    mode: str = "B",
) -> tuple[str, str]:
    """
    全量风控检查。
    mode: "A"=无风控, "B"=全开, "C"=仅极端回撤
    """
    if mode == "A":
        return "none", ""

    # 1. 极端回撤
    triggered, reason = check_extreme_drawdown(risk_state, total_value)
    if triggered:
        return "extreme_drawdown", reason

    # 2. 止损 + 止盈（仅 B 模式）
    if mode == "B" and has_position and hold_symbol is not None:
        triggered, reason = check_stop_loss(
            risk_state.entry_price, current_low, current_atr
        )
        if triggered:
            return "stop_loss", reason

        floating_return = (current_close - risk_state.entry_price) / risk_state.entry_price
        triggered, reason = check_stop_profit(risk_state, current_low, floating_return)
        if triggered:
            return "stop_profit", reason

    return "none", ""
