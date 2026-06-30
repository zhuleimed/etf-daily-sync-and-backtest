"""风控模块"""
from typing import Optional
import numpy as np
from .config import STOP_PROFIT_THRESHOLD, STOP_PROFIT_DRAWBACK, ATR_MULTIPLIER, MAX_DRAWDOWN


class RiskState:
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


def check_stop_profit(risk_state: RiskState, current_low: float, floating_return: float) -> tuple[bool, str]:
    if not risk_state.trailing_active and floating_return >= STOP_PROFIT_THRESHOLD:
        risk_state.trailing_active = True
        return False, "浮动止盈已激活"
    if risk_state.trailing_active and risk_state.peak_price > 0:
        dd = (risk_state.peak_price - current_low) / risk_state.peak_price
        if dd >= STOP_PROFIT_DRAWBACK:
            return True, f"浮动止盈触发：最高{risk_state.peak_price:.4f}回撤{dd:.2%}"
    return False, ""


def check_stop_loss(entry_price: float, current_low: float, current_atr: float) -> tuple[bool, str]:
    if entry_price <= 0 or current_atr <= 0:
        return False, ""
    stop_line = entry_price - ATR_MULTIPLIER * current_atr
    if current_low < stop_line:
        return True, f"ATR止损触发：{current_low:.4f}<止损线{stop_line:.4f}"
    return False, ""


def check_extreme_drawdown(risk_state: RiskState, current_total_value: float) -> tuple[bool, str]:
    if risk_state.peak_total_value <= 0:
        return False, ""
    dd = (risk_state.peak_total_value - current_total_value) / risk_state.peak_total_value
    if dd >= MAX_DRAWDOWN:
        return True, f"极端回撤触发：最高{risk_state.peak_total_value:.2f}回撤{dd:.2%}"
    return False, ""


def run_all_risk_checks(risk_state: RiskState, total_value: float, has_position: bool,
                         hold_symbol: Optional[str], current_high: float, current_low: float,
                         current_close: float, current_atr: float, mode: str = "B") -> tuple[str, str]:
    if mode == "A":
        return "none", ""
    triggered, reason = check_extreme_drawdown(risk_state, total_value)
    if triggered:
        return "extreme_drawdown", reason
    if mode == "B" and has_position and hold_symbol:
        triggered, reason = check_stop_loss(risk_state.entry_price, current_low, current_atr)
        if triggered:
            return "stop_loss", reason
        fr = (current_close - risk_state.entry_price) / risk_state.entry_price
        triggered, reason = check_stop_profit(risk_state, current_low, fr)
        if triggered:
            return "stop_profit", reason
    return "none", ""
