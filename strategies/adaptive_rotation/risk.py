"""风控模块"""
from typing import Optional
from .config import STOP_PROFIT_THRESHOLD, STOP_PROFIT_DRAWBACK, ATR_MULTIPLIER, MAX_DRAWDOWN


class RiskState:
    def __init__(self):
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.trailing_active = False
        self.peak_total_value = 0.0

    def on_open_position(self, entry_price):
        self.entry_price = entry_price
        self.peak_price = entry_price
        self.trailing_active = False

    def update_peak(self, current_high):
        if current_high > self.peak_price:
            self.peak_price = current_high

    def update_peak_total_value(self, total_value):
        if total_value > self.peak_total_value:
            self.peak_total_value = total_value


def run_all_risk_checks(risk_state, total_value, has_position, hold_symbol, current_high, current_low, current_close, current_atr, mode="B"):
    if mode == "A":
        return "none", ""
    triggered, reason = _extreme_drawdown(risk_state, total_value)
    if triggered:
        return "extreme_drawdown", reason
    if mode == "B" and has_position and hold_symbol:
        triggered, reason = _stop_loss(risk_state.entry_price, current_low, current_atr)
        if triggered:
            return "stop_loss", reason
        floating_return = (current_close - risk_state.entry_price) / risk_state.entry_price
        triggered, reason = _stop_profit(risk_state, current_low, floating_return)
        if triggered:
            return "stop_profit", reason
    return "none", ""


def _stop_profit(risk_state, current_low, floating_return):
    if not risk_state.trailing_active:
        if floating_return >= STOP_PROFIT_THRESHOLD:
            risk_state.trailing_active = True
            return False, ""
    if risk_state.trailing_active and risk_state.peak_price > 0:
        drawdown = (risk_state.peak_price - current_low) / risk_state.peak_price
        if drawdown >= STOP_PROFIT_DRAWBACK:
            return True, f"浮动止盈触发"
    return False, ""


def _stop_loss(entry_price, current_low, current_atr):
    if entry_price <= 0 or current_atr <= 0:
        return False, ""
    stop_line = entry_price - ATR_MULTIPLIER * current_atr
    if current_low < stop_line:
        return True, f"ATR止损触发"
    return False, ""


def _extreme_drawdown(risk_state, current_total_value):
    if risk_state.peak_total_value <= 0:
        return False, ""
    dd = (risk_state.peak_total_value - current_total_value) / risk_state.peak_total_value
    if dd >= MAX_DRAWDOWN:
        return True, f"极端回撤触发"
    return False, ""
