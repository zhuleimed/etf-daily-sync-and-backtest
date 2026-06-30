"""风控模块"""
from .config import STOP_PROFIT_THRESHOLD, STOP_PROFIT_DRAWBACK, ATR_MULTIPLIER, MAX_DRAWDOWN


class RiskState:
    def __init__(self):
        self.entry_price = 0.0; self.peak_price = 0.0; self.trailing_active = False; self.peak_total_value = 0.0
    def on_open_position(self, ep): self.entry_price = ep; self.peak_price = ep; self.trailing_active = False
    def update_peak(self, h):
        if h > self.peak_price: self.peak_price = h
    def update_peak_total_value(self, v):
        if v > self.peak_total_value: self.peak_total_value = v


def check_stop_profit(rs, cl, fr):
    if not rs.trailing_active and fr >= STOP_PROFIT_THRESHOLD:
        rs.trailing_active = True; return False, "浮动止盈已激活"
    if rs.trailing_active and rs.peak_price > 0:
        dd = (rs.peak_price - cl) / rs.peak_price
        if dd >= STOP_PROFIT_DRAWBACK: return True, f"浮动止盈触发:最高{rs.peak_price:.4f}回撤{dd:.2%}"
    return False, ""

def check_stop_loss(ep, cl, ca):
    if ep <= 0 or ca <= 0: return False, ""
    sl = ep - ATR_MULTIPLIER * ca
    if cl < sl: return True, f"ATR止损:{cl:.4f}<止损线{sl:.4f}"
    return False, ""

def check_extreme_drawdown(rs, cv):
    if rs.peak_total_value <= 0: return False, ""
    dd = (rs.peak_total_value - cv) / rs.peak_total_value
    if dd >= MAX_DRAWDOWN: return True, f"极端回撤:最高{rs.peak_total_value:.2f}回撤{dd:.2%}"
    return False, ""

def run_all_risk_checks(rs, tv, hp, hs, ch, cl, cc, ca, mode="B"):
    if mode == "A": return "none", ""
    t, r = check_extreme_drawdown(rs, tv)
    if t: return "extreme_drawdown", r
    if mode == "B" and hp and hs:
        t, r = check_stop_loss(rs.entry_price, cl, ca)
        if t: return "stop_loss", r
        t, r = check_stop_profit(rs, cl, (cc - rs.entry_price) / rs.entry_price)
        if t: return "stop_profit", r
    return "none", ""
