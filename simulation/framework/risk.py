"""
模拟盘风控模块

提供通用的止损/止盈/极端回撤检查，由各策略 config.py 决定启用与否及参数阈值。

逻辑复用自回测 strategies/momentum_rotation/risk.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RiskResult:
    triggered: bool = False
    reason: str = ""
    action: str = ""      # "stop_loss" | "stop_profit" | "extreme_drawdown"


def check_stop_loss(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float = 0.05,
) -> RiskResult:
    """固定比例止损：当前价格 < 成本价 × (1 - 止损比例)。

    Args:
        entry_price: 持仓成本价。
        current_price: 当前价。
        stop_loss_pct: 止损比例（默认5%）。

    Returns:
        RiskResult。
    """
    if entry_price <= 0:
        return RiskResult()
    threshold = entry_price * (1 - stop_loss_pct)
    if current_price <= threshold:
        return RiskResult(
            triggered=True,
            reason=f"止损触发: 成本{entry_price:.4f}×{(1-stop_loss_pct):.2f}={threshold:.4f}，当前{current_price:.4f}",
            action="stop_loss",
        )
    return RiskResult()


def check_trailing_stop(
    highest_price: float,
    current_price: float,
    drawback_pct: float = 0.05,
) -> RiskResult:
    """移动止盈：自持仓期间最高点回撤超过比例时触发。

    Args:
        highest_price: 持仓期间最高价。
        current_price: 当前价。
        drawback_pct: 回撤比例（默认5%）。

    Returns:
        RiskResult。
    """
    if highest_price <= 0 or current_price <= 0:
        return RiskResult()
    drawback = highest_price * (1 - drawback_pct)
    if current_price <= drawback:
        return RiskResult(
            triggered=True,
            reason=f"移动止盈触发: 最高{highest_price:.4f}×(1-{drawback_pct:.0%})={drawback:.4f}，当前{current_price:.4f}",
            action="stop_profit",
        )
    return RiskResult()


def check_enter_stop_profit(
    entry_price: float,
    current_price: float,
    profit_threshold: float = 0.10,
) -> bool:
    """判断是否已进入止盈区间（收益率达到阈值后才启用移动止盈）。"""
    if entry_price <= 0:
        return False
    return (current_price / entry_price - 1) >= profit_threshold


def check_extreme_drawdown(
    peak_value: float,
    current_value: float,
    drawdown_threshold: float = 0.15,
) -> RiskResult:
    """极端回撤检查：总资产从最高点回撤超过比例时强制清仓。

    Args:
        peak_value: 历史最高总资产。
        current_value: 当前总资产。
        drawdown_threshold: 回撤阈值（默认15%）。

    Returns:
        RiskResult。
    """
    if peak_value <= 0:
        return RiskResult()
    dd = (peak_value - current_value) / peak_value
    if dd >= drawdown_threshold:
        return RiskResult(
            triggered=True,
            reason=f"极端回撤触发: 峰值{peak_value:.2f}→当前{current_value:.2f}，回撤{dd:.2%}≥{drawdown_threshold:.0%}",
            action="extreme_drawdown",
        )
    return RiskResult()


def check_atr_stop_loss(
    entry_price: float,
    current_price: float,
    atr_value: float,
    atr_multiplier: float = 4.0,
) -> RiskResult:
    """ATR 动态止损：止损线 = 成本价 - N × ATR。"""
    if entry_price <= 0 or atr_value <= 0:
        return RiskResult()
    stop_line = entry_price - atr_multiplier * atr_value
    if current_price <= stop_line:
        return RiskResult(
            triggered=True,
            reason=f"ATR止损触发: 成本{entry_price:.4f}-{atr_multiplier}×{atr_value:.4f}={stop_line:.4f}，当前{current_price:.4f}",
            action="stop_loss",
        )
    return RiskResult()


def run_all_risk_checks(
    state,
    current_price: float,
    current_value: float,
    mode: str = "A",
    # 止损参数
    stop_loss_pct: float = 0.05,
    # 移动止盈参数
    profit_threshold: float = 0.10,
    drawback_pct: float = 0.05,
    # ATR 止损参数
    atr_value: float = 0.0,
    atr_multiplier: float = 4.0,
    # 极端回撤参数
    drawdown_threshold: float = 0.15,
    peak_value: float = 0.0,
) -> Optional[RiskResult]:
    """按模式运行所有风控检查。

    Args:
        state: SimState 对象（含持仓信息）。
        current_price: 当日 ETF 收盘价。
        current_value: 当日总资产（现金+持仓市值）。
        mode: 风控模式
            "A" = 关闭（纯信号）
            "B" = 全开（止损+止盈+极端回撤）
            "C" = 仅极端回撤
        ...其他参数为各种阈值。

    Returns:
        触发时返回 RiskResult，否则 None。
    """
    if mode == "A":
        return None

    # C 和 B 都检查极端回撤
    if mode in ("B", "C"):
        result = check_extreme_drawdown(peak_value, current_value, drawdown_threshold)
        if result.triggered:
            return result

    if mode == "C":
        return None

    # B 模式：检查止损和止盈
    if state.position.shares <= 0:
        return None

    # T+1 保护：今日开仓不触发止损止盈
    if state.position.today_opened:
        return None

    entry_price = state.position.avg_cost
    highest = state.position.highest_price

    # 止损
    result = check_stop_loss(entry_price, current_price, stop_loss_pct)
    if result.triggered:
        return result

    # 移动止盈：先判断是否进入了止盈区间
    if check_enter_stop_profit(entry_price, current_price, profit_threshold):
        result = check_trailing_stop(highest, current_price, drawback_pct)
        if result.triggered:
            return result

    # ATR 止损（如有 ATR 数据）
    if atr_value > 0:
        result = check_atr_stop_loss(entry_price, current_price, atr_value, atr_multiplier)
        if result.triggered:
            return result

    return None
