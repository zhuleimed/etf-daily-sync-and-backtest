"""
风控检查模块

在 momentum_rotation 风控基础上增加黄金止损检查。
"""
from typing import Optional

from .config import GOLD_STOP_LOSS, MAX_DRAWDOWN


def check_gold_stop_loss(
    gold_ret_5d: float,
) -> tuple:
    """
    检查黄金自身是否需要止损。

    当黄金5日跌幅超过 GOLD_STOP_LOSS（默认-5%）时触发，
    卖出黄金回到现金，避免黄金自身暴跌时死扛。

    Parameters
    ----------
    gold_ret_5d : float
        黄金ETF的5日收益率

    Returns
    -------
    triggered : bool
    reason : str
    """
    if gold_ret_5d <= GOLD_STOP_LOSS:
        return True, f"黄金止损: 5日跌幅={gold_ret_5d*100:.1f}% <= {GOLD_STOP_LOSS*100:.0f}%"
    return False, ""


def check_extreme_drawdown(
    peak_value: float,
    current_value: float,
    threshold: float = MAX_DRAWDOWN,
) -> tuple:
    """
    极端回撤检查：总资产从峰值回撤超过阈值则清仓。

    Parameters
    ----------
    peak_value : float
        历史峰值总资产
    current_value : float
        当前总资产
    threshold : float
        回撤阈值（默认15%）

    Returns
    -------
    triggered : bool
    reason : str
    """
    if peak_value <= 0:
        return False, ""
    dd = (current_value - peak_value) / peak_value
    if dd <= -threshold:
        return True, f"极端回撤触发: 峰值{peak_value:.2f}→当前{current_value:.2f}，回撤{abs(dd)*100:.2f}%>={threshold*100:.0f}%"
    return False, ""
