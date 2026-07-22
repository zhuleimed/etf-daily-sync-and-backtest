"""
动量信号计算模块

直接复用 momentum_rotation 的动量信号逻辑。
正常模式下的ETF排名与纯动量轮动完全一致。
"""
# 直接从 momentum_rotation 导出，避免代码重复
from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
    compute_momentum_spread,
    get_active_momentum_column,
    compute_momentum_signals_dynamic,
    determine_signals,
)
