"""
绩效指标计算模块

直接复用 momentum_rotation 的指标计算逻辑。
"""
from strategies.momentum_rotation.metrics import (
    BacktestMetrics,
    MetricsCalculator,
    TRADING_DAYS_PER_YEAR,
)
