"""
RSI 趋势确认轮动 — 模拟盘配置
"""

from __future__ import annotations
from pathlib import Path

from strategies.rsi_trend_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, COMMISSION_RATE, SLIPPAGE, DB_PATH,
    MARKET_INDEX, MARKET_MA_PERIOD, RSI_BULL_THRESHOLD, MIN_HOLD_DAYS,
)

# 模拟盘特有配置
INITIAL_CAPITAL = 10000
STRATEGY_NAME = "RSI趋势确认模拟盘"

# RSI策略用纯信号模式（RSI>50本身即过滤器）
RISK_MODE = "A"
MOMENTUM_WINDOW = 20  # 给 DailySimEngine 用（实际RSI策略不使用）

# 风控参数（RISK_MODE保持A，这些不生效但对DailySimEngine构造函数必填）
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15
MIN_SWITCH_CONVICTION = 1.0        # RSI v2评分范围0-4.5，需合理阈值

# 输出路径
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
