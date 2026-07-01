"""
自适应轮动 — 模拟盘配置
"""

from __future__ import annotations
from pathlib import Path

from strategies.adaptive_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, COMMISSION_RATE, SLIPPAGE, DB_PATH,
    MARKET_INDEX, MOM_WINDOW, MOM_SWITCH_CONVICTION,
)

INITIAL_CAPITAL = 10000
STRATEGY_NAME = "自适应轮动模拟盘"

RISK_MODE = "A"
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15
MIN_SWITCH_CONVICTION = 0.03
MIN_HOLD_DAYS = 5

# DailySimEngine 兼容参数
MOMENTUM_WINDOW = MOM_WINDOW              # 供 daily.py 导入

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
