"""
HS300均线择时策略 — 模拟盘配置

HS300 > MA10 → 正常动量轮动
HS300 < MA10 → 空仓避险
"""

from __future__ import annotations
from pathlib import Path

from strategies.hs300_ma_timing.config import (
    ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, MA_PERIOD,
    MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, DB_PATH,
)

INITIAL_CAPITAL = 10000
RISK_MODE = "A"

STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
