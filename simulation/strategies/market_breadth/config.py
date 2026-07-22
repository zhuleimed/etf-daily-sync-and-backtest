"""
市场宽度择时策略 — 模拟盘配置

引用回测 config 的共享参数，追加模拟盘特有配置。
最优参数（#2 保守）：MA20 strong=70% weak=30% neutral=cash
"""

from __future__ import annotations

from pathlib import Path

# ── 引用回测参数 ──
from strategies.market_breadth.config import (
    ETF_POOL,
    ETF_SYMBOLS,
    MOMENTUM_WINDOW,       # 20
    MIN_SWITCH_CONVICTION,  # 3%
    MIN_HOLD_DAYS,          # 10
    COMMISSION_RATE,
    SLIPPAGE,
    DB_PATH,
    BREADTH_MA_PERIOD,      # 20
    BREADTH_STRONG,         # 0.70
    BREADTH_WEAK,           # 0.30
    NEUTRAL_MODE,           # "cash"
)

# ── 模拟盘特有配置 ──
INITIAL_CAPITAL = 10000
RISK_MODE = "A"                    # 宽度择时本身就是风控

# 风控参数（RISK_MODE=B 时生效）
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15

# 输出路径
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
