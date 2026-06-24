"""
动量轮动策略 — 模拟盘配置

引用回测 config 的共享参数，追加模拟盘特有配置。
"""

from __future__ import annotations

import os
from pathlib import Path

# ── 引用回测参数 ──
from strategies.momentum_rotation.config import (
    ETF_POOL,
    ETF_SYMBOLS,
    MOMENTUM_WINDOW,
    MIN_SWITCH_CONVICTION,
    MIN_HOLD_DAYS,
    COMMISSION_RATE,
    SLIPPAGE,
    DB_PATH,
)

# ── 模拟盘特有配置 ──
INITIAL_CAPITAL = 10000            # 初始资金
RISK_MODE = "A"                    # A=纯信号, B=全风控, C=仅极端回撤

# 风控参数（RISK_MODE=B 时生效）
STOP_LOSS_PCT = 0.05               # 止损比例 5%
PROFIT_THRESHOLD = 0.10            # 止盈进入阈值 10%
DRAWBACK_PCT = 0.05                # 移动止盈回撤 5%
DRAWDOWN_THRESHOLD = 0.15          # 极端回撤 15%

# 输出路径
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
