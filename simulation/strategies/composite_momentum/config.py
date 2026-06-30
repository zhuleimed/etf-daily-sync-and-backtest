"""
复合动量轮动策略 — 模拟盘配置

引用回测策略的参数，追加模拟盘特有配置。
"""

from __future__ import annotations

from pathlib import Path

# ── 引用回测参数 ──
from strategies.composite_momentum.config import (
    ETF_POOL,
    ETF_SYMBOLS,
    COMMISSION_RATE,
    SLIPPAGE,
    DB_PATH,
    MARKET_INDEX,
    MARKET_MA_PERIOD,
    BEAR_THRESHOLD,
    MIN_HOLD_DAYS,
)

# ── 多因子权重（与回测一致） ──
from strategies.composite_momentum.config import (
    FACTOR_WEIGHT_TREND,
    FACTOR_WEIGHT_SHARPE,
    FACTOR_WEIGHT_QUALITY,
    FACTOR_WEIGHT_VOLUME,
)

# ── 模拟盘特有配置 ──
INITIAL_CAPITAL = 10000
STRATEGY_NAME = "复合动量模拟盘"

# 风控参数（DailySimEngine 使用）
RISK_MODE = "B"                    # 风控全开（回测显示B模式最佳）
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15

# 动量窗口（给 DailySimEngine 的信号函数用）
MOMENTUM_WINDOW = 20

# 切换参数
MIN_SWITCH_CONVICTION = 0.03       # 切换置信度

# 输出路径
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
