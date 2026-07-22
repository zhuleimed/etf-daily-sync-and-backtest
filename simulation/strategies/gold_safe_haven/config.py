"""
黄金避险轮动策略 - 模拟盘配置

继承回测策略参数，添加模拟盘专用配置。
"""
from pathlib import Path

# 从回测策略导入共享参数
from strategies.gold_safe_haven.config import (
    ETF_POOL,
    ETF_SYMBOLS,
    BROAD_SYMBOLS,
    GOLD_SYMBOL,
    MOMENTUM_WINDOW,
    MIN_SWITCH_CONVICTION,
    MIN_HOLD_DAYS,
    COMMISSION_RATE,
    SLIPPAGE,
    DB_PATH,
    PANIC_THRESHOLD,
    MIN_GOLD_HOLD,
    GOLD_MAX_HOLD,
    GOLD_STOP_LOSS,
    PANIC_EXIT_THRESHOLD,
    ZSCORE_WINDOW,
)

# 模拟盘专用参数
INITIAL_CAPITAL = 10000
RISK_MODE = "A"              # 纯信号（恐慌切换自身即风控）
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15

STRATEGY_NAME = "黄金避险轮动模拟盘"
STRATEGY_ID = "gold_safe_haven"

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
