"""pair_trading 纯多头风格轮动 — 模拟盘配置"""
from strategies.pair_trading.config import (
    PAIRS, INITIAL_CAPITAL, ZSCORE_PERIOD, ZSCORE_OPEN,
    ZSCORE_CLOSE, ZSCORE_STOP, COMMISSION_RATE, SLIPPAGE, DB_PATH,
)
from simulation.strategies.momentum_rotation.config import OUTPUT_DIR

STRATEGY_NAME = "配对交易风格轮动模拟盘"
STATE_FILE_DIR = OUTPUT_DIR
