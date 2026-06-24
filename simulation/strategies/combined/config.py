"""组合策略 — 模拟盘配置"""
from strategies.combined.config import (
    TOTAL_CAPITAL, MOMENTUM_PCT, PAIR_PCT,
)
from simulation.strategies.momentum_rotation.config import OUTPUT_DIR

STRATEGY_NAME = "组合策略模拟盘(80%动量+20%配对)"
STATE_FILE_DIR = OUTPUT_DIR
