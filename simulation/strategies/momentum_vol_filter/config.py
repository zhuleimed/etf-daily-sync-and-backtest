"""momentum_vol_filter 波动率过滤 — 模拟盘配置"""
from strategies.momentum_vol_filter.config import (
    ETF_POOL, ETF_SYMBOLS, INITIAL_CAPITAL, MOMENTUM_WINDOW,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, VOL_THRESHOLD, VOL_WINDOW,
    SHORT_TERM_MOMENTUM_CHECK,   # 短期动量确认（2026-10-02 接线，见 daily.py 切换闸门）
)
from simulation.strategies.momentum_rotation.config import (
    MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    RISK_MODE, STOP_LOSS_PCT, PROFIT_THRESHOLD, DRAWBACK_PCT,
    DRAWDOWN_THRESHOLD, OUTPUT_DIR,
)

STRATEGY_NAME = "波动率过滤轮动模拟盘"
STATE_FILE_DIR = OUTPUT_DIR
