"""
MACD 趋势轮动 — 模拟盘配置
"""

from __future__ import annotations
from pathlib import Path

from strategies.macd_trend_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, COMMISSION_RATE, SLIPPAGE, DB_PATH,
    MARKET_INDEX, MIN_HOLD_DAYS,
)

INITIAL_CAPITAL = 10000
STRATEGY_NAME = "MACD趋势轮动模拟盘"
RISK_MODE = "B"                # MACD策略需风控全开
MOMENTUM_WINDOW = 20           # 给DailySimEngine用（实际MACD不使用）
# 开仓/切换目标需连续 N 日保持 top 才发（平滑单日假 leader）。
# 修复性回测(2026-09-05) 表明 N=2 对 macd 双窗口非负优化: focal -7.58→-3.18,
# full -10.17→-7.04。设 1=关闭(旧行为)，2=启用(本次促销)。
CONFIRM_DAYS = 2
MIN_SWITCH_CONVICTION = 0.03
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
