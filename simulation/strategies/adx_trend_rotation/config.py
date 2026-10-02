"""
ADX 趋势强度轮动 — 模拟盘配置
"""

from __future__ import annotations
from pathlib import Path

from strategies.adx_trend_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, COMMISSION_RATE, SLIPPAGE, DB_PATH,
    MARKET_INDEX, MARKET_MA_PERIOD, ADX_MIN_STRENGTH, MIN_HOLD_DAYS,
)

# 模拟盘特有配置
INITIAL_CAPITAL = 10000
STRATEGY_NAME = "ADX趋势强度模拟盘"

# ADX策略用纯信号模式（ADX自身即过滤器）
RISK_MODE = "A"
MOMENTUM_WINDOW = 20  # 给 DailySimEngine 用（实际ADX策略不使用）

# ── 与回测引擎对齐的两处接线（2026-10-02 修复，详见 strategies/ALIGN_ALERT_FIX_20261002.md）──
# ① 持仓 ADX 得分归零 → 平仓。回测 _make_decision 第一条规定"得分=0 平仓"，
#    模拟盘此前一直是默认 False（只持有不退出）→ 回测 52% 交易日空仓、模拟盘却常满仓。
EXIT_WHEN_SIGNAL_DEAD = True
# ② 熊市开仓闸门：回测"regime==bear 且 目标得分<=0.5 → 不开仓"（仅限空仓开仓）。
#    ADX_MIN_STRENGTH 之上还有这层更严的门槛，模拟盘此前完全看不到 regime。
BEAR_OPEN_MIN_SCORE = 0.5

# 风控参数（RISK_MODE保持A，这些不生效但对DailySimEngine构造函数必填）
STOP_LOSS_PCT = 0.05
PROFIT_THRESHOLD = 0.10
DRAWBACK_PCT = 0.05
DRAWDOWN_THRESHOLD = 0.15
MIN_SWITCH_CONVICTION = 0.03

# 输出路径
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
