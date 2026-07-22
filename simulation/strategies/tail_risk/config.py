"""尾部风险轮动 模拟盘配置"""
from __future__ import annotations
from pathlib import Path
from strategies.tail_risk.config import ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, COMMISSION_RATE, SLIPPAGE, DB_PATH
try: from strategies.tail_risk.config import MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS, RISK_MODE
except: MIN_SWITCH_CONVICTION=0.03; MIN_HOLD_DAYS=10; RISK_MODE="A"
INITIAL_CAPITAL = 10000
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
STATE_FILE_DIR = OUTPUT_DIR
