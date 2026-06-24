"""
波动率过滤轮动策略 - 配置模块

核心思路：
  用沪深300的历史波动率判断市场状态：
  - 波动率低（市场平稳）→ 正常动量轮动
  - 波动率高（市场动荡）→ 空仓避险

动量策略在低波动趋势市表现最好，高波动震荡市容易反复打脸。
"""

ETF_POOL = {
    "510050": "上证50ETF（华夏）",
    "510300": "沪深300ETF（华泰柏瑞）",
    "510500": "中证500ETF（南方）",
    "512100": "中证1000ETF（南方）",
    "563000": "中证2000ETF（华夏）",
    "159915": "创业板ETF（易方达）",
    "588000": "科创50ETF（华夏）",
}
ETF_SYMBOLS = list(ETF_POOL.keys())
INITIAL_CAPITAL = 10000
START_DATE = "2024-01-01"
END_DATE = ""

# ── 波动率过滤（核心！） ──
VOL_FILTER_ENABLED = True           # 是否启用
VOL_WINDOW = 20                     # 波动率计算窗口（交易日）
VOL_THRESHOLD = 0.3                # 年化波动率阈值：>30%时空仓
VOL_BENCHMARK = "000300"            # 用沪深300衡量市场波动

# ── 动量 ──
MOMENTUM_WINDOW = 20
DYNAMIC_WINDOW_ENABLED = False
MOMENTUM_WINDOW_SHORT = 10
MOMENTUM_WINDOW_LONG = 20
WINDOW_SWITCH_THRESHOLD = 0.03
USE_RELATIVE_MOMENTUM = False
RELATIVE_MOMENTUM_FACTOR = 0.3
ETF_BENCHMARK_MAP = {
    "510050": "000016", "510300": "000300", "510500": "000905",
    "512100": "000852", "563000": "000852", "159915": "399006", "588000": "000688",
}
SHORT_TERM_MOMENTUM_CHECK = True
MIN_SWITCH_CONVICTION = 0.03
MIN_HOLD_DAYS = 10
TOP_N = 1

# ── 费用 ──
COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
TAX_RATE = 0.0
MIN_COMMISSION = 0.0
ADJUSTMENT_DAYS = 3
IMPACT_COST_COEF = 0.1

# ── 风控 ──
RISK_MODE = "A"
STOP_PROFIT_THRESHOLD = 0.10
STOP_PROFIT_DRAWBACK = 0.05
ATR_PERIOD = 20
ATR_MULTIPLIER = 4.0
MAX_DRAWDOWN = 0.15

# ── 路径 ──
DB_PATH = "data/etf_daily.db"
BENCHMARK_SYMBOL = "000300"
OUTPUT_DIR = "strategies/momentum_vol_filter/output"
