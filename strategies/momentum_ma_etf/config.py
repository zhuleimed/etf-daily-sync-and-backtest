"""
动量+均线过滤（逐ETF）轮动策略 - 配置模块

核心思路：
  每个ETF独立判断：close > MA(自身, N) → 可进入候选池
  close ≤ MA(自身, N) → 排除（不买入、已持有的卖出）
  候选池中按动量排名选最强持有。

与 momentum_ma_filter 的区别：
  - ma_filter 用大盘（沪深300）均线做统一开关
  - ma_etf 用每个ETF自身的均线做逐个筛选
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

# ── 均线过滤（每个ETF自身） ──
MA_FILTER_ENABLED = True
MA_FILTER_PERIOD = 60

# ── 动量 ──
MOMENTUM_WINDOW = 20
DYNAMIC_WINDOW_ENABLED = False
MOMENTUM_WINDOW_SHORT = 10
MOMENTUM_WINDOW_LONG = 20
WINDOW_SWITCH_THRESHOLD = 0.03
USE_RELATIVE_MOMENTUM = False
RELATIVE_MOMENTUM_FACTOR = 0.3
ETF_BENCHMARK_MAP = {
    "510050": "000016",
    "510300": "000300",
    "510500": "000905",
    "512100": "000852",
    "563000": "000852",
    "159915": "399006",
    "588000": "000688",
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
ADJUSTMENT_DAYS = 5
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
OUTPUT_DIR = "strategies/momentum_ma_etf/output"
