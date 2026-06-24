"""
双均线交叉轮动策略 - 配置模块

核心逻辑：
  每只ETF独立判断：快线上穿慢线→上升趋势→持有
                  快线下穿慢线→下降趋势→卖出
  所有上升趋势的ETF等权持有，无符合条件的全仓现金。

与动量策略的区别：
  - 动量策略选"过去涨得最好的"（排名制）
  - 均线交叉选"现在在上升趋势中的"（判断制）
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

# ── 双均线 ──
FAST_MA_PERIOD = 10        # 快线
MOMENTUM_WINDOW = 20       # 给 data.py 的动量列计算用（本策略不使用）
SLOW_MA_PERIOD = 60        # 慢线（慢线>快线即下降趋势，卖出）
MAX_HOLD_ETFS = 5          # 最多同时持有几只（0=不限制）

# ── 费用 ──
COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
TAX_RATE = 0.0
MIN_COMMISSION = 0.0
ADJUSTMENT_DAYS = 3
IMPACT_COST_COEF = 0.1

# ── 风控 ──
RISK_MODE = "A"

# ── 路径 ──
DB_PATH = "data/etf_daily.db"
BENCHMARK_SYMBOL = "000300"
OUTPUT_DIR = "strategies/dual_ma_crossover/output"
