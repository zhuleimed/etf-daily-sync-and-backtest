"""
相对强度持久性策略 - 配置模块

核心思路：与动量轮动不同，不看"涨了多少"（绝对动量），而看
"持续跑赢同类的比例"（相对强度持续性）。

逻辑：
  1. 每日计算每只ETF日收益率 vs 7只ETF等权平均
  2. 统计过去N天中跑赢同类的天数占比 = 持续性得分
  3. 持续性 > 阈值 + 绝对动量 > 0 → 参与排名
  4. 排名 = 绝对动量（经过持续性过滤后）

这个策略区分"因为市场好而涨"(高动量低持续性) vs "真的有alpha"(高持续性)。
"""

ETF_POOL = {
    "510050": "上证50ETF", "510300": "沪深300ETF",
    "510500": "中证500ETF", "512100": "中证1000ETF",
    "563000": "中证2000ETF", "159915": "创业板ETF",
    "588000": "科创50ETF",
}
ETF_SYMBOLS = list(ETF_POOL.keys())

INITIAL_CAPITAL = 10000
START_DATE = "2024-01-01"
END_DATE = ""

# ============================================================================
# 相对强度参数（核心！）
# ============================================================================
RELATIVE_WINDOW = 20           # 持续性统计窗口（日）
MIN_PERSISTENCE = 0.60         # 最小持续性阈值：跑赢同类>=60%的日子
                               # 0.5=一半以上日子跑赢, 0.6=60%的日子跑赢, 0.7=70%

# ============================================================================
# 动量信号
# ============================================================================
MOMENTUM_WINDOW = 20
DYNAMIC_WINDOW_ENABLED = False
TOP_N = 1

# 交易费用
COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
TAX_RATE = 0.0
MIN_COMMISSION = 0.0
IMPACT_COST_COEF = 0.1
ADJUSTMENT_DAYS = 5

# 切换控制
MIN_SWITCH_CONVICTION = 0.03
SHORT_TERM_MOMENTUM_CHECK = True
MIN_HOLD_DAYS = 10

# 风控
STOP_PROFIT_THRESHOLD = 0.10
STOP_PROFIT_DRAWBACK = 0.05
ATR_PERIOD = 20
ATR_MULTIPLIER = 4.0
MAX_DRAWDOWN = 0.15
RISK_MODE = "A"

DB_PATH = "data/etf_daily.db"
BENCHMARK_SYMBOL = "000300"
OUTPUT_DIR = "strategies/relative_strength/output"
