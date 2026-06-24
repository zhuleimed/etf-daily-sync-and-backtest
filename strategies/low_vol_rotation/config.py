"""
低波动率轮动策略 - 配置模块

核心逻辑：持有波动率最低的ETF，低波动 = 更稳定的收益。
"""

ETF_POOL = {
    "510050": "上证50ETF（华夏）", "510300": "沪深300ETF（华泰柏瑞）",
    "510500": "中证500ETF（南方）", "512100": "中证1000ETF（南方）",
    "563000": "中证2000ETF（华夏）", "159915": "创业板ETF（易方达）",
    "588000": "科创50ETF（华夏）",
}
ETF_SYMBOLS = list(ETF_POOL.keys())
INITIAL_CAPITAL = 10000
START_DATE = "2024-01-01"
END_DATE = ""

VOL_WINDOW = 20
TOP_N = 1

# dummy
MOMENTUM_WINDOW = 20
COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
TAX_RATE = 0.0
ADJUSTMENT_DAYS = 5
RISK_MODE = "A"
DB_PATH = "data/etf_daily.db"
BENCHMARK_SYMBOL = "000300"
OUTPUT_DIR = "strategies/low_vol_rotation/output"
