"""
沪深300均线择时策略 - 配置模块

比市场宽度更简单直接：只看HS300指数是否站上MA。
站上=牛市=全力动量轮动，跌破=熊市=空仓避险。
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

# 均线择时
MA_PERIOD = 10                  # HS300均线周期（10日=双周线，网格扫描最优）
MOMENTUM_WINDOW = 20

# 交易费用
COMMISSION_RATE = 0.0002; SLIPPAGE = 0.0001; TAX_RATE = 0.0
ADJUSTMENT_DAYS = 5
MIN_SWITCH_CONVICTION = 0.03; MIN_HOLD_DAYS = 10
TOP_N = 1; DYNAMIC_WINDOW_ENABLED = False
RISK_MODE = "A"

DB_PATH = "data/etf_daily.db"
BENCHMARK_SYMBOL = "000300"
OUTPUT_DIR = "strategies/hs300_ma_timing/output"
