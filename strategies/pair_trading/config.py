"""
配对交易策略 - 配置模块

核心逻辑：
  利用 ETF 对之间的价差均值回归特性，多空对冲交易。
  价差偏离历史均值时开仓，回归时平仓。

市场中性：不依赖大盘涨跌，赚取价差回归收益。
"""

# ── ETF 配对列表 ──
# 选择不同风格的ETF配对（大盘vs成长），避免高相关性组合
PAIRS = [
    {"name": "上证50↔创业板", "a": "510050", "b": "159915"},
    {"name": "上证50↔科创50", "a": "510050", "b": "588000"},
]

# ── 参数 ──
INITIAL_CAPITAL = 10000
CAPITAL_PER_PAIR = 5000       # 因只有2对，每对资金提高至5000

ZSCORE_PERIOD = 60            # z-score 统计窗口（60日≈3个月）
ZSCORE_OPEN = 2.0             # 开仓阈值：|z| > 2.0
ZSCORE_CLOSE = 0.5            # 平仓阈值：|z| < 0.5（获利了结）
ZSCORE_STOP = 3.0             # 止损阈值：|z| > 3.0

COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
DB_PATH = "data/etf_daily.db"
OUTPUT_DIR = "strategies/pair_trading/output"
