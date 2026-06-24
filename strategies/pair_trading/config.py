"""
配对交易策略 - 配置模块

核心逻辑：
  利用 ETF 对之间的价差均值回归特性，多空对冲交易。
  价差偏离历史均值时开仓，回归时平仓。

市场中性：不依赖大盘涨跌，赚取价差回归收益。
"""

# ── ETF 配对列表 ──
# 每对包含 code_a, code_b 及各自名称
PAIRS = [
    {"name": "上证50↔沪深300", "a": "510050", "b": "510300"},
    {"name": "中证500↔中证1000", "a": "510500", "b": "512100"},
    {"name": "沪深300↔中证500", "a": "510300", "b": "510500"},
    {"name": "创业板↔科创50", "a": "159915", "b": "588000"},
]

# ── 参数 ──
INITIAL_CAPITAL = 10000       # 总资产
CAPITAL_PER_PAIR = 2500       # 每对分配资金（1250多 + 1250空名义本金）

ZSCORE_PERIOD = 60            # z-score 统计窗口
ZSCORE_OPEN = 2.0             # 开仓阈值：|z| > 2.0 触发开仓
ZSCORE_CLOSE = 0.5            # 平仓阈值：|z| < 0.5 获利了结
ZSCORE_STOP = 3.0             # 止损阈值：|z| > 3.0 强制止损

COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
DB_PATH = "data/etf_daily.db"
OUTPUT_DIR = "strategies/pair_trading/output"
