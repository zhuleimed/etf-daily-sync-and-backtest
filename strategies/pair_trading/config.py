"""
配对交易策略 - 配置模块

核心逻辑：
  利用 ETF 对之间的价差均值回归特性，多空对冲交易。
  价差偏离历史均值时开仓，回归时平仓。

市场中性：不依赖大盘涨跌，赚取价差回归收益。
关键：选择不同风格的ETF（大盘vs成长），避免同质化配对。
"""

# ── ETF 配对列表（大盘vs成长，三对不同风格）──
PAIRS = [
    {"name": "上证50↔创业板", "a": "510050", "b": "159915"},
    {"name": "沪深300↔创业板", "a": "510300", "b": "159915"},
    {"name": "上证50↔科创50", "a": "510050", "b": "588000"},
]

# ── 参数 ──
INITIAL_CAPITAL = 10000
CAPITAL_PER_PAIR = 3333          # 三对均分

ZSCORE_PERIOD = 60               # z-score 统计窗口
ZSCORE_OPEN = 2.0                # 开仓阈值：|z| > 2.0（从3.0下调，增加触发频率，模拟盘自加入以来零交易）
                                     # 2.5也能用，但2.0在2026牛市中更易触发信号

# 不对称阈值（方向感知开仓）
# 持有价值(上证50/沪深300)时切换至成长(创业板/科创50) 与 反向 使用不同阈值
# None=使用对称的 ZSCORE_OPEN；设为具体值则覆盖对应方向
ZSCORE_OPEN_GROWTH: float | None = None  # z > +this → 买成长（小于 ZSCORE_OPEN 则更容易切向成长）
ZSCORE_OPEN_VALUE: float | None = None   # z < -this → 买价值
ZSCORE_CLOSE = 0.3               # 平仓阈值：|z| < 0.3（早获利）
ZSCORE_STOP = 3.0                # 止损阈值：|z| > 3.0

COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
DB_PATH = "data/etf_daily.db"
OUTPUT_DIR = "strategies/pair_trading/output"
