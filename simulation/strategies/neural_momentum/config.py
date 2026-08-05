"""Neural Momentum 模拟盘 — 配置（与回测定稿一致：w=0.25 Top1）"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ETF_POOL = {
    "510050": "上证50ETF（华夏）",
    "510300": "沪深300ETF（华泰柏瑞）",
    "510500": "中证500ETF（南方）",
    "512100": "中证1000ETF（南方）",
    "159915": "创业板ETF（易方达）",
    "588000": "科创50ETF（华夏）",
    "510180": "上证180ETF（华安）",
}
ETF_SYMBOLS = list(ETF_POOL.keys())

STRATEGY_ID = "neural_momentum"
STRATEGY_NAME = "Neural Momentum(动量+神经网络)"

# ═══ 混合权重（回测定稿：w=0.25，全周期 +44.1% vs 纯动量 -0.7%） ═══
WEIGHT_W = 0.25
NEURAL_SCORES_PATH = str(PROJECT_ROOT / "strategies" / "neural_momentum" / "output" / "neural_scores.csv")

# ═══ 引擎参数（与 019 动量轮动一致） ═══
MOMENTUM_WINDOW = 20
MIN_SWITCH_CONVICTION = 0.03
MIN_HOLD_DAYS = 10
RISK_MODE = "A"           # 纯信号（趋势策略 A 模式——铁律：动量类用 A）
INITIAL_CAPITAL = 10000
COMMISSION_RATE = 0.0002
SLIPPAGE = 0.0001
DB_PATH = "data/etf_daily.db"
