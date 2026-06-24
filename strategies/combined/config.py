"""组合策略配置 — 动量轮动 + 配对交易"""

# 资金分配
TOTAL_CAPITAL = 10000
MOMENTUM_PCT = 0.8       # 80% 给动量轮动（主攻收益）
PAIR_PCT = 0.2           # 20% 给配对交易（降低回撤）

# 子策略参数覆盖
MOMENTUM_CONFIG = {
    "risk_mode": "A",
}

PAIR_CONFIG = {}

OUTPUT_DIR = "strategies/combined/output"
