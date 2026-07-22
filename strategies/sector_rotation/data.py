"""数据加载模块 — 复用 momentum_rotation 实现，使用本地 config"""
from strategies.momentum_rotation.data import (
    load_all_etf_data, load_benchmark_data, compute_equal_weight_benchmark,
)
# 注意：load_all_etf_data 默认使用传入的 symbols 参数或 config.ETF_SYMBOLS
# 调用时需传入 sector_rotation 的 ETF_SYMBOLS
