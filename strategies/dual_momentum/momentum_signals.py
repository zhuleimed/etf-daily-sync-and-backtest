"""双动量 — 模拟盘信号函数（自 strategies/dual_momentum/engine.py 移植）

原回测逻辑（DualMomentumEngine）：
    动量 = compute_momentum_signals(..., MOMENTUM_WINDOW=15)   # 相对强度
    过滤：收盘价 ≤ 自身 ABS_MA(30) 日均线的 ETF → 动量置 NaN，不参与排名
    排名：过滤后的动量降序，取第 1 名

    这是经典"双动量"（Gary Antonacci）：
      绝对动量（价格在均线之上）决定"要不要买"
      相对动量（涨幅排名）     决定"买哪一只"
    → 全市场都在均线之下时，一只都不选，自然空仓避险。

移植说明：得分函数逐行照搬原回测，未做改动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
)
from . import config as cfg


def compute_dual_momentum_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 15,
) -> pd.Series:
    """动量得分，但"跌破自身 ABS_MA 均线"的 ETF 被剔除（置 NaN）。"""
    momentum = compute_momentum_signals(
        etf_data, date_idx, cfg.MOMENTUM_WINDOW,
    )

    abs_ma = cfg.ABS_MA
    for sym in momentum.index:
        if pd.isna(momentum[sym]):
            continue
        df = etf_data.get(sym)
        if df is None or date_idx >= len(df):
            momentum[sym] = np.nan
            continue
        if date_idx < abs_ma:
            continue  # 均线尚不可用 → 与原回测一致，视为通过绝对动量过滤
        close = df["close"].iloc[date_idx]
        ma = df["close"].iloc[date_idx - abs_ma + 1: date_idx + 1].mean()
        if close <= ma:
            momentum[sym] = np.nan

    return momentum


def rank_etfs_by_dual_momentum(scores: pd.Series) -> pd.Series:
    """按过滤后的动量降序排列。"""
    return rank_etfs_by_momentum(scores)
