"""量价配合 — 模拟盘信号函数（自 strategies/volume_price/engine.py 移植）

原回测逻辑（VolumePriceEngine）：
    动量 = compute_momentum_signals(...)           # 基础仍是动量排名
    量比 = 近 VOL_SHORT_PERIOD 日均量 / 近 VOL_LONG_PERIOD 日均量
    过滤：量比 < VOL_THRESHOLD 的 ETF → 动量置 NaN，不参与排名
    排名：过滤后的动量降序，取第 1 名

    核心思想：缩量上涨可能是假突破，只有"放量上涨"（量比≥1.2）才值得追。

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


def compute_volume_price_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """动量得分，但"量比不足"的 ETF 被剔除（置 NaN）。"""
    momentum = compute_momentum_signals(etf_data, date_idx, momentum_window)

    short_p = cfg.VOL_SHORT_PERIOD
    long_p = cfg.VOL_LONG_PERIOD
    threshold = cfg.VOL_THRESHOLD

    for sym in momentum.index:
        df = etf_data.get(sym)
        if df is None or date_idx < long_p or date_idx >= len(df):
            momentum[sym] = np.nan
            continue
        short_vol = df["volume"].iloc[date_idx - short_p + 1: date_idx + 1].mean()
        long_vol = df["volume"].iloc[date_idx - long_p + 1: date_idx + 1].mean()
        ratio = short_vol / long_vol if long_vol > 0 else 1.0
        if ratio < threshold:
            momentum[sym] = np.nan  # 量比不足 → 不参与排名

    return momentum


def rank_etfs_by_volume_price(scores: pd.Series) -> pd.Series:
    """按过滤后的动量降序排列。"""
    return rank_etfs_by_momentum(scores)
