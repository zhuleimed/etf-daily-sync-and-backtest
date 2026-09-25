"""中位数动量 — 模拟盘信号函数（自 strategies/median_momentum/engine.py 移植）

原回测逻辑（MedianMomentumEngine）：
    mom = compute_momentum_signals(...)        # 与动量轮动完全相同的动量值
    ranking = rank_etfs_by_momentum(mom)
    target = ranking.get(RANK_POSITION)        # ← 唯一区别：买第 2 名，不买第 1 名

移植说明：
    得分与动量轮动一模一样（本策略的设计就是"动量值不变、只改买第几名"），
    区别体现在**排名函数** rank_etfs_by_median 上：它把第 RANK_POSITION 名
    提升到第 1 位，这样模拟盘引擎（只买 ranking[1]）买到的就是第 2 名。

    为什么要在排名里做手脚：模拟盘 DailySimEngine 固定取 ranking[1] 作为目标，
    原回测则是自己取 ranking[rank_position]。为了让引擎"买第2名"，只能让
    排名函数的第 1 位就是那个第 2 名。
"""

from __future__ import annotations

import pandas as pd

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
)
from . import config as cfg


def compute_median_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """动量得分——与原回测一致，直接复用动量轮动的计算。"""
    return compute_momentum_signals(etf_data, date_idx, momentum_window)


def rank_etfs_by_median(scores: pd.Series) -> pd.Series:
    """把第 RANK_POSITION 名提到第 1 位，其余按原顺序跟在后面。

    原回测的兜底：候选不足 RANK_POSITION 只时退回第 1 名。
    """
    ranked = rank_etfs_by_momentum(scores)
    pos = cfg.RANK_POSITION
    if len(ranked) < pos:
        return ranked

    target = ranked[pos]
    rest = [ranked[i] for i in ranked.index if i != pos]
    return pd.Series([target] + rest, index=range(1, len(rest) + 2))
