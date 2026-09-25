"""Sortino 排名 — 模拟盘信号函数（自 strategies/sortino_ranking/engine.py 移植）

原回测逻辑（SortinoRankingEngine._score）：
    收益    = 收盘/前 MOMENTUM_WINDOW 日收盘 − 1
    下行波动 = 只取"下跌那几天"的收益率标准差 × √252（年化）
    得分    = 收益 / max(下行波动, 0.01)

    与 Sharpe 的区别：涨的时候波动不算风险，只有跌才算。
    → 得分偏向"稳步上涨、少大跌"的 ETF，而非"猛涨猛跌"的。

    边界：下跌天数不足 2 天时退回用全部收益率的标准差（原回测同此处理）。

移植说明：得分函数逐行照搬原回测，未做改动。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def compute_sortino_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """计算各 ETF 的 Sortino 得分（收益 ÷ 年化下行波动率）。"""
    window = cfg.MOMENTUM_WINDOW
    vol_window = cfg.VOL_WINDOW

    scores: dict[str, float] = {}
    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < max(window, vol_window) or date_idx >= len(df):
            scores[sym] = np.nan
            continue

        ret = df["close"].iloc[date_idx] / df["close"].iloc[date_idx - window] - 1
        rets = df["pct_chg"].iloc[date_idx - vol_window + 1: date_idx + 1]
        neg = rets[rets < 0]
        if len(neg) > 1:
            down_vol = neg.std() * np.sqrt(252)
        else:
            down_vol = rets.std() * np.sqrt(252)
        scores[sym] = ret / max(down_vol, 0.01)

    return pd.Series(scores, dtype=float)


def rank_etfs_by_sortino(scores: pd.Series) -> pd.Series:
    """按得分降序排列，返回 {1: 最优ETF代码, 2: 次优, ...}。"""
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    return pd.Series(
        valid.sort_values(ascending=False).index.values,
        index=range(1, len(valid) + 1),
    )
