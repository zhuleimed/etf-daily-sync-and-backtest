"""价差回归 — 模拟盘信号函数（自 strategies/spread_reversion/engine.py 移植）

原回测逻辑（SpreadReversionEngine）：
    每 rebalance_days 天调仓，先算 7 只 ETF 近 LOOKBACK 日的累计涨幅，做横向比较：
        z = (本只累计涨幅 − 7只平均涨幅) / 7只标准差
        权重 = 1/(1+|z|)   当 |z| > ENTRY_Z（明显跑偏）→ 大幅低配
             = 1.0         当 |z| < EXIT_Z （贴近平均）→ 满配
             = 0.5         中间地带 → 半配
    即"谁偏离大部队越远，越少配；越贴近平均，越多配"。

移植说明（单标的适配）：
    原回测是**永远满仓的多标的加权组合**（权重恒为正，从不空仓），
    模拟盘 DailySimEngine 只能持有一只。因此把"原始权重公式"直接当作得分：
        得分 = 上面的权重值（恒为正）
    引擎取排名第 1 → 即"权重最大的那一只"，等价于原组合里配得最重的标的。
    这是权重函数在单标的约束下的保序归约。

    ⚠️ 已知局限：原策略同时持有全部 7 只并动态调权，本移植只能持其中 1 只，
       因此不会出现原回测那种"分散持有一篮子"的平滑度。若要完整还原，
       需要把该策略改造成"独立多标的策略"（不走 DailySimEngine），属另一项工作。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def compute_spread_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,  # 占位参数：保持引擎接口一致，本策略用 LOOKBACK
) -> pd.Series:
    """按原回测的权重公式给出得分：越贴近 7 只平均，得分越高。"""
    lookback = cfg.LOOKBACK
    entry_z = cfg.ENTRY_Z
    exit_z = cfg.EXIT_Z

    # 原回测要求 si >= lb + 10 才调仓
    if date_idx < lookback + 10:
        return pd.Series({sym: np.nan for sym in cfg.ETF_SYMBOLS}, dtype=float)

    cum_rets: dict[str, float] = {}
    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx >= len(df):
            continue
        cum_rets[sym] = df["close"].iloc[date_idx] / df["close"].iloc[date_idx - lookback] - 1

    if len(cum_rets) < 2:
        return pd.Series({sym: np.nan for sym in cfg.ETF_SYMBOLS}, dtype=float)

    vals = np.array(list(cum_rets.values()))
    avg = vals.mean()
    std = max(vals.std(), 0.001)

    scores: dict[str, float] = {}
    for sym in cfg.ETF_SYMBOLS:
        if sym not in cum_rets:
            scores[sym] = np.nan
            continue
        az = abs((cum_rets[sym] - avg) / std)
        if az > entry_z:
            scores[sym] = 1.0 / (1.0 + az)
        elif az < exit_z:
            scores[sym] = 1.0
        else:
            scores[sym] = 0.5

    return pd.Series(scores, dtype=float)


def rank_etfs_by_spread(scores: pd.Series) -> pd.Series:
    """按得分降序排列，返回 {1: 权重最大的ETF, 2: 次之, ...}。"""
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    return pd.Series(
        valid.sort_values(ascending=False).index.values,
        index=range(1, len(valid) + 1),
    )
