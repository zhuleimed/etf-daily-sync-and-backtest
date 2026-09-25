"""布林带回归 — 模拟盘信号函数（自 strategies/bollinger_reversion/engine.py 移植）

原回测逻辑（BollingerReversionEngine）：
  每 rebalance_days 天调仓一次，把"价格跌破布林带下轨"的 ETF 全部买入，
  仓位权重 = (PctB阈值 − PctB)，即跌破越深、买得越多。

移植说明（单标的适配）：
  模拟盘 DailySimEngine 是"单标的持有"结构（模拟盘配置 TOP_N=1），无法同时持有
  多只 ETF。因此把"多标的按跌破深度加权"收敛为"只取跌破最深的那一只"：
      得分 = PctB阈值 − PctB
  只有 PctB < 阈值（跌破下轨）的 ETF 得分为正，未跌破的一律 NaN → 不参与排名。
  引擎只在目标得分 > 0 时才开仓，正好对应原回测的"有标的跌破才买入"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def compute_bollinger_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,  # 占位参数：保持与引擎接口一致，本策略不用动量
) -> pd.Series:
    """计算布林带回归得分：跌破下轨越深，得分越高（正分＝可买）。

    %B 的定义：(收盘 − 下轨) / (上轨 − 下轨)
      %B = 1 → 贴上轨；%B = 0 → 贴下轨；%B < 0 → 已跌破下轨。
    """
    period = cfg.BB_PERIOD
    std_mult = cfg.BB_STD
    threshold = cfg.PctB_THRESHOLD

    scores: dict[str, float] = {}
    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < period or date_idx >= len(df):
            scores[sym] = np.nan
            continue

        closes = df["close"].iloc[date_idx - period + 1: date_idx + 1]
        ma = closes.mean()
        sd = closes.std()
        upper = ma + std_mult * sd
        lower = ma - std_mult * sd
        close = df["close"].iloc[date_idx]

        pct_b = (close - lower) / (upper - lower) if upper > lower else 0.5
        # 只有跌破下轨（PctB < 阈值）才给正分，跌破越深分越高
        scores[sym] = (threshold - pct_b) if pct_b < threshold else np.nan

    return pd.Series(scores, dtype=float)


def rank_etfs_by_bollinger(scores: pd.Series) -> pd.Series:
    """按得分降序排列，返回 {1: 最优ETF代码, 2: 次优, ...}。"""
    valid = scores.dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    return pd.Series(
        valid.sort_values(ascending=False).index.values,
        index=range(1, len(valid) + 1),
    )
