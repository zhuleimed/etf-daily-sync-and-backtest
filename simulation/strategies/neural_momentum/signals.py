"""Neural Momentum 混合信号：score = w × 动量z + (1-w) × 神经z

回测定稿（w=0.25）：全周期 +44.1% vs 纯动量 -0.7%——动量分 25% 修正、神经预测主导。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.strategies.neural_momentum.config import (
    WEIGHT_W, NEURAL_SCORES_PATH,
)

logger = logging.getLogger("neural_signals")

# 神经评分（date × ETF，z-score）——每日加载一次缓存
_NEURAL_CACHE: pd.DataFrame | None = None


def load_neural_scores() -> pd.DataFrame | None:
    """加载神经评分（缓存，文件更新后重载）。"""
    global _NEURAL_CACHE
    p = Path(NEURAL_SCORES_PATH)
    if not p.exists():
        logger.warning(f"神经评分文件不存在: {p}（退化为纯动量）")
        return None
    mtime = p.stat().st_mtime
    if _NEURAL_CACHE is None or getattr(_NEURAL_CACHE, "_mtime", 0) != mtime:
        df = pd.read_csv(p, index_col=0)
        df.index = df.index.astype(str)
        df._mtime = mtime
        _NEURAL_CACHE = df
    return _NEURAL_CACHE


def neural_momentum_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """混合评分信号（与 DailySimEngine 的 signal_func 接口兼容）。

    Args:
        etf_data: {symbol: df}，df 含 date/momentum 列（data.py 预计算）。
        date_idx: 信号日索引（T 日收盘，T+1 执行——引擎保证）。
        momentum_window: 动量窗口。

    Returns:
        pd.Series（index=ETF 代码, values=混合分，NaN=数据不足）。
    """
    # ── 1. 动量分（与纯动量一致） ──
    momentums: dict[str, float] = {}
    signal_date = None
    for sym, df in etf_data.items():
        if date_idx >= momentum_window and date_idx < len(df):
            val = df.loc[date_idx, "momentum"]
            momentums[sym] = float(val) if not pd.isna(val) else np.nan
            if signal_date is None:
                signal_date = str(df.loc[date_idx, "date"])[:10]
        else:
            momentums[sym] = np.nan

    # ── 2. 神经分（当日评分） ──
    neural = load_neural_scores()
    n_row = None
    if neural is not None and signal_date is not None:
        if signal_date in neural.index:
            n_row = neural.loc[signal_date]

    # ── 3. 混合：动量 z-score + 神经分 ──
    m_series = pd.Series(momentums).dropna()
    if len(m_series) < 3:
        return m_series
    m_z = (m_series - m_series.mean()) / (m_series.std() + 1e-9)

    merged: dict[str, float] = {}
    for sym in m_z.index:
        nz = 0.0
        if n_row is not None and sym in n_row.index and not pd.isna(n_row[sym]):
            nz = float(n_row[sym])
        merged[sym] = WEIGHT_W * float(m_z[sym]) + (1 - WEIGHT_W) * nz
    return pd.Series(merged)
