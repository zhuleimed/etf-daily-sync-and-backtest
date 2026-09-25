"""尾部风险轮动 — 模拟盘信号函数（自 strategies/tail_risk/engine.py 移植）

原回测逻辑（TailRiskEngine）：
  平时        → 正常动量排名（买最强）
  触发尾部风险 → 清仓，切到"波动率最低"的那只 ETF 避险

  尾部风险触发条件（满足任一）：
    1. 沪深300 近 5 日跌幅 < TAIL_THRESHOLD（-3%）
    2. 沪深300 近 10 日年化波动 > 近 20 日年化波动 × 1.5（波动率突然飙升）

移植说明：
    指数数据与回测**同源**——都从 index_daily 表读沪深300（000300）指数，
    用 strategies.momentum_rotation.data.load_benchmark_data 加载，
    进程内缓存一次（每日管线是独立进程，缓存不会跨日失效）。

    信号表达：引擎只在"排名第 1 且得分 > 0"时买入，因此避险触发时，
    让最低波动的那只拿到正分、其余全部 NaN，使引擎必然切到它。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
)
from . import config as cfg

# 沪深300 指数收盘序列（index=日期字符串）的进程内缓存
_BENCH_CACHE: pd.Series | None = None


def _benchmark_close() -> pd.Series:
    """加载沪深300指数收盘价；失败时返回空 Series（等价于"从不触发避险"）。"""
    global _BENCH_CACHE
    if _BENCH_CACHE is None:
        try:
            from strategies.momentum_rotation.data import load_benchmark_data
            df = load_benchmark_data()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            _BENCH_CACHE = df.set_index("date")["close"]
        except Exception:
            _BENCH_CACHE = pd.Series(dtype=float)
    return _BENCH_CACHE


def _tail_triggered(etf_data: dict[str, pd.DataFrame], date_idx: int) -> bool:
    """判断是否触发尾部风险（逻辑与回测 TailRiskEngine._check_tail 一致）。"""
    if date_idx < 22:
        return False

    bench = _benchmark_close()
    if bench.empty:
        return False

    # 取当日日期，对齐到指数序列
    df0 = next((d for d in etf_data.values() if d is not None and len(d) > date_idx), None)
    if df0 is None:
        return False
    date_str = pd.to_datetime(df0["date"].iloc[date_idx]).strftime("%Y-%m-%d")
    if date_str not in bench.index:
        return False
    hi = bench.index.get_loc(date_str)
    if isinstance(hi, slice):  # 索引有重复时 pandas 返回 slice
        hi = hi.start
    if hi < 22:
        return False

    # 条件1：近5日跌幅超阈值
    if bench.iloc[hi] / bench.iloc[hi - 5] - 1 < cfg.TAIL_THRESHOLD:
        return True

    # 条件2：短期波动率 > 长期波动率 × 1.5
    short = bench.iloc[hi - 9: hi + 1].pct_change().dropna()
    long = bench.iloc[hi - 19: hi + 1].pct_change().dropna()
    if len(short) > 1 and len(long) > 1:
        if short.std() * np.sqrt(252) > long.std() * np.sqrt(252) * 1.5:
            return True

    return False


def _lowest_vol_symbol(etf_data: dict[str, pd.DataFrame], date_idx: int) -> str | None:
    """找出近 VOL_WINDOW 日年化波动率最低的 ETF。"""
    vw = cfg.VOL_WINDOW
    best_sym, best_vol = None, np.inf
    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < vw or date_idx >= len(df):
            continue
        rets = df["pct_chg"].iloc[date_idx - vw + 1: date_idx + 1]
        vol = rets.std() * np.sqrt(252) if len(rets) > 1 else np.inf
        if vol < best_vol:
            best_vol, best_sym = vol, sym
    return best_sym


def compute_tail_risk_signals(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """尾部风险触发 → 只有最低波动 ETF 得正分；否则走正常动量。"""
    if _tail_triggered(etf_data, date_idx):
        safe = _lowest_vol_symbol(etf_data, date_idx)
        scores = {sym: np.nan for sym in cfg.ETF_SYMBOLS}
        if safe is not None:
            scores[safe] = 1.0  # 正分 → 引擎会切到它
        return pd.Series(scores, dtype=float)

    return compute_momentum_signals(etf_data, date_idx, momentum_window)


def rank_etfs_by_tail_risk(scores: pd.Series) -> pd.Series:
    """按得分降序排列（避险时最低波动 ETF 自然排第 1）。"""
    return rank_etfs_by_momentum(scores)
