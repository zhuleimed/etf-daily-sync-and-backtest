"""
信号计算模块 — 根据市场状态动态切换信号

包含三种信号函数：
  1. 动量信号（牛市）：20日收益率排名
  2. 均值回归信号（震荡）：%B超卖+RSI低位+缩量
  3. 空仓信号（熊市）：全部得分为0

顶层入口：compute_adaptive_scores() — 自动识别当前状态并选择信号
"""

import numpy as np
import pandas as pd

from . import config as cfg
from .regime import detect_regime


# ═════════════════════════════════════════════════════════════════════
#  信号1: 动量信号（牛市 → 买强势）
# ═════════════════════════════════════════════════════════════════════

def _compute_momentum_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算动量评分（牛市模式用）。

    Score = 20日收益率，Z-score标准化。
    得分越高 = 近期表现越强势 → 买入。
    """
    scores = {}
    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < cfg.MOM_WINDOW or date_idx >= len(df):
            scores[sym] = np.nan
            continue
        close = df["close"].values
        mom = close[date_idx] / close[date_idx - cfg.MOM_WINDOW] - 1
        scores[sym] = mom

    s = pd.Series(scores, dtype=float)
    clean = s.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return pd.Series(0.0, index=cfg.ETF_SYMBOLS)
    return (s - clean.mean()) / clean.std()


# ═════════════════════════════════════════════════════════════════════
#  信号2: 均值回归信号（震荡 → 买超卖）
# ═════════════════════════════════════════════════════════════════════

def _compute_pct_b(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> np.ndarray:
    """计算布林带 %B。"""
    n = len(close)
    pct_b = np.full(n, np.nan)
    if n < period:
        return pct_b
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = close[i - period + 1:i + 1]
        sma[i] = np.mean(w)
        std[i] = np.std(w, ddof=1)
    for i in range(period - 1, n):
        upper = sma[i] + std_mult * std[i]
        lower = sma[i] - std_mult * std[i]
        bw = upper - lower
        pct_b[i] = (close[i] - lower) / bw if bw > 0 else 0.5
    return pct_b


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI。"""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g = np.full(n, np.nan)
    avg_l = np.full(n, np.nan)
    avg_g[period] = np.mean(gains[:period])
    avg_l[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + losses[i - 1]) / period
    for i in range(period, n):
        if avg_l[i] == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_g[i] / avg_l[i]))
    return rsi


def _compute_reversion_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算均值回归评分（震荡模式用）。

    高分 = 超卖程度高（该买入），低分 = 超卖已修复。
    硬过滤器：RSI < THRESHOLD 且 %B < THRESHOLD。
    """
    needed = max(cfg.REV_BB_PERIOD, cfg.REV_RSI_PERIOD) + 10
    pct_b_vals = {}
    rsi_vals = {}
    vol_vals = {}

    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < needed or date_idx >= len(df):
            pct_b_vals[sym] = np.nan
            continue
        close = df["close"].values[:date_idx + 1]
        vol = df["volume"].values[:date_idx + 1]

        pct_b_arr = _compute_pct_b(close, cfg.REV_BB_PERIOD, cfg.REV_BB_STD)
        rsi_arr = _compute_rsi(close, cfg.REV_RSI_PERIOD)

        pct_b_vals[sym] = pct_b_arr[date_idx]
        rsi_vals[sym] = rsi_arr[date_idx]

        vol_ma = np.mean(vol[-cfg.REV_BB_PERIOD:]) if len(vol) >= cfg.REV_BB_PERIOD else np.nan
        vol_vals[sym] = vol[-1] / vol_ma if vol_ma and vol_ma > 0 else np.nan

    pct_b_s = pd.Series(pct_b_vals, dtype=float)
    rsi_s = pd.Series(rsi_vals, dtype=float)
    vol_s = pd.Series(vol_vals, dtype=float)

    # 硬过滤：RSI不够低 或 %B不够低 → 0分
    mask = (rsi_s >= cfg.REV_OVERSOLD_RSI) | rsi_s.isna() | \
           (pct_b_s >= cfg.REV_OVERSOLD_PCT_B) | pct_b_s.isna()

    # 分量转换
    pct_b_raw = (-pct_b_s).clip(lower=0)
    rsi_raw = ((cfg.REV_OVERSOLD_RSI - rsi_s) / cfg.REV_OVERSOLD_RSI).clip(lower=0)
    vol_raw = (1.0 - vol_s).clip(lower=0)

    def _z(s):
        c = s.dropna()
        if len(c) < 2 or c.std() == 0:
            return pd.Series(0.0, index=s.index)
        return (s - c.mean()) / c.std()

    pct_b_z = _z(pct_b_raw.fillna(0))
    rsi_z = _z(rsi_raw.fillna(0))
    vol_z = _z(vol_raw.fillna(0))

    scores = pd.Series(0.0, index=cfg.ETF_SYMBOLS)
    for sym in cfg.ETF_SYMBOLS:
        if mask.get(sym, True):
            scores[sym] = 0.0
        else:
            scores[sym] = pct_b_z.get(sym, 0) * 0.40 + rsi_z.get(sym, 0) * 0.40 + vol_z.get(sym, 0) * 0.20

    return scores


# ═════════════════════════════════════════════════════════════════════
#  顶层入口：自适应评分
# ═════════════════════════════════════════════════════════════════════

def compute_adaptive_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    index_data: pd.DataFrame = None,
) -> pd.Series:
    """
    自适应信号计算 — 顶层入口。

    自动检测市场状态，选择对应的信号函数：
      - 牛市 → 动量信号（买强势）
      - 震荡 → 均值回归信号（买超卖）
      - 熊市 → 全部0分（空仓）

    Parameters
    ----------
    etf_data : dict[str, pd.DataFrame]
        {symbol: df} 格式，需有 close 列。
    date_idx : int
        当前日期索引。
    index_data : pd.DataFrame, optional
        沪深300数据，用于市场状态识别。

    Returns
    -------
    pd.Series: index=ETF代码, values=综合得分
    """
    # 1. 检测市场状态
    regime_info = detect_regime(index_data, date_idx)
    regime = regime_info["regime"]

    # 2. 根据状态选择信号
    if regime == "bull":
        scores = _compute_momentum_scores(etf_data, date_idx)
    elif regime == "neutral":
        scores = _compute_reversion_scores(etf_data, date_idx)
    else:  # bear
        scores = pd.Series(0.0, index=cfg.ETF_SYMBOLS)

    return scores


def rank_etfs_by_adaptive(scores: pd.Series) -> pd.Series:
    """按自适应评分降序排列。"""
    valid = scores[scores > 0].dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))
