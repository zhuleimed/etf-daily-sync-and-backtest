"""
MACD 趋势信号计算模块

核心指标：MACD (Moving Average Convergence Divergence) — Gerald Appel 经典指标。

计算步骤：
  1. EMA12 = 12日指数移动平均 of close
  2. EMA26 = 26日指数移动平均 of close
  3. MACD line = EMA12 - EMA26
  4. Signal line = EMA9 of MACD line
  5. Histogram = MACD line - Signal line

策略逻辑：
  - MACD > 0 → 多头趋势，可交易
  - MACD ≤ 0 → 空头趋势，回避
  - 综合分 = MACD强度×40% + 柱状图动量×30% + 趋势位置×30%

注意：使用 `from . import config as cfg` 动态引用，支持运行时参数调优。
"""

import numpy as np
import pandas as pd

from . import config as cfg


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均（EMA）。"""
    n = len(values)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # 第一个值为 SMA
    result[period - 1] = np.mean(values[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        result[i] = (values[i] - result[i - 1]) * alpha + result[i - 1]
    return result


def compute_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """
    计算 MACD 线、信号线、柱状图。

    Parameters
    ----------
    close : np.ndarray
        收盘价序列。
    fast, slow, signal : int
        MACD 经典参数 12/26/9。

    Returns
    -------
    dict with keys: 'macd', 'signal_line', 'histogram', 'ema_fast', 'ema_slow'
    """
    n = len(close)
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
            macd[i] = ema_fast[i] - ema_slow[i]

    signal_line = _ema(np.nan_to_num(macd, 0), signal)
    # 修复 signal_line 起始位置
    for i in range(n):
        if np.isnan(macd[i]):
            signal_line[i] = np.nan

    histogram = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(macd[i]) and not np.isnan(signal_line[i]):
            histogram[i] = macd[i] - signal_line[i]

    return {
        "macd": macd,
        "signal_line": signal_line,
        "histogram": histogram,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
    }


def judge_market_regime(
    index_data: pd.DataFrame,
    date_idx: int,
    ma_period: int = 60,
    bear_threshold: float = -0.02,
) -> dict:
    """沪深300 MA60 市场状态判断。"""
    if index_data is None or index_data.empty:
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}
    if date_idx < ma_period or date_idx >= len(index_data):
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}
    close = index_data.iloc[date_idx]["close"]
    ma = index_data.iloc[date_idx - ma_period + 1:date_idx + 1]["close"].mean()
    ratio = close / ma - 1
    if ratio > abs(bear_threshold):
        return {"regime": "bull", "ma_value": ma, "ratio": ratio}
    elif ratio < bear_threshold:
        return {"regime": "bear", "ma_value": ma, "ratio": ratio}
    return {"regime": "neutral", "ma_value": ma, "ratio": ratio}


def compute_macd_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算每只 ETF 的 MACD 综合评分。

    评分逻辑：
      1. MACD ≤ 0 → 得分 = 0（空头趋势，回避）
      2. 评分 = MACD强度_z × W_MACD + Histogram_z × W_HIST + 趋势位置_z × W_TREND
        其中各因子除以收盘价做归一化，再截面 Z-Score

    Returns
    -------
    pd.Series: index=ETF代码, values=综合得分（≤0 表示不满足条件）
    """
    needed = cfg.MACD_SLOW + cfg.MACD_SIGNAL + 5  # 至少需要 slow+signal+缓冲
    macd_norm = {}    # MACD / close
    hist_norm = {}    # Histogram / close
    trend_pos = {}    # close / EMA26 - 1

    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < needed or date_idx >= len(df):
            macd_norm[sym] = np.nan
            continue

        close = df["close"].values[:date_idx + 1]
        result = compute_macd(close, cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)

        c = close[date_idx]
        if c <= 0:
            macd_norm[sym] = np.nan
            continue

        macd_val = result["macd"][date_idx]
        hist_val = result["histogram"][date_idx]
        ema_slow = result["ema_slow"][date_idx]

        macd_norm[sym] = macd_val / c if not np.isnan(macd_val) else np.nan
        hist_norm[sym] = hist_val / c if not np.isnan(hist_val) else np.nan
        trend_pos[sym] = (c / ema_slow - 1) if not np.isnan(ema_slow) and ema_slow > 0 else np.nan

    # 转为 Series
    macd_s = pd.Series(macd_norm, dtype=float)
    hist_s = pd.Series(hist_norm, dtype=float)
    trend_s = pd.Series(trend_pos, dtype=float)

    # 过滤：MACD ≤ 0 → 不得交易
    mask_bear = macd_s <= 0

    # Z-Score 标准化
    def _zscore(s):
        clean = s.dropna()
        if len(clean) < 2 or clean.std() == 0:
            return pd.Series(0.0, index=s.index)
        return (s - clean.mean()) / clean.std()

    macd_z = _zscore(macd_s.fillna(0))
    hist_z = _zscore(hist_s.fillna(0))
    trend_z = _zscore(trend_s.fillna(0))

    # 合成评分
    scores = pd.Series(0.0, index=cfg.ETF_SYMBOLS)
    for sym in cfg.ETF_SYMBOLS:
        if pd.isna(macd_s.get(sym)) or mask_bear.get(sym, True):
            scores[sym] = 0.0
        else:
            scores[sym] = (
                cfg.WEIGHT_MACD * macd_z.get(sym, 0)
                + cfg.WEIGHT_HIST * hist_z.get(sym, 0)
                + cfg.WEIGHT_TREND * trend_z.get(sym, 0)
            )

    return scores


def rank_etfs_by_macd(macd_scores: pd.Series) -> pd.Series:
    """按 MACD 综合评分降序排列（只考虑得分>0的标的）。"""
    valid = macd_scores[macd_scores > 0].dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))


def compute_macd_spread(macd_scores: pd.Series) -> float:
    """MACD 得分截面标准差。"""
    valid = macd_scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
