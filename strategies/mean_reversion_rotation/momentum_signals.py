"""
均值回归信号计算模块

核心指标：%B（布林带位置）+ RSI（超卖确认）+ 量能（恐慌退潮）

评分逻辑（与趋势策略完全相反）：
  - 高分 = 超卖程度高 → 买入信号（别人恐惧我贪婪）
  - 低分 = 价格已回归 → 卖出信号（别人贪婪我恐惧）

计算步骤：
  1. 布林带： SMA20, upper=SMA20+2×std, lower=SMA20-2×std
  2. %B = (close - lower) / (upper - lower)  → 位置0-1，<0=跌破下轨
  3. RSI(14) → <35=超卖确认
  4. 量比 = volume / MA20_volume → <1=缩量=抛压减轻
"""

import numpy as np
import pandas as pd

from . import config as cfg


# ═════════════════════════════════════════════════════════════════════
#  布林带 %B 计算
# ═════════════════════════════════════════════════════════════════════

def compute_pct_b(
    close: np.ndarray,
    period: int = 20,
    std_mult: float = 2.0,
) -> np.ndarray:
    """
    计算布林带 %B 位置。

    %B = (close - lower) / (upper - lower)
    %B < 0   → 跌破下轨（超卖）
    %B > 1   → 突破上轨（超买）
    %B = 0.5 → 中轨位置（均值）

    Parameters
    ----------
    close : np.ndarray
        收盘价序列。
    period : int
        布林带周期。
    std_mult : float
        标准差倍数。

    Returns
    -------
    np.ndarray
        %B 值数组，前 period 个为 NaN。
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)

    # 移动平均 + 标准差
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    pct_b = np.full(n, np.nan)

    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        sma[i] = np.mean(window)
        std[i] = np.std(window, ddof=1)

    # %B
    for i in range(period - 1, n):
        upper = sma[i] + std_mult * std[i]
        lower = sma[i] - std_mult * std[i]
        band_width = upper - lower
        if band_width > 0:
            pct_b[i] = (close[i] - lower) / band_width
        else:
            pct_b[i] = 0.5  # 零宽度时取中值

    return pct_b


# ═════════════════════════════════════════════════════════════════════
#  RSI 计算（与 RSI 策略相同，复用确保一致性）
# ═════════════════════════════════════════════════════════════════════

def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder 平滑 RSI。"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ═════════════════════════════════════════════════════════════════════
#  市场状态判断
# ═════════════════════════════════════════════════════════════════════

def judge_market_regime(
    index_data: pd.DataFrame,
    date_idx: int,
    ma_period: int = 200,
    bear_threshold: float = -0.05,
) -> dict:
    """
    沪深300 MA200 市场状态判断。

    均值回归策略要求更严格的市场过滤器：
    - 只有非熊市（close > MA200）才交易
    - 熊市中不做均值回归（抄底可能抄在半山腰）
    """
    if index_data is None or index_data.empty:
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}
    if date_idx < ma_period or date_idx >= len(index_data):
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}
    close = index_data.iloc[date_idx]["close"]
    # MA200：获取最近200个交易日收盘价
    ma = index_data.iloc[date_idx - ma_period + 1:date_idx + 1]["close"].mean()
    ratio = close / ma - 1
    if ratio > abs(bear_threshold):
        return {"regime": "bull", "ma_value": ma, "ratio": ratio}
    elif ratio < bear_threshold:
        return {"regime": "bear", "ma_value": ma, "ratio": ratio}
    return {"regime": "neutral", "ma_value": ma, "ratio": ratio}


# ═════════════════════════════════════════════════════════════════════
#  均值回归综合评分
# ═════════════════════════════════════════════════════════════════════

def compute_reversion_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算每只 ETF 的均值回归综合评分。

    评分逻辑（高分=该买入）：
      1. 硬过滤器：RSI < OVERSOLD_RSI_THRESHOLD 且 %B < OVERSOLD_PCT_B_THRESHOLD
         → 不满足任一条件的直接得分为 0
      2. 评分 = %B分×权重 + RSI分×权重 + 量能分×权重 + 趋势分×权重
      3. 各分量先经非线性转换（越超卖得分越高），再 Z-score 标准化

    Parameters
    ----------
    etf_data : dict[str, pd.DataFrame]
        {symbol: df} 格式。
    date_idx : int
        当前日期索引。

    Returns
    -------
    pd.Series: index=ETF代码, values=综合得分（<=0 表示不满足条件）
    """
    needed = max(cfg.BB_PERIOD, cfg.RSI_PERIOD) + 10
    pct_b_vals = {}
    rsi_vals = {}
    vol_ratio_vals = {}
    dist_ma_vals = {}

    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < needed or date_idx >= len(df):
            pct_b_vals[sym] = np.nan
            continue

        close = df["close"].values[:date_idx + 1]
        volume = df["volume"].values[:date_idx + 1]

        # %B 计算
        pct_b_arr = compute_pct_b(close, cfg.BB_PERIOD, cfg.BB_STD)
        pct_b = pct_b_arr[date_idx]

        # RSI 计算
        rsi_arr = compute_rsi(close, cfg.RSI_PERIOD)
        rsi = rsi_arr[date_idx]

        # 量比：当期量 / 均量
        vol_ma = np.mean(volume[-cfg.BB_PERIOD:]) if len(volume) >= cfg.BB_PERIOD else np.nan
        vol_ratio = volume[-1] / vol_ma if vol_ma and vol_ma > 0 else np.nan

        # 偏离均线距离：close / SMA20 - 1
        close_sma = np.mean(close[-cfg.BB_PERIOD:]) if len(close) >= cfg.BB_PERIOD else np.nan
        dist_ma = (close[-1] - close_sma) / close_sma if close_sma and close_sma > 0 else np.nan

        pct_b_vals[sym] = pct_b
        rsi_vals[sym] = rsi
        vol_ratio_vals[sym] = vol_ratio
        dist_ma_vals[sym] = dist_ma

    # 转为 Series
    pct_b_s = pd.Series(pct_b_vals, dtype=float)
    rsi_s = pd.Series(rsi_vals, dtype=float)
    vol_s = pd.Series(vol_ratio_vals, dtype=float)
    dist_s = pd.Series(dist_ma_vals, dtype=float)

    # ── 硬过滤器：RSI不够低 或 %B不够低 → 得分=0 ──
    mask_not_oversold = (
        (rsi_s >= cfg.OVERSOLD_RSI_THRESHOLD) | rsi_s.isna() |
        (pct_b_s >= cfg.OVERSOLD_PCT_B_THRESHOLD) | pct_b_s.isna()
    )

    # ── 分量转换（非线性，超卖越严重得分越高）──

    # %B分：-%B 越高越好（跌破下轨越多越好），但非线性
    # %B=-1.0 → score=1.5, %B=-0.5 → score=1.0, %B=0 → score=0.5
    pct_b_raw = -pct_b_s  # 负的%B
    # 将pct_b_raw映射到0-3范围，但先保留原始值用于Z-score

    # RSI分：越低越好，35以下线性递增
    # RSI=35 → score=0, RSI=15 → score=1, RSI=0 → score=1.17
    rsi_raw = (cfg.OVERSOLD_RSI_THRESHOLD - rsi_s) / cfg.OVERSOLD_RSI_THRESHOLD
    rsi_raw = rsi_raw.clip(lower=0)

    # 量能分：越缩量越好，vol_ratio=0.5 → score=0.5, vol_ratio=1.0 → score=0
    vol_raw = (1.0 - vol_s).clip(lower=0)

    # 趋势分：偏离均线越远越好（越远越可能回归）
    dist_raw = (-dist_s).clip(lower=0)  # 负偏离=超卖

    # Z-score 标准化
    def _zscore(s):
        clean = s.dropna()
        if len(clean) < 2 or clean.std() == 0:
            return pd.Series(0.0, index=s.index)
        return (s - clean.mean()) / clean.std()

    pct_b_z = _zscore(pct_b_raw.fillna(0))
    rsi_z = _zscore(rsi_raw.fillna(0))
    vol_z = _zscore(vol_raw.fillna(0))
    dist_z = _zscore(dist_raw.fillna(0))

    # 合成评分
    scores = pd.Series(0.0, index=cfg.ETF_SYMBOLS)
    for sym in cfg.ETF_SYMBOLS:
        if mask_not_oversold.get(sym, True):
            scores[sym] = 0.0
        else:
            scores[sym] = (
                cfg.WEIGHT_PCT_B * pct_b_z.get(sym, 0)
                + cfg.WEIGHT_RSI * rsi_z.get(sym, 0)
                + cfg.WEIGHT_VOLUME * vol_z.get(sym, 0)
                + cfg.WEIGHT_TREND * dist_z.get(sym, 0)
            )

    return scores


def rank_etfs_by_reversion(scores: pd.Series) -> pd.Series:
    """按均值回归评分降序排列。"""
    valid = scores[scores > 0].dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))


def compute_score_spread(scores: pd.Series) -> float:
    """得分截面标准差。"""
    valid = scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
