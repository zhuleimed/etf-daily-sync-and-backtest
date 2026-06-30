"""
RSI 趋势确认信号计算模块

核心指标：RSI (Relative Strength Index) — J. Welles Wilder 经典动量振荡器。

计算步骤（Wilder 平滑 RSI）：
  1. 计算每日价格涨跌：gain = max(close - prev_close, 0), loss = max(prev_close - close, 0)
  2. Wilder 平滑平均 gain/loss（SMMA 递推）
  3. RS = Avg Gain / Avg Loss
  4. RSI = 100 - 100 / (1 + RS)

策略用法（趋势跟踪，非均值回复）：
  - RSI > 50：多头趋势，ETF 处于健康上升状态（通过 RSI_BULL_THRESHOLD 过滤）
  - RSI 值越高：内部动量越强 → 得分越高
  - RSI 上升（斜率 > 0）：趋势加速 → 加分
  - RSI < 50：空头/弱势 → 得分直接归零（不参与）

⚠ v2 改进：采用非截面评分方式，解决7只ETF高相关性问题。
   - 每只ETF独立评分（基于自身RSI状态），而非跨ETF Z-score比较
   - 评分 = RSI_persistence（RSI>50连续天数占比）+ RSI_regime（当前RSI区间得分）
   - 这种方法在高度相关的标的中表现得更好

评分体系（每只ETF独立评分）：
  - RSI Regime: RSI > 60 = 2分, 50-60 = 1分, < 50 = 0分
  - RSI Persistence: 过去N天RSI>50的占比（0-1），乘以2
  - RSI Slope: 过去5天RSI变化量，正向加分（最高0.5）
  - 总分 0-4.5 区间
"""

import numpy as np
import pandas as pd

from . import config as cfg  # 动态引用，支持运行时参数修改


# ═════════════════════════════════════════════════════════════════════
#  RSI 核心计算
# ═════════════════════════════════════════════════════════════════════

def compute_rsi(
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Wilder 平滑 RSI 计算。

    使用 Wilder 的递推平滑方法（SMMA 风格），与通达信/同花顺 RSI 一致。

    Parameters
    ----------
    close : np.ndarray
        收盘价序列。
    period : int
        RSI 计算周期（Wilder 标准 14）。

    Returns
    -------
    np.ndarray
        RSI 值数组，前 period 个为 NaN。
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)

    # 1. 计算每日涨跌
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # 2. Wilder 平滑
    # 初始值：前 period 个的简单均值
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    # 递推：Avg(i) = (Avg(i-1) * (period-1) + value(i)) / period
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    # 3. RSI
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0  # 连续上涨，RSI=100
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


# ═════════════════════════════════════════════════════════════════════
#  RSI 综合评分（v2 — 非截面，每只ETF独立评分）
# ═════════════════════════════════════════════════════════════════════

def compute_rsi_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算每只 ETF 的 RSI 综合评分（v2 非截面评分）。

    每只 ETF 独立评分（不与其它 ETF 比较），基于自身 RSI 绝对状态。

    评分分量（每只ETF独立计算）：
      1. RSI Regime Score (0-2): RSI > 60 = 2, 50-60 = 1, < 50 = 0
      2. RSI Persistence Score (0-2): 过去10天中RSI>50的天数占比 × 2
      3. RSI Slope Score (0-0.5): 过去5天RSI变化量/10，上限0.5

    总分范围 0-4.5，得分 > 2.0 才考虑买入。

    Parameters
    ----------
    etf_data : dict[str, pd.DataFrame]
        {symbol: df} 格式，每只 ETF 必须有 close 列。
    date_idx : int
        当前日期在 DataFrame 中的索引位置。

    Returns
    -------
    pd.Series: index=ETF代码, values=综合得分（不在候选区间的归零）
    """
    needed = cfg.RSI_PERIOD + cfg.RSI_SLOPE_PERIOD + 15  # 需要足够缓冲
    scores = {}

    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < needed or date_idx >= len(df):
            scores[sym] = 0.0
            continue

        close = df["close"].values[:date_idx + 1]

        # RSI(14) 计算
        rsi_arr = compute_rsi(close, cfg.RSI_PERIOD)
        rsi_val = rsi_arr[date_idx]
        if np.isnan(rsi_val):
            scores[sym] = 0.0
            continue

        # ── Component 1: RSI Regime Score (0-2) ──
        rsi_bull = getattr(cfg, 'RSI_BULL_THRESHOLD', 50)
        if rsi_val > 60:
            regime_score = 2.0
        elif rsi_val > rsi_bull:
            regime_score = 1.0
        else:
            regime_score = 0.0

        # ── Component 2: RSI Persistence Score (0-2) ──
        # 过去 PERSISTENCE_WINDOW 天中 RSI > RSI_BULL_THRESHOLD 的占比 × 2
        persistence_window = getattr(cfg, 'RSI_PERSISTENCE_WINDOW', 10)
        lookback_start = max(cfg.RSI_PERIOD, date_idx - persistence_window + 1)
        if lookback_start < date_idx:
            rsi_window = rsi_arr[lookback_start:date_idx + 1]
            above_threshold = np.nansum(rsi_window > rsi_bull)
            total_valid = np.sum(~np.isnan(rsi_window))
            ratio = above_threshold / total_valid if total_valid > 0 else 0
            persistence_score = ratio * 2.0  # 0-2 scale
        else:
            persistence_score = 0.0

        # ── Component 3: RSI Slope Boost (0-0.5) ──
        slope_start = max(cfg.RSI_PERIOD + 1, date_idx - cfg.RSI_SLOPE_PERIOD)
        if slope_start > 0 and slope_start < date_idx and not np.isnan(rsi_arr[slope_start]):
            rsi_change = rsi_val - rsi_arr[slope_start]
            slope_score = min(max(rsi_change / 10.0, 0), 0.5)  # 每10点RSI变化给0.5分，上限0.5
        else:
            slope_score = 0.0

        # ── Total Score ──
        total = regime_score + persistence_score + slope_score

        # 最低得分门槛：低于一定值的归零（避免弱信号交易）
        min_score = getattr(cfg, 'RSI_MIN_SCORE', 2.0)
        scores[sym] = total if total >= min_score else 0.0

    return pd.Series(scores, dtype=float)


def rank_etfs_by_rsi(rsi_scores: pd.Series) -> pd.Series:
    """按 RSI 综合评分降序排列。"""
    valid = rsi_scores[rsi_scores > 0].dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))


def compute_rsi_spread(rsi_scores: pd.Series) -> float:
    """RSI 得分截面标准差。"""
    valid = rsi_scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
