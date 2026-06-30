"""
复合动量信号计算模块 — 多因子复合打分系统

核心创新：将单一动量因子扩展为四维复合打分，每次打分都做
截面 Z-Score 标准化，确保各因子量纲一致。

因子构成：
  1. 趋势动量因子 (40%)  — 多时间尺度加权收益率
  2. 夏普比率因子 (25%)  — 风险调整收益
  3. 趋势质量因子 (20%)  — 均线排列层次
  4. 成交量确认因子 (15%) — 量价配合方向

打出综合分后排名，选出 TOP-1 ETF。
"""

from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    TREND_WINDOWS,
    TREND_WEIGHTS,
    FACTOR_WEIGHT_TREND,
    FACTOR_WEIGHT_SHARPE,
    FACTOR_WEIGHT_QUALITY,
    FACTOR_WEIGHT_VOLUME,
    SHARPE_WINDOW,
    QUALITY_WINDOWS,
    VOLUME_WINDOW,
    ETF_SYMBOLS,
)


def _safe_zscore(series: pd.Series) -> pd.Series:
    """安全地计算截面 Z-Score，处理全 NaN 和单值情况。"""
    clean = series.dropna()
    if len(clean) < 2 or clean.std() == 0:
        return pd.Series(0.0, index=series.index)
    mean = clean.mean()
    std = clean.std()
    return (series - mean) / std


# ═════════════════════════════════════════════════════════════════════
#  因子1: 趋势动量因子 (Trend Momentum Factor)
#  多时间尺度加权：短期(5d)做确认，中期(20d)做核心，长期(60d)做方向
# ═════════════════════════════════════════════════════════════════════

def _compute_trend_momentum(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算多时间尺度加权动量。

    对每只ETF，分别计算 short/medium/long 三个时间窗口的收益率，
    按权重加权平均，再做截面 Z-Score 标准化。
    """
    raw_scores = {}
    for sym in ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < max(TREND_WINDOWS.values()):
            raw_scores[sym] = np.nan
            continue

        weighted = 0.0
        for name, window in TREND_WINDOWS.items():
            w = TREND_WEIGHTS[name]
            # 用 close / close.shift(window) - 1
            ret = df.iloc[date_idx]["close"] / df.iloc[date_idx - window]["close"] - 1
            weighted += w * ret

        raw_scores[sym] = weighted

    return _safe_zscore(pd.Series(raw_scores, dtype=float))


# ═════════════════════════════════════════════════════════════════════
#  因子2: 夏普比率因子 (Sharpe Ratio Factor)
#  收益/波动，衡量风险调整后表现
# ═════════════════════════════════════════════════════════════════════

def _compute_sharpe_factor(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算20日夏普比率（年化）。

    Sharpe = (mean_daily_return - 0) / std_daily_return * sqrt(252)
    使用无风险利率≈0，因为ETF交易中忽略。
    """
    raw_scores = {}
    for sym in ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < SHARPE_WINDOW + 2:
            raw_scores[sym] = np.nan
            continue

        # 取过去 SHARPE_WINDOW 日的日收益率序列
        start = date_idx - SHARPE_WINDOW
        closes = df.iloc[start:date_idx + 1]["close"].values
        daily_rets = np.diff(closes) / closes[:-1]

        if len(daily_rets) < 2 or np.std(daily_rets) == 0:
            raw_scores[sym] = 0.0
        else:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
            raw_scores[sym] = sharpe

    return _safe_zscore(pd.Series(raw_scores, dtype=float))


# ═════════════════════════════════════════════════════════════════════
#  因子3: 趋势质量因子 (Trend Quality Factor)
#  价格相对于多个均线的位置，衡量趋势的"纯度"
# ═════════════════════════════════════════════════════════════════════

def _compute_trend_quality(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算趋势质量得分。

    思路：价格相对均线位置越高 → 趋势越强
    对每只ETF，计算 close/SMA_N - 1 的均值（跨多个均线周期）。
    N=5短期排列, N=20中期趋势, N=60长期趋势
    全部在均线上方=高质量趋势。
    """
    raw_scores = {}
    for sym in ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < max(QUALITY_WINDOWS):
            raw_scores[sym] = np.nan
            continue

        close = df.iloc[date_idx]["close"]
        qualities = []
        for w in QUALITY_WINDOWS:
            if date_idx >= w:
                sma = df.iloc[date_idx - w + 1:date_idx + 1]["close"].mean()
                qualities.append(close / sma - 1)

        raw_scores[sym] = np.mean(qualities) if qualities else np.nan

    return _safe_zscore(pd.Series(raw_scores, dtype=float))


# ═════════════════════════════════════════════════════════════════════
#  因子4: 成交量确认因子 (Volume Confirmation Factor)
#  量价配合度：上涨放量=强势，下跌缩量=健康调整
# ═════════════════════════════════════════════════════════════════════

def _compute_volume_confirmation(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算量价配合得分。

    核心逻辑：
      - 计算当日量比 = volume / MA20_volume
      - 计算当日收益率方向 = sign(pct_chg)
      - 得分 = 量比 × 方向
      - 上涨(+) × 放量(+) = 正分（健康上涨）
      - 下跌(-) × 缩量(量比<1, 取绝对值负数) = 正分（健康调整）
      - 上涨(+) × 缩量(-) = 负分（上涨乏力）
      - 下跌(-) × 放量(+) = 负分（恐慌下跌）
    """
    raw_scores = {}
    for sym in ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < VOLUME_WINDOW + 1:
            raw_scores[sym] = np.nan
            continue

        volume = df.iloc[date_idx]["volume"]
        vol_ma20 = df.iloc[date_idx - VOLUME_WINDOW + 1:date_idx + 1]["volume"].mean()

        if vol_ma20 <= 0:
            raw_scores[sym] = 0.0
            continue

        vol_ratio = volume / vol_ma20           # 量比
        pct_chg = df.iloc[date_idx].get("pct_chg", 0)

        # 方向得分：正收益=+1，负收益=-1，平=0
        direction = 1 if pct_chg > 0 else (-1 if pct_chg < 0 else 0)

        # 综合得分 = (量比-1) × direction
        #  上涨放量: (+)*(+)=+  上涨缩量: (-)*(+)=-
        #  下跌放量: (+)*(-)=-  下跌缩量: (-)*(-)=+
        score = (vol_ratio - 1) * direction
        raw_scores[sym] = score

    return _safe_zscore(pd.Series(raw_scores, dtype=float))


# ═════════════════════════════════════════════════════════════════════
#  市场状态判断（用沪深300指数）
# ═════════════════════════════════════════════════════════════════════

def judge_market_regime(
    index_data: pd.DataFrame,
    date_idx: int,
    ma_period: int = 60,
    bear_threshold: float = -0.02,
) -> dict:
    """
    判断当前市场状态：牛市/震荡/熊市

    用沪深300收盘价与 MA60 的关系判断：
      - close > MA60 * (1 + 2%) → 牛市（进取）
      - close < MA60 * (1 - 2%) → 熊市（防守）
      - 其他 → 震荡（中性）

    返回 dict：
      regime: "bull" / "bear" / "neutral"
      ma_value: MA60 值
      ratio: close / MA60 - 1
    """
    if index_data is None or index_data.empty:
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}

    df = index_data
    if date_idx < ma_period or date_idx >= len(df):
        return {"regime": "neutral", "ma_value": 0, "ratio": 0}

    close = df.iloc[date_idx]["close"]
    ma = df.iloc[date_idx - ma_period + 1:date_idx + 1]["close"].mean()
    ratio = close / ma - 1

    if ratio > abs(bear_threshold):
        regime = "bull"
    elif ratio < bear_threshold:
        regime = "bear"
    else:
        regime = "neutral"

    return {"regime": regime, "ma_value": ma, "ratio": ratio}


# ═════════════════════════════════════════════════════════════════════
#  综合打分主函数
# ═════════════════════════════════════════════════════════════════════

def compute_composite_score(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
    index_data: pd.DataFrame | None = None,
    market_ma_period: int = 60,
) -> pd.Series:
    """
    计算所有 ETF 的综合复合得分。

    Parameters
    ----------
    etf_data : dict
        {symbol: DataFrame}，每只ETF的日线数据
    date_idx : int
        当前日期索引
    index_data : DataFrame or None
        沪深300指数数据，用于市场状态判断
    market_ma_period : int
        市场状态判断MA周期

    Returns
    -------
    pd.Series
        index = ETF代码, values = 综合得分
    """
    # 1. 四项因子打分
    trend_scores = _compute_trend_momentum(etf_data, date_idx)
    sharpe_scores = _compute_sharpe_factor(etf_data, date_idx)
    quality_scores = _compute_trend_quality(etf_data, date_idx)
    volume_scores = _compute_volume_confirmation(etf_data, date_idx)

    # 2. 加权合成（各因子已 Z-Score 标准化）
    composite = (
        FACTOR_WEIGHT_TREND * trend_scores
        + FACTOR_WEIGHT_SHARPE * sharpe_scores
        + FACTOR_WEIGHT_QUALITY * quality_scores
        + FACTOR_WEIGHT_VOLUME * volume_scores
    )

    # 3. 市场状态调整：如果是熊市，额外压低综合得分
    regime_info = judge_market_regime(index_data, date_idx, market_ma_period)
    if regime_info["regime"] == "bear":
        # 熊市打八折，更保守
        composite *= 0.8
    elif regime_info["regime"] == "bull":
        # 牛市上浮10%，更进取
        composite *= 1.1

    return composite


# ═════════════════════════════════════════════════════════════════════
#  排名函数
# ═════════════════════════════════════════════════════════════════════

def rank_etfs_by_composite(composite_scores: pd.Series) -> pd.Series:
    """
    按综合得分降序排列 ETF。

    Returns
    -------
    pd.Series
        index = 排名（1=最佳）, values = ETF代码
    """
    valid = composite_scores.dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))


def compute_composite_spread(
    composite_scores: pd.Series,
) -> float:
    """
    计算当日综合得分的截面标准差，衡量因子分化程度。

    标准差大 → ETF分化明显（选股意义大）
    标准差小 → 各ETF走势趋同（选股意义小）
    """
    valid = composite_scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
