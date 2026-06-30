"""
市场状态识别模块

将市场划分为三种状态，每种状态对应不同的交易策略：

  牛市（BULL）:   HS300 > MA60 + 3%   → 动量模式
  震荡（NEUTRAL）: MA60±3%之间          → 均值回归模式
  熊市（BEAR）:    HS300 < MA60 - 3%   → 空仓模式

检测指标：
  1. MA60位置比（close/MA60 - 1） — 主要判断依据
  2. MA60斜率（过去20天MA的变化） — 辅助确认趋势方向
"""

import numpy as np
import pandas as pd

from . import config as cfg


def detect_regime(
    index_data: pd.DataFrame,
    date_idx: int,
) -> dict:
    """
    检测当前市场状态。

    Parameters
    ----------
    index_data : pd.DataFrame
        沪深300指数日线数据（必须有 close 列）。
    date_idx : int
        当前日期索引。

    Returns
    -------
    dict with keys:
      - regime: 'bull' | 'neutral' | 'bear'
      - ratio: close/MA60 - 1
      - ma_slope: MA60 20日变化率（%）
      - ma_value: MA60当前值
    """
    result = {
        "regime": "neutral",
        "ratio": 0.0,
        "ma_slope": 0.0,
        "ma_value": 0.0,
    }

    if index_data is None or index_data.empty:
        return result

    if date_idx < cfg.REGIME_MA_PERIOD or date_idx >= len(index_data):
        return result

    close = index_data.iloc[date_idx]["close"]
    ma60 = index_data.iloc[date_idx - cfg.REGIME_MA_PERIOD + 1:date_idx + 1]["close"].mean()
    ratio = close / ma60 - 1

    # MA60斜率（过去20天MA的变化率）
    if date_idx >= cfg.REGIME_MA_PERIOD + 20:
        ma60_prev = index_data.iloc[date_idx - 20 - cfg.REGIME_MA_PERIOD + 1:date_idx - 20 + 1]["close"].mean()
        ma_slope = ma60 / ma60_prev - 1 if ma60_prev > 0 else 0.0
    else:
        ma_slope = 0.0

    result["ratio"] = ratio
    result["ma_slope"] = ma_slope
    result["ma_value"] = ma60

    if ratio > cfg.BULL_THRESHOLD:
        result["regime"] = "bull"
    elif ratio < cfg.BEAR_THRESHOLD:
        result["regime"] = "bear"
    else:
        result["regime"] = "neutral"

    return result


def regime_description(regime: str) -> str:
    """返回状态的中文描述。"""
    desc = {
        "bull": "📈 牛市（动量模式）",
        "neutral": "⚖️ 震荡（均值回归模式）",
        "bear": "📉 熊市（空仓模式）",
    }
    return desc.get(regime, regime)
