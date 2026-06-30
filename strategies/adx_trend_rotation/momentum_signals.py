"""
ADX 趋势强度信号计算模块

核心指标：ADX (Average Directional Index) — J. Welles Wilder 经典趋势强度指标。

计算步骤：
  1. TR = max(H-L, |H-pC|, |L-pC|)   ← True Range
  2. +DM / -DM                          ← Directional Movement
  3. Wilder 平滑: TR_S, +DM_S, -DM_S   ← SMMA(period)
  4. DI+ = +DM_S/TR_S, DI- = -DM_S/TR_S
  5. DX = |DI+ - DI-| / (DI+ + DI-)
  6. ADX = SMMA(DX, period)             ← 最终趋势强度

策略逻辑：
  - ADX > threshold: 趋势存在，可以操作
  - DI+ > DI-: 多头主导，做多
  - DI- > DI+: 空头主导，回避
  - 综合得分 = ADX × 0.5 + (DI+ - DI-) × 0.3 + 短期动量 × 0.2
"""

import numpy as np
import pandas as pd

from . import config as cfg  # 动态引用，支持运行时参数修改


# ═════════════════════════════════════════════════════════════════════
#  ADX 核心计算
# ═════════════════════════════════════════════════════════════════════

def _wilder_smma(values: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder 平滑移动平均（SMMA）。

    与简单 SMA 不同，Wilder SMMA 用前值递推：
      SMMA[0] = sum(values[:period]) / period
      SMMA[i] = (SMMA[i-1] * (period-1) + values[i+period-1]) / period
    """
    n = len(values)
    if n < period + 1:
        return np.full(n, np.nan)

    result = np.full(n, np.nan)
    # 第一个值：前 period 个的均值
    result[period - 1] = np.mean(values[:period])
    # 递推
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period
    return result


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> dict:
    """
    计算 ADX、DI+、DI-。

    Parameters
    ----------
    high, low, close : np.ndarray
        价格序列（需等长）。
    period : int
        ADX 计算周期（Wilder 标准 14）。

    Returns
    -------
    dict with keys: 'adx', 'di_plus', 'di_minus', 'dx'
        每个值为 np.ndarray，长度与输入一致，前 period*2 个为 NaN。
    """
    n = len(high)
    if n < period * 2:
        return {"adx": np.full(n, np.nan), "di_plus": np.full(n, np.nan),
                "di_minus": np.full(n, np.nan), "dx": np.full(n, np.nan)}

    # 1. True Range
    tr = np.full(n, np.nan)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # 2. Directional Movement
    dm_plus = np.full(n, np.nan)
    dm_minus = np.full(n, np.nan)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0.0

        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0.0

    # 3. Wilder 平滑
    tr_smooth = _wilder_smma(np.nan_to_num(tr, 0), period)
    dm_plus_smooth = _wilder_smma(np.nan_to_num(dm_plus, 0), period)
    dm_minus_smooth = _wilder_smma(np.nan_to_num(dm_minus, 0), period)

    # 4. DI+ / DI-
    di_plus = np.full(n, np.nan)
    di_minus = np.full(n, np.nan)
    for i in range(n):
        if tr_smooth[i] > 0 and not np.isnan(tr_smooth[i]):
            di_plus[i] = 100.0 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100.0 * dm_minus_smooth[i] / tr_smooth[i]

    # 5. DX
    dx = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum

    # 6. ADX = SMMA of DX
    adx = _wilder_smma(np.nan_to_num(dx, 0), period)

    return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus, "dx": dx}


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
#  ADX 综合评分
# ═════════════════════════════════════════════════════════════════════

def compute_adx_scores(
    etf_data: dict[str, pd.DataFrame],
    date_idx: int,
) -> pd.Series:
    """
    计算每只 ETF 的 ADX 综合评分。

    评分逻辑：
      1. ADX < ADX_MIN_STRENGTH → 得分 = 0（趋势不够强，不参与）
      2. DI- > DI+ → 得分 = 0（空头主导，回避）
      3. 评分 = ADX_z × ADX_WEIGHT + DI_advantage_z × DI_WEIGHT + mom_z × MOM_WEIGHT

    Returns
    -------
    pd.Series: index=ETF代码, values=综合得分（<=0 表示不满足条件）
    """
    needed = cfg.ADX_PERIOD * 2 + 5  # 至少需要 2×period 个数据点
    adx_values = {}
    di_plus_vals = {}
    di_minus_vals = {}
    mom_5d = {}

    for sym in cfg.ETF_SYMBOLS:
        df = etf_data.get(sym)
        if df is None or date_idx < needed or date_idx >= len(df):
            adx_values[sym] = np.nan
            continue

        high = df["high"].values[:date_idx + 1]
        low = df["low"].values[:date_idx + 1]
        close = df["close"].values[:date_idx + 1]

        result = compute_adx(high, low, close, cfg.ADX_PERIOD)
        adx_val = result["adx"][date_idx]
        dip = result["di_plus"][date_idx]
        dim = result["di_minus"][date_idx]

        # 5日动量
        mom = close[date_idx] / close[date_idx - 5] - 1 if date_idx >= 5 else np.nan

        adx_values[sym] = adx_val
        di_plus_vals[sym] = dip
        di_minus_vals[sym] = dim
        mom_5d[sym] = mom

    # 转为 Series
    adx_s = pd.Series(adx_values, dtype=float)
    dip_s = pd.Series(di_plus_vals, dtype=float)
    dim_s = pd.Series(di_minus_vals, dtype=float)
    mom_s = pd.Series(mom_5d, dtype=float)

    # 过滤：ADX 不够强 or 空头主导 → 得分为 0
    mask_weak = (adx_s < cfg.ADX_MIN_STRENGTH) | adx_s.isna()
    mask_bear = (dim_s > dip_s) | dip_s.isna() | dim_s.isna()

    # DI 优势分
    di_advantage = dip_s - dim_s

    # Z-Score 标准化
    def _zscore(s):
        clean = s.dropna()
        if len(clean) < 2 or clean.std() == 0:
            return pd.Series(0.0, index=s.index)
        return (s - clean.mean()) / clean.std()

    adx_z = _zscore(adx_s.fillna(0))
    di_z = _zscore(di_advantage.fillna(0))
    mom_z = _zscore(mom_s.fillna(0))

    # 合成评分
    scores = pd.Series(0.0, index=cfg.ETF_SYMBOLS)
    for sym in cfg.ETF_SYMBOLS:
        if mask_weak.get(sym, True) or mask_bear.get(sym, True):
            scores[sym] = 0.0
        else:
            scores[sym] = (
                cfg.ADX_WEIGHT * adx_z.get(sym, 0)
                + cfg.DI_WEIGHT * di_z.get(sym, 0)
                + cfg.MOM_WEIGHT * mom_z.get(sym, 0)
            )

    return scores


def rank_etfs_by_adx(adx_scores: pd.Series) -> pd.Series:
    """按 ADX 综合评分降序排列。"""
    valid = adx_scores[adx_scores > 0].dropna()  # 只考虑得分>0的标的
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_scores = valid.sort_values(ascending=False)
    return pd.Series(sorted_scores.index.values, index=range(1, len(sorted_scores) + 1))


def compute_adx_spread(adx_scores: pd.Series) -> float:
    """ADX 得分截面标准差。"""
    valid = adx_scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
