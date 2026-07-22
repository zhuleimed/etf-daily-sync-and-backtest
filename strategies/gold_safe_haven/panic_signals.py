"""
恐慌指数信号计算模块（策略核心创新）

职责：
  1. 每日计算三个恐慌子指标
  2. 用滚动Z-score标准化并加权合成恐慌分
  3. 判断是否触发恐慌模式

恐慌子指标：
  1. max_5d_drawdown：7只宽基ETF的5日收益率最小值（跌最惨的那只跌了多少）
  2. avg_vol_ratio：7只宽基ETF的平均波动率比（21d年化vol / 63d年化vol）
  3. breadth：7只宽基ETF中5日收益为负的比例（0~1）

合成方式：
  Z-score标准化（滚动252日窗口）→ 加权求和（0.4/0.3/0.3）

注意：Z-score使用滚动窗口计算，避免 look-ahead bias。
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import (
    BROAD_SYMBOLS, GOLD_SYMBOL,
    PANIC_WEIGHT_MAX_DD, PANIC_WEIGHT_VOL, PANIC_WEIGHT_BREADTH,
    PANIC_THRESHOLD, ZSCORE_WINDOW,
)


def compute_panic_indicators(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
) -> Tuple[float, float, float]:
    """
    计算三个恐慌子指标的原始值。

    Parameters
    ----------
    etf_data : dict
        {symbol: DataFrame}，含 ret_5d, vol_ratio 列
    date_idx : int
        当前日期行索引

    Returns
    -------
    max_dd : float
        宽基ETF中最大的5日跌幅（最负的ret_5d），如 -0.08 表示跌了8%
    avg_vol_ratio : float
        宽基ETF的平均波动率比
    breadth : float
        宽基ETF中5日收益为负的比例（0~1）
    """
    rets_5d = []
    vol_ratios = []
    down_count = 0
    total_broad = 0

    for sym in BROAD_SYMBOLS:
        if sym not in etf_data:
            continue
        df = etf_data[sym]
        try:
            ret_5d = df.loc[date_idx, "ret_5d"]
            vol_r = df.loc[date_idx, "vol_ratio"]
        except (KeyError, IndexError):
            continue

        if pd.isna(ret_5d) or pd.isna(vol_r):
            continue

        total_broad += 1
        rets_5d.append(ret_5d)
        vol_ratios.append(vol_r)
        if ret_5d < 0:
            down_count += 1

    if total_broad == 0:
        return 0.0, 1.0, 0.0

    max_dd = min(rets_5d)  # 最负的那个（跌最多的）
    avg_vol_ratio = np.mean(vol_ratios)
    breadth = down_count / total_broad

    return max_dd, avg_vol_ratio, breadth


def _rolling_zscore(series: np.ndarray, idx: int, window: int = ZSCORE_WINDOW) -> float:
    """
    计算某个值在其滚动窗口历史中的Z-score。

    使用 idx 之前（不含）的 window 个值作为历史样本。
    这样确保不使用未来数据（look-ahead bias free）。

    Parameters
    ----------
    series : np.ndarray
        完整的历史序列
    idx : int
        当前位置（用 idx-window 到 idx-1 的数据计算均值和标准差）
    window : int
        滚动窗口大小

    Returns
    -------
    zscore : float
        如果历史数据不足，返回 0.0
    """
    start = max(0, idx - window)
    end = idx
    history = series[start:end]

    if len(history) < 20:  # 至少需要20个样本才有统计意义
        return 0.0

    mean = np.mean(history)
    std = np.std(history, ddof=1)
    if std < 1e-10:
        return 0.0

    return (series[idx] - mean) / std


def compute_panic_score(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    panic_history: Dict[str, list],
) -> Dict:
    """
    计算恐慌综合分并判断是否触发恐慌模式。

    Parameters
    ----------
    etf_data : dict
        {symbol: DataFrame}
    date_idx : int
        当前日期行索引
    panic_history : dict
        历史恐慌指标记录（用于滚动Z-score计算）
        {"max_dd": [...], "avg_vol_ratio": [...], "breadth": [...], "panic_score": [...]}

    Returns
    -------
    result : dict
        {
            "max_dd": float,           # 最大5日跌幅原始值
            "avg_vol_ratio": float,     # 平均波动率比原始值
            "breadth": float,           # 下跌广度原始值
            "z_max_dd": float,          # Z-score标准化后
            "z_vol_ratio": float,
            "z_breadth": float,
            "panic_score": float,       # 综合恐慌分
            "is_panic": bool,           # 是否触发恐慌模式
        }
    """
    # 1. 计算原始值
    max_dd, avg_vol_ratio, breadth = compute_panic_indicators(etf_data, date_idx)

    # 2. 追加到历史序列
    panic_history["max_dd_raw"].append(max_dd)
    panic_history["vol_ratio_raw"].append(avg_vol_ratio)
    panic_history["breadth_raw"].append(breadth)

    dd_arr = np.array(panic_history["max_dd_raw"])
    vol_arr = np.array(panic_history["vol_ratio_raw"])
    brd_arr = np.array(panic_history["breadth_raw"])

    current_idx = len(dd_arr) - 1

    # 3. 滚动Z-score标准化
    # 注意：max_dd 是负值（跌幅），Z-score也是负的。恐慌时 max_dd 越负 → Z越小
    # 我们取 -Z 使其变成正值（恐慌越大分数越高）
    z_dd = -_rolling_zscore(dd_arr, current_idx, ZSCORE_WINDOW)
    # 波动率比 > 1 表示波动率上升，Z越大表示越恐慌
    z_vol = _rolling_zscore(vol_arr, current_idx, ZSCORE_WINDOW)
    # breadth 越大（越多的ETF在跌）越恐慌
    z_brd = _rolling_zscore(brd_arr, current_idx, ZSCORE_WINDOW)

    # 4. 加权合成
    panic_score = (
        z_dd * PANIC_WEIGHT_MAX_DD
        + z_vol * PANIC_WEIGHT_VOL
        + z_brd * PANIC_WEIGHT_BREADTH
    )

    panic_history["panic_score_raw"].append(panic_score)

    # 5. 判断触发
    is_panic = panic_score > PANIC_THRESHOLD

    return {
        "max_dd": max_dd,
        "avg_vol_ratio": avg_vol_ratio,
        "breadth": breadth,
        "z_max_dd": round(z_dd, 3),
        "z_vol_ratio": round(z_vol, 3),
        "z_breadth": round(z_brd, 3),
        "panic_score": round(panic_score, 3),
        "is_panic": is_panic,
    }


def init_panic_history() -> Dict[str, list]:
    """初始化恐慌历史记录字典。"""
    return {
        "max_dd_raw": [],
        "vol_ratio_raw": [],
        "breadth_raw": [],
        "panic_score_raw": [],
    }


def get_broad_avg_5d_return(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
) -> float:
    """
    计算宽基ETF的平均5日收益率，用于判断恐慌是否解除。

    Returns
    -------
    float : 平均5日收益率（如 -0.03 表示平均跌3%）
    """
    rets = []
    for sym in BROAD_SYMBOLS:
        if sym not in etf_data:
            continue
        try:
            r = etf_data[sym].loc[date_idx, "ret_5d"]
        except (KeyError, IndexError):
            continue
        if not pd.isna(r):
            rets.append(r)

    if not rets:
        return 0.0
    return float(np.mean(rets))


# ============================================================================
# 硬阈值恐慌判断（不需要Z-score，直接用绝对值比较）
# ============================================================================

def compute_panic_hard(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    dd_threshold: float = -0.04,
    vol_threshold: float = 1.2,
    breadth_threshold: float = 0.7,
) -> Dict:
    """
    硬阈值恐慌判断（替代Z-score复合分）。

    三个条件必须同时满足才触发恐慌：
      1. max_5d_dd < dd_threshold（如 -0.04 = -4%）
      2. avg_vol_ratio > vol_threshold（如 1.2）
      3. breadth > breadth_threshold（如 0.7 = 70%ETF在跌）

    优点：物理含义明确，不受滚动窗口历史波动影响。
    """
    max_dd, avg_vol_ratio, breadth = compute_panic_indicators(etf_data, date_idx)

    is_panic = (
        max_dd < dd_threshold
        and avg_vol_ratio > vol_threshold
        and breadth > breadth_threshold
    )

    return {
        "max_dd": max_dd,
        "avg_vol_ratio": avg_vol_ratio,
        "breadth": breadth,
        "is_panic": is_panic,
        "dd_threshold": dd_threshold,
        "vol_threshold": vol_threshold,
        "breadth_threshold": breadth_threshold,
        "dd_ok": max_dd < dd_threshold,
        "vol_ok": avg_vol_ratio > vol_threshold,
        "breadth_ok": breadth > breadth_threshold,
    }


# ============================================================================
# 牛市过滤器
# ============================================================================

def load_hs300_data(db_path: str = "data/etf_daily.db") -> pd.DataFrame:
    """从index_daily表加载沪深300指数日线数据。"""
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol='000300' ORDER BY date",
            conn, parse_dates=["date"],
        )
    if df.empty:
        raise ValueError("沪深300指数数据不存在")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ma"] = df["close"].rolling(window=60, min_periods=1).mean()
    return df


def is_bull_market(
    hs300_df: pd.DataFrame,
    date_idx: int,
    ma_period: int = 60,
) -> bool:
    """
    判断是否处于牛市（沪深300高于其N日均线）。

    牛市 → 忽略恐慌信号，坚持动量策略。
    熊市 → 允许恐慌触发。

    Returns
    -------
    True if bull market (HS300 > MA), False otherwise.
    """
    if date_idx >= len(hs300_df):
        return True  # 数据不足时默认为牛市（安全侧）

    close = hs300_df.iloc[date_idx]["close"]
    ma_val = hs300_df.iloc[date_idx]["ma"]

    if pd.isna(close) or pd.isna(ma_val) or ma_val <= 0:
        return True

    return close > ma_val


def check_bull_market_from_data(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    ma_period: int = 60,
) -> bool:
    """
    从已加载的etf_data中以510300(沪深300ETF)为代理判断牛熊。

    使用510300作为代理，无需额外加载index_daily数据。
    注意：510300跟踪沪深300，价格约是沪深300指数的1/1000，但cross/above关系一致。
    """
    proxy_sym = "510300"
    if proxy_sym not in etf_data:
        return True  # 数据不足默认牛市

    df = etf_data[proxy_sym]
    if date_idx >= len(df) or date_idx < ma_period:
        return True

    close = float(df.loc[date_idx, "close"])
    ma_val = float(df["close"].iloc[max(0, date_idx - ma_period):date_idx + 1].mean())

    if pd.isna(close) or pd.isna(ma_val) or ma_val <= 0:
        return True

    return close > ma_val
