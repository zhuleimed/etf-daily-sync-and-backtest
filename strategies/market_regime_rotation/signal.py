"""
动量信号计算模块

职责：
  1. 每日计算每只 ETF 的 N 日动量值（N日收益率）
  2. 按动量值降序排序，确定排名
  3. 支持动态动量窗口（根据市场分化程度自动选择短/长窗口）
  4. 支持 TOP-N 持仓（持有排名前N只ETF）

对齐策略原文：
  - 严格数值排序，不加入主观权重
  - 默认 N=15 个交易日
  - 以当日收盘价为基准计算
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_momentum_signals(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """
    计算在指定日期的所有 ETF 动量值。

    Parameters
    ----------
    etf_data : dict
        {symbol: DataFrame}，每只 ETF 的数据
    date_idx : int
        当前日期在 DataFrame 中的行索引
    momentum_window : int
        动量计算窗口

    Returns
    -------
    pd.Series
        index = ETF 代码, values = 动量值
        NaN 表示动量值尚不可用（数据不足）
    """
    momentums = {}
    for sym, df in etf_data.items():
        if date_idx < momentum_window:
            # 数据不足以计算 momentum，返回 NaN
            momentums[sym] = np.nan
        else:
            val = df.loc[date_idx, "momentum"]
            momentums[sym] = val if not pd.isna(val) else np.nan

    return pd.Series(momentums)


def rank_etfs_by_momentum(momentum_series: pd.Series) -> pd.Series:
    """
    按动量值降序排列 ETF。

    Returns
    -------
    pd.Series
        index = 排名（1=最强）, values = ETF 代码
    """
    # 只对有有效动量值的 ETF 排序
    valid = momentum_series.dropna()
    if valid.empty:
        return pd.Series(dtype=str)

    sorted_etfs = valid.sort_values(ascending=False)
    # 返回排名 Series: {1:最强ETF代码, 2:次强ETF代码, ...}
    return pd.Series(sorted_etfs.index.values, index=range(1, len(sorted_etfs) + 1))


def compute_momentum_spread(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
) -> float:
    """
    计算当日各ETF动量值的标准差，衡量市场分化程度。

    标准差大 → 分化市（各ETF走势差异明显）
    标准差小 → 同步市（各ETF走势一致）

    Returns
    -------
    float : 各ETF momentum 值的标准差
    """
    vals = []
    for sym, df in etf_data.items():
        if date_idx < 15:
            continue
        v = df.loc[date_idx, "momentum"]
        if not pd.isna(v):
            vals.append(v)
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def get_active_momentum_column(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    threshold: float = 0.03,
    short_col: str = "momentum_10",
    long_col: str = "momentum_20",
) -> str:
    """
    根据市场分化程度动态选择动量列名。

    Parameters
    ----------
    threshold : float
        动量标准差阈值：
        - 实际标准差 > threshold → 分化市 → 使用短窗口(10日)
        - 实际标准差 ≤ threshold → 同步市 → 使用长窗口(20日)

    Returns
    -------
    str : 选中的动量列名 ("momentum_10" 或 "momentum_20")
    """
    spread = compute_momentum_spread(etf_data, date_idx)
    col = short_col if spread > threshold else long_col
    return col


def compute_momentum_signals_dynamic(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    threshold: float = 0.03,
) -> pd.Series:
    """
    计算动态动量信号——根据市场分化程度自动选择窗口。

    等同于 compute_momentum_signals，但列名由 get_active_momentum_column 决定。
    """
    col = get_active_momentum_column(etf_data, date_idx, threshold)
    momentums = {}
    for sym, df in etf_data.items():
        if date_idx < 15:
            momentums[sym] = np.nan
        else:
            val = df.loc[date_idx, col]
            momentums[sym] = val if not pd.isna(val) else np.nan
    return pd.Series(momentums)


def determine_signals(
    current_holding: Optional[str],
    target_etf: Optional[str],
    momentum_series: pd.Series,
    current_holding_momentum: float,
    target_momentum: float,
    friction_cost: float,
) -> Tuple[int, Optional[str], str]:
    """
    核心决策逻辑：判断是否执行调仓。

    规则（对齐策略原文 3.3.3）：
      1. 无持仓 → 若最强标的动量 > 0，开仓买入（信号=1）
      2. 有持仓且最强标的不同于当前持仓 →
         只有在「预期超额收益 > 双边摩擦成本」时才调仓
      3. 有持仓且最强标的=当前持仓 → 继续持有（信号=0）

    Parameters
    ----------
    current_holding : str or None
        当前持仓的 ETF 代码，None 表示空仓
    target_etf : str or None
        当前最强标的 ETF 代码
    momentum_series : pd.Series
        各 ETF 的动量值
    current_holding_momentum : float
        当前持仓 ETF 的动量值
    target_momentum : float
        最强标的 ETF 的动量值
    friction_cost : float
        双边摩擦成本（切换总成本）

    Returns
    -------
    signal : int
        1=买入/调仓, -1=卖出, 0=持有
    target : str or None
        需要买入/切换到的 ETF 代码
    reason : str
        信号原因描述，用于日志
    """
    # 无有效动量信号时保持不动
    if target_etf is None or pd.isna(target_momentum):
        return 0, current_holding, "动量信号无效"

    # ---- 情况 A：无持仓 ----
    if current_holding is None:
        if target_momentum > 0:
            # 最强标的有正动量 → 开仓
            return 1, target_etf, f"开仓买入 {target_etf}（动量={target_momentum:.4f}>0）"
        else:
            # 全市场动量均为负 → 空仓避险
            return 0, None, f"全市场动量非正，空仓观望（最强={target_etf}动量={target_momentum:.4f}）"

    # ---- 情况 B：有持仓，且最强标的即当前持仓 ----
    if target_etf == current_holding:
        return 0, current_holding, f"当前持仓 {current_holding} 即为最强标的，继续持有"

    # ---- 情况 C：有持仓，且最强标的 ≠ 当前持仓 ----
    excess_return = target_momentum - current_holding_momentum

    if excess_return > friction_cost:
        # 预期超额收益 > 摩擦成本 → 切换
        return 1, target_etf, (
            f"切换 {current_holding}→{target_etf}："
            f"超额收益={excess_return:.4f} > 摩擦成本={friction_cost:.4f}"
        )
    else:
        # 超额收益不足以覆盖成本 → 维持
        return 0, current_holding, (
            f"维持 {current_holding}：超额收益={excess_return:.4f} "
            f"≤ 摩擦成本={friction_cost:.4f}"
        )
