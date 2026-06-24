"""
数据加载模块

职责：
  1. 从 SQLite 数据库加载 5 只宽基 ETF 日线数据
  2. 日期对齐（取共同交易日）
  3. 计算辅助列：收益率、ATR、20日均成交额、N日动量
  4. 加载基准指数沪深300数据
  5. 计算等权组合基准收益率

数据来源：
  - ETF 日线：etf_daily 表（symbol/date/open/high/low/close/volume）
  - 指数日线：index_daily 表（结构同上）
"""

import sqlite3
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ETF_SYMBOLS, DB_PATH, MOMENTUM_WINDOW


def load_all_etf_data(
    symbols: Optional[List[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
    momentum_window: int = MOMENTUM_WINDOW,
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    加载所有ETF日线数据并做日期对齐。

    为了提高动量计算精度，实际加载起始日期会前移 momentum_window+10 个自然日，
    计算完动量后裁剪回 start_date。

    Parameters
    ----------
    symbols : list of str
        ETF代码列表，默认使用 config.ETF_SYMBOLS
    start_date : str
        回测开始日期 YYYY-MM-DD
    end_date : str
        回测结束日期 YYYY-MM-DD，空字符串表示不限制
    db_path : str
        SQLite 数据库路径
    momentum_window : int
        动量计算窗口

    Returns
    -------
    etf_data : dict
        {symbol: DataFrame}，每个 DataFrame 含列：
        date, open, high, low, close, volume,
        pct_chg, cumulative_returns, amount,
        amount_ma20, atr, momentum
    common_dates : DatetimeIndex
        所有 ETF 共同的交易日索引
    """
    if symbols is None:
        symbols = ETF_SYMBOLS

    # 为了计算动量，数据加载起始日期前移（确保有足够的历史数据算 shift(momentum_window)）
    start_dt = pd.to_datetime(start_date)
    extended_start = (start_dt - timedelta(days=momentum_window * 3)).strftime("%Y-%m-%d")

    # ---- 逐只加载 ----
    etf_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _load_single_etf(sym, extended_start, end_date, db_path, momentum_window)
        if df is not None and len(df) > momentum_window:
            etf_data[sym] = df

    if not etf_data:
        raise ValueError(f"没有加载到任何 ETF 数据，请检查数据库路径: {db_path}")

    # ---- 日期对齐（取所有 ETF 的共同交易日）----
    date_sets = [set(df["date"].values) for df in etf_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    common_dates_dt = pd.DatetimeIndex(common_dates)

    # 过滤至共同日期
    for sym in list(etf_data.keys()):
        etf_data[sym] = etf_data[sym][etf_data[sym]["date"].isin(common_dates)].copy()
        etf_data[sym] = etf_data[sym].reset_index(drop=True)

    # ---- 裁剪回 start_date ----
    mask = common_dates_dt >= start_dt
    trimmed_dates = common_dates_dt[mask]
    for sym in etf_data:
        etf_data[sym] = etf_data[sym][etf_data[sym]["date"] >= pd.Timestamp(start_date)].copy()
        etf_data[sym] = etf_data[sym].reset_index(drop=True)

    return etf_data, trimmed_dates


def _load_single_etf(
    symbol: str,
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
    momentum_window: int = MOMENTUM_WINDOW,
) -> Optional[pd.DataFrame]:
    """从 SQLite 加载单只 ETF 日线数据并计算辅助列。"""
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT date, open, high, low, close, volume
            FROM etf_daily
            WHERE symbol = ? AND date >= ?
        """
        params: list = [symbol, start_date]
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return None

    # 类型安全转换
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["date"] = pd.to_datetime(df["date"])

    # 日收益率 & 累计收益率
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0

    # 成交额（volume 单位：份，amount = close × volume）
    df["amount"] = df["close"] * df["volume"]

    # 20日移动平均成交额（冲击成本用）
    df["amount_ma20"] = (
        df["amount"].rolling(window=20).mean().bfill().fillna(df["amount"])
    )

    # ATR（Average True Range）
    df["tr"] = _compute_true_range(df)
    df["atr"] = df["tr"].rolling(window=20).mean().bfill().fillna(df["tr"])

    # N日动量（核心信号），默认使用传入的 momentum_window
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
    # 额外预计算10日和20日动量（供动态窗口切换使用）
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    df["symbol"] = symbol
    return df


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
    """计算 True Range = max(high-low, |high-prev_close|, |low-prev_close|)。"""
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.fillna(tr1)  # 第一行没有prev_close时用tr1


def load_benchmark_data(
    symbol: str = "000300",
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
    momentum_window: int = MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """
    加载沪深300指数日线数据作为基准。

    返回 DataFrame 包含：date, close, pct_chg, cumulative_returns, momentum
    """
    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT date, close
            FROM index_daily
            WHERE symbol = ? AND date >= ?
        """
        params: list = [symbol, start_date]
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError(f"基准指数 {symbol} 在数据库中无数据")

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0)

    # 为了对齐ETF交易日，不做额外处理，由调用方进行日期对齐
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0

    # 基准动量（用于相对动量计算）
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1

    return df


def compute_equal_weight_benchmark(
    etf_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    计算 5 只 ETF 等权组合的收益率曲线。

    每个交易日每只 ETF 权重 1/5，组合日收益率为各 ETF 日收益率的算术平均。

    Returns
    -------
    pd.DataFrame
        索引为日期，包含列：
        - equal_weight_return: 组合日收益率
        - cumulative_returns: 组合累计收益率
    """
    # 提取所有 ETF 的日收益率到一个 DataFrame
    daily_returns = {}
    for sym, df in etf_data.items():
        daily_returns[sym] = df["pct_chg"].values

    ew_df = pd.DataFrame(daily_returns, index=pd.to_datetime(etf_data[list(etf_data.keys())[0]]["date"]))
    ew_df["equal_weight_return"] = ew_df.mean(axis=1)
    ew_df["cumulative_returns"] = (1 + ew_df["equal_weight_return"]).cumprod()
    ew_df.loc[ew_df.index[0], "cumulative_returns"] = 1.0

    return ew_df[["equal_weight_return", "cumulative_returns"]]
