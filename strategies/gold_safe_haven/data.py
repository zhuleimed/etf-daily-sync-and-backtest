"""
数据加载模块

职责：
  1. 从 SQLite 加载 8 只 ETF（7宽基 + 1黄金）日线数据
  2. 日期对齐（取共同交易日）
  3. 预计算动量列和恐慌指标列
  4. 加载基准指数数据
"""
import sqlite3
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    ETF_SYMBOLS, BROAD_SYMBOLS, GOLD_SYMBOL,
    DB_PATH, MOMENTUM_WINDOW, ZSCORE_WINDOW,
)


def load_all_etf_data(
    symbols: Optional[List[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
    momentum_window: int = MOMENTUM_WINDOW,
    zscore_window: int = ZSCORE_WINDOW,
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    加载所有ETF日线数据并做日期对齐。

    Parameters
    ----------
    symbols : list of str
        ETF代码列表，默认使用 config.ETF_SYMBOLS（7宽基+1黄金）
    start_date : str
        回测开始日期 YYYY-MM-DD
    end_date : str
        回测结束日期
    db_path : str
        SQLite 数据库路径
    momentum_window : int
        动量计算窗口（默认20）
    zscore_window : int
        Z-score滚动窗口（默认252）

    Returns
    -------
    etf_data : dict
        {symbol: DataFrame}，含基础列和辅助列
    common_dates : DatetimeIndex
        所有 ETF 共同的交易日索引
    """
    if symbols is None:
        symbols = ETF_SYMBOLS

    # 为了计算动量和Z-score，起始日期前移
    start_dt = pd.to_datetime(start_date)
    extended_start = (start_dt - timedelta(days=max(momentum_window * 3, zscore_window + 100))).strftime("%Y-%m-%d")

    # ---- 逐只加载 ----
    etf_data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _load_single_etf(sym, extended_start, end_date, db_path, momentum_window)
        if df is not None and len(df) > momentum_window:
            etf_data[sym] = df
        else:
            print(f"  ⚠ {sym} 数据不足，跳过（{len(df) if df is not None else 0}行）")

    if not etf_data:
        raise ValueError(f"没有加载到任何 ETF 数据，请检查数据库路径: {db_path}")

    # ---- 日期对齐 ----
    date_sets = [set(df["date"].values) for df in etf_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    common_dates_dt = pd.DatetimeIndex(common_dates)

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
    """从 SQLite 加载单只 ETF 日线数据并计算所有辅助列。"""
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

    # ---- 基础列 ----
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0

    # 成交额
    df["amount"] = df["close"] * df["volume"]
    df["amount_ma20"] = (
        df["amount"].rolling(window=20).mean().bfill().fillna(df["amount"])
    )

    # ATR
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)
    df["atr"] = df["tr"].rolling(window=20).mean().bfill().fillna(df["tr"])

    # ---- 动量列 ----
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # ---- 恐慌指标基础列 ----
    # 5日收益率（用于计算最大跌幅和广度）
    df["ret_5d"] = df["close"] / df["close"].shift(5) - 1

    # 日收益率（用于计算波动率）
    df["ret_1d"] = df["close"].pct_change().fillna(0.0)

    # 年化波动率21日和63日
    df["vol_21d"] = df["ret_1d"].rolling(window=21).std() * np.sqrt(252)
    df["vol_63d"] = df["ret_1d"].rolling(window=63).std() * np.sqrt(252)

    # 波动率比（短期/长期，>1表示波动率在上升）
    df["vol_ratio"] = np.where(
        df["vol_63d"] > 0,
        df["vol_21d"] / df["vol_63d"],
        1.0,
    )

    df["symbol"] = symbol
    return df


def load_benchmark_data(
    symbol: str = "000300",
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
    momentum_window: int = MOMENTUM_WINDOW,
) -> pd.DataFrame:
    """加载基准指数日线数据。"""
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
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
    return df


def compute_equal_weight_benchmark(
    etf_data: Dict[str, pd.DataFrame],
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """计算宽基ETF等权组合的收益率曲线（不含黄金）。"""
    if symbols is None:
        symbols = BROAD_SYMBOLS

    daily_returns = {}
    for sym in symbols:
        if sym in etf_data:
            daily_returns[sym] = etf_data[sym]["pct_chg"].values

    if not daily_returns:
        raise ValueError("没有宽基ETF数据可用于计算等权基准")

    first_sym = list(daily_returns.keys())[0]
    ew_df = pd.DataFrame(
        daily_returns,
        index=pd.to_datetime(etf_data[first_sym]["date"])
    )
    ew_df["equal_weight_return"] = ew_df.mean(axis=1)
    ew_df["cumulative_returns"] = (1 + ew_df["equal_weight_return"]).cumprod()
    ew_df.loc[ew_df.index[0], "cumulative_returns"] = 1.0
    return ew_df[["equal_weight_return", "cumulative_returns"]]
