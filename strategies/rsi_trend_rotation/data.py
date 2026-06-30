"""
数据加载模块

职责：
  1. 从 SQLite 加载 ETF 日线数据（含 RSI 所需的前期数据缓冲）
  2. 日期对齐（各 ETF 取共同交易日）
  3. 加载沪深300指数数据（用于市场状态判断）
  4. 计算等权组合基准
"""

import sqlite3
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .config import ETF_SYMBOLS, DB_PATH, RSI_PERIOD, RSI_SLOPE_PERIOD


def load_all_etf_data(
    symbols: Optional[list[str]] = None,
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    加载所有 ETF 日线数据并做日期对齐。

    预计算 RSI 所需的辅助列。
    """
    if symbols is None:
        symbols = ETF_SYMBOLS

    # RSI 需要足够的前期数据缓冲（至少 period + 缓冲）
    buffer_periods = RSI_PERIOD + RSI_SLOPE_PERIOD + 10
    start_dt = pd.to_datetime(start_date)
    extended_start = (start_dt - timedelta(days=buffer_periods * 3)).strftime("%Y-%m-%d")

    etf_data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _load_single_etf(sym, extended_start, end_date, db_path)
        if df is not None and len(df) > buffer_periods:
            etf_data[sym] = df

    if not etf_data:
        raise ValueError(f"没有加载到任何 ETF 数据，请检查数据库路径: {db_path}")

    # 日期对齐（取共同交易日）
    date_sets = [set(df["date"].values) for df in etf_data.values()]
    common_dates = sorted(set.intersection(*date_sets))
    common_dates_dt = pd.DatetimeIndex(common_dates)

    for sym in list(etf_data.keys()):
        etf_data[sym] = etf_data[sym][etf_data[sym]["date"].isin(common_dates)].copy()
        etf_data[sym] = etf_data[sym].reset_index(drop=True)

    # 裁剪回 start_date
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
) -> Optional[pd.DataFrame]:
    """加载单只 ETF 日线数据并预计算辅助列。"""
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

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["date"] = pd.to_datetime(df["date"])

    # 日收益率 & 累计收益率 & 成交额
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0
    df["amount"] = df["close"] * df["volume"]
    df["amount_ma20"] = df["amount"].rolling(window=20).mean().bfill().fillna(df["amount"])

    # ATR（风控用）
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).fillna(tr1)
    df["atr"] = tr.rolling(window=20).mean().bfill().fillna(tr)

    # 动量列（供回测报告使用）
    df["momentum"] = df["close"] / df["close"].shift(20) - 1

    df["symbol"] = symbol
    return df


def load_index_data(
    symbol: str = "000300",
    start_date: str = "2024-01-01",
    end_date: str = "",
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    加载指数日线数据（沪深300等），用于市场状态判断。
    """
    buffer_periods = RSI_PERIOD + RSI_SLOPE_PERIOD + 10
    start_dt = pd.to_datetime(start_date)
    extended_start = (start_dt - timedelta(days=buffer_periods * 3)).strftime("%Y-%m-%d")

    with sqlite3.connect(db_path) as conn:
        query = """
            SELECT date, close
            FROM index_daily
            WHERE symbol = ? AND date >= ?
        """
        params: list = [symbol, extended_start]
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0)
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[0, "cumulative_returns"] = 1.0

    return df


def compute_equal_weight_benchmark(
    etf_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """等权组合基准收益率。"""
    daily_returns = {}
    for sym, df in etf_data.items():
        daily_returns[sym] = df["pct_chg"].values

    first_sym = list(etf_data.keys())[0]
    ew_df = pd.DataFrame(
        daily_returns,
        index=pd.to_datetime(etf_data[first_sym]["date"]),
    )
    ew_df["equal_weight_return"] = ew_df.mean(axis=1)
    ew_df["cumulative_returns"] = (1 + ew_df["equal_weight_return"]).cumprod()
    ew_df.loc[ew_df.index[0], "cumulative_returns"] = 1.0
    return ew_df[["equal_weight_return", "cumulative_returns"]]
