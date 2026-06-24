"""
模拟盘数据加载模块

从 etf_daily.db 加载最近 N 个交易日的数据供策略计算信号。
与回测 data.py 共享数据库，但只取增量数据（最近 N 天）。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def load_latest_data(
    symbols: list[str],
    db_path: str | Path = "data/etf_daily.db",
    lookback_days: int = 30,
    momentum_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """加载最近 N 个交易日所有 ETF 数据。

    Args:
        symbols: ETF 代码列表。
        db_path: SQLite 数据库路径。
        lookback_days: 加载过去多少自然日的数据（含动量窗口余量）。
        momentum_window: 动量计算窗口（默认20日），决定 "momentum" 列的值。

    Returns:
        {symbol: DataFrame}，每只 ETF 含 OHLCV + 动量列。
        数据按日期升序排列。
    """
    # 计算起始日期（留余量确保动量计算所需历史足够）
    start_dt = (datetime.today() - timedelta(days=lookback_days + momentum_window + 10)).strftime("%Y-%m-%d")

    result: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(str(db_path)) as conn:
        for sym in symbols:
            df = pd.read_sql_query(
                """
                SELECT date, open, high, low, close, volume
                FROM etf_daily
                WHERE symbol = ? AND date >= ?
                ORDER BY date
                """,
                conn,
                params=[sym, start_dt],
            )
            if df.empty:
                continue

            # 类型转换
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df["date"] = pd.to_datetime(df["date"])
            df["pct_chg"] = df["close"].pct_change().fillna(0.0)

            # 动量列（供信号函数 compute_momentum_signals 使用）
            df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
            # 额外预计算固定窗口（供动态窗口切换使用）
            for w in [10, 15, 20]:
                df[f"momentum_{w}"] = df["close"] / df["close"].shift(w) - 1

            result[sym] = df

    return result


def get_latest_trading_day(
    symbols: list[str],
    db_path: str | Path = "data/etf_daily.db",
) -> str | None:
    """获取所有 ETF 共有的最后一个交易日。

    Returns:
        日期字符串 YYYY-MM-DD，无共同交易日时返回 None。
    """
    with sqlite3.connect(str(db_path)) as conn:
        dates = None
        for sym in symbols:
            df = pd.read_sql_query(
                "SELECT DISTINCT date FROM etf_daily WHERE symbol = ? ORDER BY date DESC LIMIT 5",
                conn,
                params=[sym],
            )
            if dates is None:
                dates = set(df["date"])
            else:
                dates &= set(df["date"])
        if not dates:
            return None
        return max(dates)


def is_trading_day(check_date: str | None = None) -> bool:
    """简易交易日判断（仅通过数据库中有无数据判断）。"""
    if check_date is None:
        check_date = datetime.today().strftime("%Y-%m-%d")

    with sqlite3.connect("data/etf_daily.db") as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM etf_daily WHERE date = ? LIMIT 1",
            (check_date,),
        )
        cnt = cur.fetchone()[0]
    return cnt > 0
