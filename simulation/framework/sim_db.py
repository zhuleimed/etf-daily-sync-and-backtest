"""
模拟盘结构化数据持久化模块

在 JSON 状态文件 + CSV 日志之外，增加 SQLite 数据库辅助存储：
  - sim_closed_trades  — 已平仓交易记录（每笔卖出/切换写入一条）
  - sim_account_daily  — 每日资产快照（每个策略每天一条）

与 JSON/CSV 互补而非替代，三者可同时存在。
JSON → 运行时状态，CSV → 可读历史，SQLite → 结构化查询。
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

logger = __import__("logging").getLogger(__name__)

# ── 默认数据库路径（相对于本文件的 output 目录） ──
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "output"
_DEFAULT_DB_PATH = str(_DEFAULT_DB_DIR / "sim_trading.db")


# ════════════════════════════════════════════════════════════════
#  建表
# ════════════════════════════════════════════════════════════════

def init_db(db_path: str = _DEFAULT_DB_PATH) -> None:
    """建表（幂等），首次运行时自动创建。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sim_closed_trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy    TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                etf_name    TEXT DEFAULT '',
                action      TEXT NOT NULL,
                buy_date    TEXT DEFAULT '',
                sell_date   TEXT NOT NULL,
                hold_days   INTEGER DEFAULT 0,
                buy_price   REAL DEFAULT 0,
                sell_price  REAL NOT NULL,
                shares      INTEGER NOT NULL,
                total_cost  REAL NOT NULL,
                net_revenue REAL NOT NULL,
                commission  REAL DEFAULT 0,
                pnl         REAL NOT NULL,
                pnl_pct     REAL DEFAULT 0,
                exit_reason TEXT DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now','localtime'))
            );
            CREATE INDEX IF NOT EXISTS idx_ct_strategy
                ON sim_closed_trades (strategy, sell_date);

            CREATE TABLE IF NOT EXISTS sim_account_daily (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                strategy        TEXT NOT NULL,
                strategy_name   TEXT DEFAULT '',
                cash            REAL NOT NULL,
                stock_value     REAL NOT NULL,
                total_value     REAL NOT NULL,
                total_return    REAL DEFAULT 0,
                position_symbol TEXT DEFAULT '',
                position_shares INTEGER DEFAULT 0,
                UNIQUE(date, strategy)
            );
            CREATE INDEX IF NOT EXISTS idx_ad_strategy
                ON sim_account_daily (strategy, date);
        """)


# ════════════════════════════════════════════════════════════════
#  已平仓交易
# ════════════════════════════════════════════════════════════════


def record_closed_trade(
    trade: dict,
    db_path: str = _DEFAULT_DB_PATH,
) -> int | None:
    """写入一条已平仓交易。

    Args:
        trade: 包含 strategy, symbol, sell_date, shares, total_cost 等字段。
        db_path: 数据库路径（默认 sim_trading.db）。

    Returns:
        记录 id，失败返回 None。
    """
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """INSERT INTO sim_closed_trades
                   (strategy, symbol, etf_name, action, buy_date, sell_date,
                    hold_days, buy_price, sell_price, shares,
                    total_cost, net_revenue, commission,
                    pnl, pnl_pct, exit_reason)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.get("strategy", ""),
                    trade.get("symbol", ""),
                    trade.get("etf_name", ""),
                    trade.get("action", "sell"),
                    trade.get("buy_date", ""),
                    trade.get("sell_date", date.today().isoformat()),
                    trade.get("hold_days", 0),
                    trade.get("buy_price", 0.0),
                    trade.get("sell_price", 0.0),
                    trade.get("shares", 0),
                    trade.get("total_cost", 0.0),
                    trade.get("net_revenue", 0.0),
                    trade.get("commission", 0.0),
                    trade.get("pnl", 0.0),
                    trade.get("pnl_pct", 0.0),
                    trade.get("exit_reason", ""),
                ),
            )
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        logger.error(f"记录已平仓交易失败: {e}")
        return None


def get_closed_trades(
    strategy: str | None = None,
    limit: int = 100,
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict]:
    """查询已平仓交易。

    Args:
        strategy: 策略名，None=全部策略。
        limit: 最大条数。
        db_path: 数据库路径。

    Returns:
        [{"strategy": "...", "symbol": "...", ...}, ...]
    """
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy:
                rows = conn.execute(
                    "SELECT * FROM sim_closed_trades WHERE strategy=? "
                    "ORDER BY sell_date DESC, id DESC LIMIT ?",
                    (strategy, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sim_closed_trades "
                    "ORDER BY sell_date DESC, id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"查询已平仓交易失败: {e}")
        return []


# ════════════════════════════════════════════════════════════════
#  每日资产快照
# ════════════════════════════════════════════════════════════════


def record_account_daily(
    snapshot: dict,
    db_path: str = _DEFAULT_DB_PATH,
) -> None:
    """写入（或替换）一条每日资产快照。

    Args:
        snapshot: 包含 date, strategy, cash, stock_value, total_value 等字段。
        db_path: 数据库路径。
    """
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sim_account_daily
                   (date, strategy, strategy_name,
                    cash, stock_value, total_value, total_return,
                    position_symbol, position_shares)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    snapshot.get("date", date.today().isoformat()),
                    snapshot.get("strategy", ""),
                    snapshot.get("strategy_name", ""),
                    snapshot.get("cash", 0.0),
                    snapshot.get("stock_value", 0.0),
                    snapshot.get("total_value", 0.0),
                    snapshot.get("total_return", 0.0),
                    snapshot.get("position_symbol", ""),
                    snapshot.get("position_shares", 0),
                ),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"记录每日快照失败: {e}")


def get_account_daily(
    strategy: str | None = None,
    days: int = 30,
    db_path: str = _DEFAULT_DB_PATH,
) -> list[dict]:
    """查询每日资产快照。

    Args:
        strategy: 策略名，None=全部。
        days: 最近 N 天。
        db_path: 数据库路径。

    Returns:
        [{"date": "...", "strategy": "...", "total_value": ..., ...}, ...]
    """
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            if strategy:
                rows = conn.execute(
                    "SELECT * FROM sim_account_daily WHERE strategy=? "
                    "ORDER BY date DESC LIMIT ?",
                    (strategy, days),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sim_account_daily "
                    "ORDER BY date DESC LIMIT ?",
                    (days,),
                ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"查询每日快照失败: {e}")
        return []


def get_all_strategies(db_path: str = _DEFAULT_DB_PATH) -> list[str]:
    """获取数据库中有记录的所有策略名。"""
    try:
        init_db(db_path)
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT strategy FROM sim_account_daily ORDER BY strategy"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
