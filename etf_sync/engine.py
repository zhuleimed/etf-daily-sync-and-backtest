"""数据引擎模块：SQLite 存储与查询。

与 004_sequoia-x/sequoia_x/data/engine.py 结构一致：
  - etf_daily：ETF 日线表（OHLCV，不含财务字段）
  - index_daily：指数日线表（结构同 etf_daily，物理隔离）
  - etf_list：ETF 列表（代码、名称、退市标记）
  - sync_log：同步日志

字段说明（两表一致）：
  symbol, date, open, high, low, close, volume — 仅 OHLCV，不含 amount/pctChg/估值。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from etf_sync.config import Settings
from etf_sync.logger import get_logger

logger = get_logger(__name__)


def _migrate_columns(
    conn: sqlite3.Connection, table: str, columns: list[tuple[str, str]]
) -> None:
    """安全地给已有表新增列（列已存在则跳过）。"""
    for col_name, col_def in columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        except sqlite3.OperationalError:
            pass


# ── 建表 SQL ──

_CREATE_ETF_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS etf_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_ETF_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_etf_symbol_date ON etf_daily (symbol, date);
"""

_CREATE_INDEX_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS index_daily (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol   TEXT    NOT NULL,
    date     TEXT    NOT NULL,
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   REAL,
    UNIQUE (symbol, date)
);
"""

_CREATE_INDEX_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_index_symbol_date ON index_daily (symbol, date);
"""

_CREATE_ETF_LIST_SQL = """
CREATE TABLE IF NOT EXISTS etf_list (
    symbol       TEXT PRIMARY KEY,
    name         TEXT DEFAULT '',
    delisted_date TEXT,
    updated_at   TEXT DEFAULT (datetime('now','localtime'))
);
"""

_CREATE_SYNC_LOG_SQL = """
CREATE TABLE IF NOT EXISTS sync_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT    NOT NULL UNIQUE,
    status           TEXT    NOT NULL,
    etf_count        INTEGER DEFAULT 0,
    index_count      INTEGER DEFAULT 0,
    new_etf_count    INTEGER DEFAULT 0,
    delisted_etf_count INTEGER DEFAULT 0,
    is_trade_day     INTEGER DEFAULT 1,
    duration_seconds REAL    DEFAULT 0.0,
    error_msg        TEXT,
    created_at       TEXT    DEFAULT (datetime('now','localtime'))
);
"""


class DataEngine:
    """行情数据引擎，负责 SQLite 存储和查询。

    与 004 DataEngine 模式一致，但仅管理 ETF + 指数表。
    """

    def __init__(self, settings: Settings) -> None:
        self.db_path: str = settings.db_path
        self.start_date: str = settings.start_date
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库：建表 + 兼容性迁移。"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_ETF_TABLE_SQL)
            conn.execute(_CREATE_ETF_INDEX_SQL)
            conn.execute(_CREATE_INDEX_TABLE_SQL)
            conn.execute(_CREATE_INDEX_INDEX_SQL)
            conn.execute(_CREATE_ETF_LIST_SQL)
            conn.execute(_CREATE_SYNC_LOG_SQL)
            conn.commit()
        logger.info(f"数据库初始化完成：{self.db_path}")

    # ── ETF 日线 ──

    def get_etf_last_date(self, symbol: str) -> str | None:
        """获取单只 ETF 最新数据日期。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM etf_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_all_etf_last_dates(self) -> dict[str, str]:
        """获取所有 ETF 的最新日期。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT symbol, MAX(date) FROM etf_daily GROUP BY symbol"
                ).fetchall()
            return {row[0]: row[1] for row in rows if row[0] and row[1]}
        except Exception as e:
            logger.warning(f"get_all_etf_last_dates 异常: {e}")
            return {}

    def get_etf_symbols(self) -> list[str]:
        """获取本地数据库中有数据的 ETF 代码列表。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM etf_daily"
            ).fetchall()
        return [row[0] for row in rows]

    def get_etf_ohlcv(self, symbol: str) -> pd.DataFrame:
        """获取单只 ETF 全量日线。"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM etf_daily WHERE symbol = ? ORDER BY date",
                conn,
                params=(symbol,),
            )
        return df

    def write_etf_daily(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        """将 ETF 日线 DataFrame 写入 etf_daily 表。

        Args:
            df: 必须含 symbol, date, open, high, low, close, volume 列。
            conn: 已打开的数据库连接（便于外部事务管理）。

        Returns:
            写入行数。
        """
        if df.empty:
            return 0

        required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
        cols_present = [c for c in required_cols if c in df.columns]
        if "symbol" not in df.columns:
            logger.error("write_etf_daily: 缺少 symbol 列")
            return 0

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        if df.empty:
            return 0

        records = df[cols_present].to_dict("records")
        placeholders = ", ".join(f":{c}" for c in cols_present)
        cols_str = ", ".join(cols_present)
        conn.executemany(
            f"INSERT OR REPLACE INTO etf_daily ({cols_str}) VALUES ({placeholders})",
            records,
        )
        return len(records)

    # ── 指数日线 ──

    def get_index_last_date(self, symbol: str) -> str | None:
        """获取单只指数最新数据日期。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM index_daily WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def write_index_daily(self, df: pd.DataFrame, conn: sqlite3.Connection) -> int:
        """将指数日线 DataFrame 写入 index_daily 表。

        Args:
            df: 必须含 symbol, date, open, high, low, close, volume 列。
            conn: 已打开的数据库连接。

        Returns:
            写入行数。
        """
        if df.empty:
            return 0

        cols_present = [c for c in [
            "symbol", "date", "open", "high", "low", "close", "volume",
        ] if c in df.columns]

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])
        if df.empty:
            return 0

        records = df[cols_present].to_dict("records")
        placeholders = ", ".join(f":{c}" for c in cols_present)
        cols_str = ", ".join(cols_present)
        conn.executemany(
            f"INSERT OR REPLACE INTO index_daily ({cols_str}) VALUES ({placeholders})",
            records,
        )
        return len(records)

    # ── ETF 列表 ──

    def get_etf_list_symbols(self) -> list[str]:
        """获取 etf_list 表中的所有代码。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol FROM etf_list WHERE delisted_date IS NULL"
            ).fetchall()
        return [row[0] for row in rows]

    def get_all_etf_list_symbols(self) -> list[str]:
        """获取 etf_list 表中所有代码（含已退市）。"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol FROM etf_list"
            ).fetchall()
        return [row[0] for row in rows]

    def save_etf_list(
        self, symbols: list[str], names: dict[str, str] | None = None
    ) -> dict:
        """保存 ETF 列表，标记新增和退市。

        Args:
            symbols: 当前活跃的 ETF 代码列表。
            names: {symbol: name} 映射（可选）。

        Returns:
            {new_listed: [...], delisted: [...], total: int}
        """
        local = set(self.get_all_etf_list_symbols())
        remote = set(symbols)

        new_listed = sorted(remote - local)
        delisted = sorted(local - remote)

        with sqlite3.connect(self.db_path) as conn:
            for sym in symbols:
                name = (names or {}).get(sym, "")
                conn.execute(
                    "INSERT OR IGNORE INTO etf_list (symbol, name) VALUES (?, ?)",
                    (sym, name),
                )
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            for sym in delisted:
                conn.execute(
                    "UPDATE etf_list SET delisted_date = ? WHERE symbol = ?",
                    (today, sym),
                )
            conn.commit()

        logger.info(
            f"ETF 列表已更新: 远程 {len(symbols)} 只, "
            f"本地 {len(local)} 只, "
            f"新增 {len(new_listed)} 只, 退市 {len(delisted)} 只"
        )
        return {
            "new_listed": new_listed,
            "delisted": delisted,
            "total": len(symbols),
        }

    # ── 同步日志 ──

    def log_sync(
        self,
        status: str,
        etf_count: int = 0,
        index_count: int = 0,
        new_etf_count: int = 0,
        delisted_etf_count: int = 0,
        is_trade_day: bool = True,
        duration_seconds: float = 0.0,
        error_msg: str = "",
    ) -> None:
        """将同步结果写入 sync_log 表。"""
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO sync_log
                       (date, status, etf_count, index_count,
                        new_etf_count, delisted_etf_count,
                        is_trade_day, duration_seconds, error_msg)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        today_str, status, etf_count, index_count,
                        new_etf_count, delisted_etf_count,
                        1 if is_trade_day else 0, duration_seconds, error_msg,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"sync_log 写入失败: {e}")
