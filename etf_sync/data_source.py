"""数据源模块：提供 ETF 和指数的日线数据获取。

双源策略（参考 004_sequoia-x 的 baostock→Tencent 双源模式）：
  - ETF 日线：腾讯接口（主） → Sina 接口（备选）
  - 指数日线：Sina 接口（腾讯接口不支持指数 K 线）
  - ETF 列表：akshare 获取（约 1500+ 只）

所有方法返回统一格式的 DataFrame（date, open, high, low, close, volume）。
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from etf_sync.logger import get_logger

logger = get_logger(__name__)


def to_tencent_code(symbol: str) -> str:
    """将纯数字代码转为腾讯接口格式。

    Args:
        symbol: 纯数字代码，如 "510050" 或 "159915"。

    Returns:
        腾讯格式代码，如 "sh510050" 或 "sz159915"。
    """
    prefix = "sh" if symbol.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{symbol}"


def to_sina_code(symbol: str) -> str:
    """将纯数字代码转为 Sina 接口格式。

    Args:
        symbol: 纯数字代码，如 "000300"（指数）或 "510050"（ETF）。

    Returns:
        Sina 格式代码，如 "sh000300"（指数）或 "sh510050"（ETF）。
    """
    # 指数代码（00 开头）：上海交易所指数，固定 sh 前缀
    # 指数代码（39 开头）：深圳交易所指数，固定 sz 前缀
    if symbol.startswith("00"):
        return f"sh{symbol}"
    if symbol.startswith("39"):
        return f"sz{symbol}"
    # ETF/股票代码：沿用 to_tencent_code 的规则
    return to_tencent_code(symbol)


class TencentSource:
    """腾讯行情数据源 — ETF 日线获取（主源）。

    接口说明：
      - 日 K 线: web.ifzq.gtimg.cn（前复权）
      - 实时行情: qt.gtimg.cn

    与 004_sequoia-x/sequoia_x/data/tencent_source.py 实现一致，
    但只保留 ETF 需要的字段（OHLCV，不含财务/估值指标）。
    """

    def __init__(self, request_interval: float = 0.15):
        """初始化 TencentSource。

        Args:
            request_interval: 请求间隔（秒），防止触发频率限制。
        """
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self.request_interval = request_interval
        self._last_request = 0.0
        # 数据源追踪（双轨制）
        self.last_source: str | None = None   # 最近一次请求使用的数据源："tencent" | "sina" | None
        self.source_count: dict[str, int] = {"tencent": 0, "sina": 0}  # 各数据源累计成功次数
        self.active_source: str = "tencent"   # 当前活跃数据源

    def _rate_limit(self) -> None:
        """请求间隔限制，避免被 API 封 IP。"""
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request = time.time()

    def get_daily(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """获取单只 ETF 日线（腾讯 → Sina 自动切换）。

        双轨制策略：
          - 默认优先使用 Tencent 接口
          - 若 Tencent 连续失败，自动切换到 Sina
          - 每 N 次尝试恢复 Tencent 一次（避免 Sina 一直背锅）
          - 通过 last_source / source_count 追踪各源的使用情况

        Args:
            code: 腾讯格式代码，如 'sh510050'。
            days: 获取最近 N 条日线。

        Returns:
            DataFrame 含 date, open, high, low, close, volume，失败返回 None。
        """
        # 根据活跃源决定首选
        if self.active_source == "tencent":
            df = self._tencent_kline(code, days)
            if df is not None and not df.empty:
                self.last_source = "tencent"
                self.source_count["tencent"] += 1
                return df
            # 腾讯失败 → 切到 Sina
            logger.debug(f"Tencent 失败（{code}），切换 Sina")
            self.active_source = "sina"
            df = self._sina_kline(code, days)
            if df is not None and not df.empty:
                self.last_source = "sina"
                self.source_count["sina"] += 1
                return df
            return None

        # 当前活跃源为 Sina → 每 50 次试一次 Tencent
        if self.source_count["sina"] > 0 and self.source_count["sina"] % 50 == 0:
            df = self._tencent_kline(code, days)
            if df is not None and not df.empty:
                logger.info(f"Tencent 已恢复，切回主源（code={code}）")
                self.active_source = "tencent"
                self.last_source = "tencent"
                self.source_count["tencent"] += 1
                return df

        df = self._sina_kline(code, days)
        if df is not None and not df.empty:
            self.last_source = "sina"
            self.source_count["sina"] += 1
            return df

        # Sina 也失败 → 再试一次 Tencent 碰运气
        df = self._tencent_kline(code, days)
        self.last_source = "tencent" if df is not None else None
        if df is not None:
            self.source_count["tencent"] += 1
        return df

    def _tencent_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从腾讯获取前复权日线。

        API: web.ifzq.gtimg.cn/appstock/app/fqkline/get
        """
        url = (
            f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/"
            f"get?param={code},day,,,{days},qfq"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            data = r.json()
            rows = data.get("data", {}).get(code, {}).get("qfqday", [])
            if not rows:
                return None
            df = pd.DataFrame(
                rows, columns=["date", "open", "close", "high", "low", "volume"]
            )
            for col in ["open", "close", "high", "low", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            # 重排列为统一顺序
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.debug(f"Tencent kline error（{code}）: {e}")
            return None

    def _sina_kline(self, code: str, days: int = 5) -> Optional[pd.DataFrame]:
        """从新浪获取日线（备选方案）。

        API: quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData
        """
        url = (
            f"https://quotes.sina.cn/cn/api/json_v2.php/"
            f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={days}"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            rows = r.json()
            if not rows or not isinstance(rows, list):
                return None
            records = []
            for row in rows:
                records.append({
                    "date": row["day"][:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            return pd.DataFrame(records)
        except Exception as e:
            logger.debug(f"Sina kline error（{code}）: {e}")
            return None

    def batch_get_daily(
        self, codes: list[str], days: int = 5
    ) -> dict[str, pd.DataFrame]:
        """批量获取多只 ETF 日线。

        Args:
            codes: 腾讯格式代码列表，如 ['sh510050', 'sz159915']。
            days: 获取最近 N 条。

        Returns:
            {code: DataFrame}，失败的不在结果中。
        """
        results: dict[str, pd.DataFrame] = {}
        for i, code in enumerate(codes):
            df = self.get_daily(code, days)
            if df is not None:
                results[code] = df
            if (i + 1) % 100 == 0:
                logger.info(f"TencentSource: {i+1}/{len(codes)} 完成")
        return results


class IndexDataSource:
    """指数行情数据源 — 通过 Sina API 获取指数日线。

    腾讯接口不支持指数 K 线（无 qfqday），因此指数数据走 Sina 接口。
    与 akshare 的数据格式一致，但不依赖 akshare 的网络稳定性。

    支持的指数代码（上证系列）：
      sh000016 上证50, sh000300 沪深300, sh000905 中证500, sh000852 中证1000
    """

    def __init__(self, request_interval: float = 0.15):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        })
        self.request_interval = request_interval
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request = time.time()

    def get_daily(
        self, code: str, days: int = 800
    ) -> Optional[pd.DataFrame]:
        """获取指数日线数据。

        Args:
            code: Sina 格式指数代码，如 'sh000300'（沪深300）。
            days: 获取最近 N 条日线（默认 800 ≈ 3年+，覆盖 2024-01 至今）。

        Returns:
            DataFrame 含 date, open, high, low, close, volume，失败返回 None。
        """
        url = (
            f"https://quotes.sina.cn/cn/api/json_v2.php/"
            f"CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen={days}"
        )
        try:
            self._rate_limit()
            r = self._session.get(url, timeout=10)
            rows = r.json()
            if not rows or not isinstance(rows, list):
                return None
            records = []
            for row in rows:
                records.append({
                    "date": row["day"][:10],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })
            df = pd.DataFrame(records)
            df = df.sort_values("date").reset_index(drop=True)
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"IndexSource kline error（{code}）: {e}")
            return None


def get_etf_list() -> list[str]:
    """通过 akshare 获取全量场内 ETF 代码列表。

    返回纯数字代码列表（如 ["510050", "159915", ...]），
    不包含已退市或非交易状态的 ETF。

    Returns:
        ETF 代码列表（空列表表示获取失败）。
    """
    try:
        import akshare as ak

        df = ak.fund_etf_spot_em()
        if df is None or df.empty:
            logger.warning("get_etf_list: akshare 返回空列表")
            return []

        codes: list[str] = df["代码"].astype(str).str.strip().tolist()
        logger.info(f"get_etf_list: 获取 {len(codes)} 只 ETF 代码")
        return codes
    except Exception as e:
        logger.warning(f"get_etf_list: akshare 获取失败（{e}），尝试本地缓存")
        return []
