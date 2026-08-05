"""定投+择时策略 — 回测（参照《ETF量化研究入门指南》第五章入门路径）

手册逻辑：
  每周定投沪深300ETF(510300)，用 20 日均线调整金额：
    价格 < MA20 → 多投（1.5 × 基准金额）——低估多买
    价格 > MA20 → 少投（0.5 × 基准金额）——高估少买

对比组：
  1. 一次性买入持有（等额本金）——基准
  2. 固定定投（1.0 × 基准金额，不择时）
  3. 定投+择时（0.5/1.5 调整）
  4. 定投+择时激进版（0.3/1.7）

信号时序（T+1）：
  每周最后交易日收盘：比较 close vs MA20 → 决定下周定投金额
  下周首个交易日开盘：买入

衡量指标：
  - 期末市值 / 累计投入 → 收益率（定投的现金流入不同期）
  - 年化收益率（XIRR 简化：用总收益率/年数）
  - 最大回撤（市值曲线）

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.dca_timing.backtest
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("dca")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "dca_results.json"

SYMBOL = "510300"         # 沪深300ETF（手册指定）
NAME = "沪深300ETF"
MA_WINDOW = 20            # 择时均线
WEEKLY_BASE = 1000.0      # 每周基准定投金额（元）
START_DATE = "2019-03-15" # 数据起点（510300 回填后 2019-03-06 起）
DB_PATH = "data/etf_daily.db"

COMMISSION = 0.0002       # 佣金万2
SLIPPAGE = 0.0001         # 滑点万1


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(str(PROJECT_ROOT / DB_PATH))
    df = pd.read_sql_query(
        "SELECT date, open, close FROM etf_daily WHERE symbol=? ORDER BY date",
        conn, params=(SYMBOL,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def backtest_dca(
    close: pd.Series,
    open_px: pd.Series,
    factor_low: float,
    factor_high: float,
    start: str = START_DATE,
) -> dict:
    """定投+择时回测。

    每周最后交易日收盘判信号 → 下周首日开盘买入。
    factor_low: 价格<MA20 时的金额倍数；factor_high: 价格>MA20 时的倍数。
    """
    dates = close.index[close.index >= pd.Timestamp(start)]
    close_s = close.loc[dates]
    open_s = open_px.loc[dates]

    ma20 = close_s.rolling(MA_WINDOW).mean()

    # 周分组：每周最后一个交易日
    weeks = dates.to_period("W")
    week_last = {w: dates[dates.to_period("W") == w].max() for w in weeks.unique()}

    invested = 0.0          # 累计投入
    shares = 0.0            # 累计持仓股数
    equity = []             # 市值曲线（持仓市值 + 未投入现金=0，定投全额投入）
    weekly_records = []

    pending_amount = 0.0    # 上周信号决定的定投金额（本周首日执行）

    for i, date in enumerate(dates):
        # ── 执行上周信号（本周首日开盘买入） ──
        if pending_amount > 0:
            px = open_s.loc[date] * (1 + SLIPPAGE)
            buy_shares = pending_amount * (1 - COMMISSION) / px
            shares += buy_shares
            invested += pending_amount
            weekly_records.append((date, pending_amount, px))
            pending_amount = 0.0

        # ── 每周最后交易日收盘：计算下周定投金额 ──
        if week_last.get(date.to_period("W")) == date:
            ma = ma20.loc[date]
            if not pd.isna(ma):
                if close_s.loc[date] < ma:
                    pending_amount = WEEKLY_BASE * factor_low   # 低估多投
                else:
                    pending_amount = WEEKLY_BASE * factor_high  # 高估少投

        # ── 估值 ──
        equity.append(shares * close_s.loc[date])

    eq = pd.Series(equity, index=dates)
    final_value = eq.iloc[-1]
    total_ret = final_value / invested - 1
    n_years = len(dates) / 252
    annual = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 and total_ret > -1 else float("nan")
    dd = (eq / eq.cummax() - 1).min()

    return {
        "status": "ok",
        "method": f"factor_low={factor_low},factor_high={factor_high}",
        "total_invested": round(invested, 2),
        "final_value": round(final_value, 2),
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual), 4) if not np.isnan(annual) else None,
        "max_drawdown": round(float(dd), 4),
        "n_weeks": len(weekly_records),
    }


def backtest_lump_sum(
    close: pd.Series, start: str = START_DATE, amount: float = WEEKLY_BASE * 100,
) -> dict:
    """一次性买入持有（等额本金基准：期初投入 amount）。"""
    dates = close.index[close.index >= pd.Timestamp(start)]
    c = close.loc[dates]
    px_in = c.iloc[0] * (1 + SLIPPAGE)
    shares = amount * (1 - COMMISSION) / px_in
    eq = shares * c
    total_ret = eq.iloc[-1] / amount - 1
    n_years = len(dates) / 252
    annual = (1 + total_ret) ** (1 / n_years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return {
        "status": "ok",
        "method": "一次性买入持有",
        "total_invested": amount,
        "final_value": round(float(eq.iloc[-1]), 2),
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual), 4),
        "max_drawdown": round(float(dd), 4),
        "n_weeks": 1,
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("定投+择时策略回测（手册第五章入门路径）")
    logger.info(f"Python: {sys.executable} | 标的: {NAME}({SYMBOL})")

    t0 = time.time()
    df = load_data()
    close, open_px = df["close"], df["open"]
    logger.info(f"数据: {len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

    results = [
        backtest_lump_sum(close),
        backtest_dca(close, open_px, 1.0, 1.0),          # 固定定投（不择时）
        backtest_dca(close, open_px, 1.5, 0.5),          # 手册版择时
        backtest_dca(close, open_px, 1.7, 0.3),          # 激进版择时
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("=" * 60)
    logger.info(f"{'方法':<22}{'累计投入':>10}{'期末市值':>10}{'总收益':>9}{'年化':>8}{'回撤':>8}{'周数':>6}")
    names = {
        "一次性买入持有": "一次性买入持有",
        "factor_low=1.0,factor_high=1.0": "固定定投(1.0)",
        "factor_low=1.5,factor_high=0.5": "定投+择时(手册版)",
        "factor_low=1.7,factor_high=0.3": "定投+择时(激进版)",
    }
    for r in results:
        logger.info(
            f"{names.get(r['method'], r['method']):<22}"
            f"{r['total_invested']:>10,.0f}{r['final_value']:>10,.0f}"
            f"{r['total_return']*100:>8.1f}%"
            f"{r['annual_return']*100 if r['annual_return'] else float('nan'):>7.1f}%"
            f"{r['max_drawdown']*100:>7.1f}%{r['n_weeks']:>6}"
        )
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
