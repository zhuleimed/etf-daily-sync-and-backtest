"""
波动率过滤轮动 — 每日模拟盘入口

在纯动量轮动基础上增加波动率门控：
  - 沪深300年化波动率 ≤ 30% → 正常动量轮动
  - 沪深300年化波动率 > 30% → 空仓避险

与 momentum_rotation 共享 DailySimEngine，仅增加波动率检查步骤。
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.engine import DailySimEngine
from simulation.framework.data import (
    load_latest_data, get_latest_trading_day, is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log

from simulation.strategies.momentum_vol_filter.config import (
    ETF_POOL, ETF_SYMBOLS, INITIAL_CAPITAL, MOMENTUM_WINDOW,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, VOL_THRESHOLD, VOL_WINDOW,
    MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    RISK_MODE, STOP_LOSS_PCT, PROFIT_THRESHOLD, DRAWBACK_PCT,
    DRAWDOWN_THRESHOLD, STRATEGY_NAME, STATE_FILE_DIR,
)

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)

logger = logging.getLogger("vol_filter_sim")


def is_high_volatility(etf_benchmark: pd.DataFrame | None, today_idx: int) -> bool:
    """判断市场是否处于高波动状态。

    用沪深300最近 VOL_WINDOW 个交易日的日收益率计算年化波动率。
    """
    if etf_benchmark is None or etf_benchmark.empty:
        return False
    if today_idx < VOL_WINDOW:
        return False
    subset = etf_benchmark.iloc[today_idx - VOL_WINDOW + 1: today_idx + 1]
    daily_vol = subset["pct_chg"].std()
    annual_vol = daily_vol * (252 ** 0.5)
    return annual_vol > VOL_THRESHOLD


def build_report(report: dict) -> list[str]:
    """统一格式日结报告（策略名替换为波动率过滤）。"""
    from simulation.strategies.momentum_rotation.daily import build_report as _build
    lines = _build(report)
    # 修正策略名称（_build输出的是动量轮动的策略名）
    for i, line in enumerate(lines):
        if "动量轮动模拟盘" in line:
            lines[i] = line.replace("动量轮动模拟盘", STRATEGY_NAME)
            break
    # 插入波动率信息
    vol_note = report.get("vol_note", "")
    if vol_note:
        # 找到分隔线后的位置插入
        for i, line in enumerate(lines):
            if "今日信号" in line and ">" in line:
                lines.insert(i + 1, f"  {vol_note}")
                break
    return lines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
        return

    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None:
        push_daily_report(STRATEGY_NAME, ["数据库无数据，跳过"])
        return
    if latest_day != today_str:
        push_error_alert(STRATEGY_NAME, f"最新数据 {latest_day}，非今日 {today_str}")
        return

    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH,
                                lookback_days=lookback,
                                momentum_window=MOMENTUM_WINDOW)
    if not etf_data:
        push_error_alert(STRATEGY_NAME, "行情数据加载失败")
        return

    # 加载基准指数（沪深300，用于波动率计算）
    etf_benchmark = None
    try:
        import sqlite3
        start_dt = (pd.Timestamp(today_str) - pd.Timedelta(days=lookback + 30)).strftime("%Y-%m-%d")
        with sqlite3.connect(str(DB_PATH)) as conn:
            bm = pd.read_sql_query(
                "SELECT date, close FROM index_daily WHERE symbol = '000300' AND date >= ? ORDER BY date",
                conn, params=[start_dt],
            )
            if not bm.empty:
                bm["date"] = pd.to_datetime(bm["date"])
                bm["pct_chg"] = bm["close"].pct_change().fillna(0)
                etf_benchmark = bm
    except Exception as e:
        logger.warning(f"基准指数加载失败: {e}")

    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            today_idx = df.index[mask][0]
            if today_idx >= MOMENTUM_WINDOW:
                break
    if today_idx is None:
        push_daily_report(STRATEGY_NAME, ["数据中未找到今日行情"])
        return

    state_mgr = StateManager(str(STATE_FILE_DIR), "momentum_vol_filter")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)

    # 创建引擎（与 momentum_rotation 相同的引擎，信号和排名函数不同）
    engine = DailySimEngine(
        state_mgr=state_mgr,
        broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_momentum_signals,
        rank_func=rank_etfs_by_momentum,
        etf_pool=ETF_POOL,
        momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION,
        min_hold_days=MIN_HOLD_DAYS,
        risk_mode=RISK_MODE,
        stop_loss_pct=STOP_LOSS_PCT,
        profit_threshold=PROFIT_THRESHOLD,
        drawback_pct=DRAWBACK_PCT,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
    )

    # 使用引擎的 run_daily 获得标准 report
    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report:
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 记录模拟盘日志
    append_simulation_log("momentum_vol_filter", STRATEGY_NAME, report, ETF_POOL)

    # 波动率过滤覆写：高波动时强制清仓
    high_vol = is_high_volatility(etf_benchmark, today_idx)
    vol_note = f"📊 年化波动: {_calc_vol_str(etf_benchmark, today_idx)}"
    report["vol_note"] = vol_note

    if high_vol:
        state = report.get("state")
        if state and state.position.shares > 0:
            # 高波动 → 产生待执行卖出订单（明天以 open 执行）
            hold_sym = state.position.symbol
            state.pending_order = {
                "action": "sell", "symbol": hold_sym,
                "reason": f"高波动清仓(年化>{VOL_THRESHOLD*100:.0f}%)",
                "created": today_str,
            }
            state_mgr.save(state)
            report["action"] = "vol_filter_pending"
            report["risk"] = {"triggered": True, "reason": f"高波动清仓(年化>{VOL_THRESHOLD*100:.0f}%)"}

    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


def _calc_vol_str(etf_benchmark, today_idx) -> str:
    """计算波动率文本。"""
    if etf_benchmark is None or today_idx < 20:
        return "数据不足"
    subset = etf_benchmark.iloc[today_idx - 19: today_idx + 1]
    if subset.empty:
        return "数据不足"
    vol = subset["pct_chg"].std() * (252 ** 0.5)
    return f"{vol*100:.1f}%"


if __name__ == "__main__":
    main()
