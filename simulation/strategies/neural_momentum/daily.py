"""Neural Momentum 模拟盘 — 每日运行入口

复用 DailySimEngine（单仓模型）+ 定制混合信号（0.25×动量z + 0.75×神经z）。
与 019 全项目同构：T+1 待执行订单、CSV 日志、SQLite 快照、微信日报。
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.data import (
    load_latest_data, get_latest_trading_day, is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.engine import DailySimEngine
from simulation.framework.report_builder import build_signal_report

from simulation.strategies.neural_momentum.config import (
    ETF_POOL, ETF_SYMBOLS, STRATEGY_ID, STRATEGY_NAME,
    MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    RISK_MODE, INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE, DB_PATH,
)
from simulation.strategies.neural_momentum.signals import neural_momentum_signals
from strategies.momentum_rotation.momentum_signals import rank_etfs_by_momentum

logger = logging.getLogger("neural_momentum_sim")

STATE_FILE_DIR = PROJECT_ROOT / "simulation" / "output"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    # 1. 交易日判断
    if not is_trading_day(today_str):
        msg = f"{today_str} 非交易日，跳过"
        logger.info(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # 2. 数据就绪检查
    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None:
        msg = "数据库无 ETF 数据，跳过"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return
    if latest_day != today_str:
        msg = f"最新数据日为 {latest_day}，非今日 {today_str}，跳过"
        logger.warning(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 3. 加载行情（momentum 列由 data 层预计算）
    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(
        ETF_SYMBOLS, DB_PATH, lookback_days=lookback,
        momentum_window=MOMENTUM_WINDOW,
    )
    if not etf_data:
        msg = "行情数据加载失败"
        logger.error(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 4. 今日索引
    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx = df.index[mask][0]
            if idx >= MOMENTUM_WINDOW:
                today_idx = idx
                break
    if today_idx is None:
        msg = f"在数据中未找到 {today_str} 的完整行情"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # 5. 引擎（定制混合信号）
    state_mgr = StateManager(str(STATE_FILE_DIR), STRATEGY_ID)
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(
        state_mgr=state_mgr,
        broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=neural_momentum_signals,
        rank_func=rank_etfs_by_momentum,
        etf_pool=ETF_POOL,
        momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION,
        min_hold_days=MIN_HOLD_DAYS,
        risk_mode=RISK_MODE,
        stop_loss_pct=0.05,
        profit_threshold=0.10,
        drawback_pct=0.05,
        drawdown_threshold=0.15,
    )

    # 6. 运行
    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report:
        logger.error(report["error"])
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 7. CSV 日志（engine 已写 SQLite 快照）
    from simulation.framework.log_writer import append_simulation_log
    append_simulation_log(STRATEGY_ID, STRATEGY_NAME, report, ETF_POOL)

    # 8. 微信日报（T+1 三板块格式）
    report_lines = build_signal_report(report, STRATEGY_NAME, ETF_POOL)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
