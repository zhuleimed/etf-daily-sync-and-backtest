"""
自适应轮动策略 — 每日模拟盘运行入口

调用方式：
    python -m simulation.strategies.adaptive_rotation.daily
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.engine import DailySimEngine
from simulation.framework.data import (
    load_latest_data, get_latest_trading_day, is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log

from simulation.strategies.adaptive_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION,
    MIN_HOLD_DAYS, COMMISSION_RATE, SLIPPAGE, DB_PATH, INITIAL_CAPITAL,
    RISK_MODE, STOP_LOSS_PCT, PROFIT_THRESHOLD, DRAWBACK_PCT,
    DRAWDOWN_THRESHOLD, STRATEGY_NAME, STATE_FILE_DIR,
)

from strategies.adaptive_rotation.signals import (
    compute_adaptive_scores, rank_etfs_by_adaptive,
)

logger = logging.getLogger("adaptive_rotation_sim")


def compute_adaptive_signals_wrapper(
    etf_data: dict[str, pd.DataFrame],
    today_idx: int,
    momentum_window: int = 20,
) -> pd.Series:
    """自适应信号包装器（兼容DailySimEngine接口）。"""
    # 模拟盘中无法获取完整index_data，用etf_data中的510300近似
    hs300_df = etf_data.get("510300")
    return compute_adaptive_scores(etf_data, today_idx, index_data=hs300_df)


def build_report(report: dict) -> list[str]:
    state = report.get("state")
    lines = []
    pool = ETF_POOL

    def name_of(sym):
        return f"{pool.get(sym, sym[:4])}({sym})" if sym else ""

    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    execd = report.get("order_executed")
    has_signal = False
    if execd:
        t = execd.get("type", "")
        if t == "buy":
            lines.append(f"  >> 今日信号: 开仓执行 买入{name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f}")
        elif t == "sell":
            lines.append(f"  >> 今日信号: 卖出执行 {name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f}")
        elif t == "switch":
            s = execd.get("sell", {}); b = execd.get("buy", {})
            lines.append(f"  >> 今日信号: 切换执行 {name_of(s.get('symbol',''))} -> {name_of(b.get('symbol',''))}")
        has_signal = True

    ranking = report.get("ranking", {})
    if ranking:
        rank_parts = []
        for rk in range(1, min(len(ranking) + 1, 4)):
            sym = ranking.get(str(rk))
            if sym:
                rank_parts.append(f"#{rk} {name_of(sym)}")
        if rank_parts:
            lines.append(f"      排名: {' > '.join(rank_parts)}")

    lines.append(f"  -------------------------------------------")
    lines.append(f"  账户日结")
    if state:
        pos = state.position
        if pos and pos.shares > 0:
            stock_val = report.get("stock_value", 0)
            lines.append(f"    持仓: {name_of(pos.symbol)} {pos.shares}股  均价{pos.avg_cost:.4f}")
            lines.append(f"    市值: {stock_val:>8.2f}")
        else:
            lines.append(f"    持仓: 空仓")
        lines.append(f"    现金: {state.cash:>8.2f}")
        total_value = report.get("total_value", 0)
        if state.initial_capital > 0:
            tr = (total_value / state.initial_capital - 1) * 100
            lines.append(f"    总资产: {total_value:>8.2f}  总收益率: {tr:+8.2f}%")
    lines.append(f"  ===========================================")
    return lines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        logger.info(f"{today_str} 非交易日，跳过")
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
        return

    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None or latest_day != today_str:
        msg = f"数据未就绪，跳过"
        logger.warning(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    lookback = 120
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback)
    if not etf_data:
        msg = "行情数据加载失败"
        logger.error(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx = df.index[mask][0]
            if idx >= 60:
                today_idx = idx
                break

    if today_idx is None:
        msg = f"数据不足"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    state_mgr = StateManager(str(STATE_FILE_DIR), "adaptive_rotation")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(
        state_mgr=state_mgr, broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_adaptive_signals_wrapper,
        rank_func=rank_etfs_by_adaptive,
        etf_pool=ETF_POOL,
        momentum_window=MOM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION,
        min_hold_days=MIN_HOLD_DAYS,
        risk_mode=RISK_MODE,
        stop_loss_pct=STOP_LOSS_PCT,
        profit_threshold=PROFIT_THRESHOLD,
        drawback_pct=DRAWBACK_PCT,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
    )

    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report:
        logger.error(report["error"])
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    append_simulation_log("adaptive_rotation", STRATEGY_NAME, report, ETF_POOL)
    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
