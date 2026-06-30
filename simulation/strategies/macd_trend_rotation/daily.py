"""
MACD 趋势轮动策略 — 每日模拟盘运行入口

调用方式：
    python -m simulation.strategies.macd_trend_rotation.daily
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
from simulation.framework.data import load_latest_data, get_latest_trading_day, is_trading_day
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log

from simulation.strategies.macd_trend_rotation.config import (
    ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION,
    MIN_HOLD_DAYS, COMMISSION_RATE, SLIPPAGE, DB_PATH, INITIAL_CAPITAL,
    RISK_MODE, STOP_LOSS_PCT, PROFIT_THRESHOLD, DRAWBACK_PCT,
    DRAWDOWN_THRESHOLD, STRATEGY_NAME, STATE_FILE_DIR,
)

from strategies.macd_trend_rotation.momentum_signals import (
    compute_macd_scores, rank_etfs_by_macd,
)

logger = logging.getLogger("macd_trend_sim")


def compute_macd_signals(etf_data, today_idx, momentum_window=20):
    """MACD评分信号（兼容DailySimEngine接口）。"""
    return compute_macd_scores(etf_data, today_idx)


def load_etf_data_with_buffer(symbols, db_path=DB_PATH, lookback_days=120, momentum_window=MOMENTUM_WINDOW):
    """加载数据（MACD需要充足历史）。"""
    return load_latest_data(symbols=symbols, db_path=db_path, lookback_days=lookback_days, momentum_window=momentum_window)


def build_report(report):
    state = report.get("state")
    lines = []
    action = report.get("action", "unknown")
    pool = ETF_POOL

    def name_of(sym):
        return f"{pool.get(sym, sym[:4])}({sym})" if sym else ""

    lines.append(""); lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    execd = report.get("order_executed"); blocked = report.get("order_blocked")
    risk = report.get("risk"); has_signal = False

    if execd:
        t = execd.get("type", "")
        if t == "buy": lines.append(f"  >> 今日信号: 开仓执行 买入{name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f}")
        elif t == "sell": lines.append(f"  >> 今日信号: 卖出执行 {name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f} 盈亏{execd.get('pnl', 0):+.2f}")
        elif t == "switch": s = execd.get("sell", {}); b = execd.get("buy", {}); lines.append(f"  >> 今日信号: 切换执行 {name_of(s.get('symbol',''))} -> {name_of(b.get('symbol',''))}")
        has_signal = True
    if blocked: lines.append(f"  >> 今日信号: 订单取消: {blocked.get('reason', '')}"); has_signal = True
    if state and state.pending_order:
        po = state.pending_order; pa = po.get("action", "?")
        if pa == "buy": lines.append(f"  >> 今日信号: 买入信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "sell": lines.append(f"  >> 今日信号: 卖出信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "switch": lines.append(f"  >> 今日信号: 切换信号 {name_of(po['sell_symbol'])}->{name_of(po['buy_symbol'])}（明日执行）")
        lines.append(f"      原因: {po.get('reason', '')}"); has_signal = True
    if risk and risk.get("triggered"): lines.append(f"  >> 今日信号: {risk['reason']}"); has_signal = True
    if not has_signal:
        h = name_of(state.position.symbol) if state and state.position.shares > 0 else ""
        if action == "hold": lines.append(f"  >> 今日信号: 持有 {h}，无新信号")
        elif action == "hold_cash": lines.append(f"  >> 今日信号: 空仓观望，无买入信号")
        else: lines.append(f"  >> 今日信号: 无新信号 ({action})")

    ranking = report.get("ranking", {})
    if ranking:
        parts = []
        for rk in range(1, min(len(ranking) + 1, 4)):
            sym = ranking.get(str(rk))
            if sym: parts.append(f"#{rk} {name_of(sym)}")
        if parts: lines.append(f"      MACD排名: {' > '.join(parts)}")

    lines.append(f"  -------------------------------------------"); lines.append(f"  账户日结")
    if state:
        pos = state.position
        if pos and pos.shares > 0:
            sv = report.get("stock_value", 0)
            lines.append(f"    持仓: {name_of(pos.symbol)} {pos.shares}股  均价{pos.avg_cost:.4f}")
            lines.append(f"    市值: {sv:>8.2f}")
        else: lines.append(f"    持仓: 空仓")
        lines.append(f"    现金: {state.cash:>8.2f}")
        tv = report.get("total_value", 0)
        if state.initial_capital > 0:
            tr = (tv / state.initial_capital - 1) * 100
            lines.append(f"    总资产: {tv:>8.2f}  总收益率: {tr:+8.2f}%")
    lines.append(f"  ===========================================")
    return lines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        logger.info(f"{today_str} 非交易日，跳过"); push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"]); return

    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None: logger.warning("数据库无ETF数据，跳过"); push_daily_report(STRATEGY_NAME, ["数据库无ETF数据"]); return
    if latest_day != today_str:
        msg = f"最新数据日为{latest_day}，非今日{today_str}，跳过"; logger.warning(msg); push_error_alert(STRATEGY_NAME, msg); return

    etf_data = load_etf_data_with_buffer(ETF_SYMBOLS, DB_PATH, lookback_days=120)
    if not etf_data: logger.error("行情数据加载失败"); push_error_alert(STRATEGY_NAME, "行情数据加载失败"); return

    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx_val = df.index[mask][0]
            if idx_val >= 40: today_idx = idx_val; break
    if today_idx is None: logger.warning(f"数据不足"); push_daily_report(STRATEGY_NAME, ["数据不足"]); return

    state_mgr = StateManager(str(STATE_FILE_DIR), "macd_trend_rotation")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(
        state_mgr=state_mgr, broker=broker, config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_macd_signals, rank_func=rank_etfs_by_macd,
        etf_pool=ETF_POOL, momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION, min_hold_days=MIN_HOLD_DAYS,
        risk_mode=RISK_MODE, stop_loss_pct=STOP_LOSS_PCT,
        profit_threshold=PROFIT_THRESHOLD, drawback_pct=DRAWBACK_PCT,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
    )
    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report: logger.error(report["error"]); push_error_alert(STRATEGY_NAME, report["error"]); return

    # 记录模拟盘日志
    append_simulation_log("macd_trend_rotation", STRATEGY_NAME, report, ETF_POOL)

    lines = build_report(report)
    for line in lines: logger.info(line)
    push_daily_report(STRATEGY_NAME, lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
