"""Sharpe排名 每日模拟盘"""
from __future__ import annotations
import logging, sys
from datetime import date
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.engine import DailySimEngine
from simulation.framework.data import load_latest_data, get_latest_trading_day, is_trading_day
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log
from simulation.strategies.sharpe_ranking.config import *
from strategies.momentum_rotation.momentum_signals import compute_momentum_signals, rank_etfs_by_momentum
logger = logging.getLogger("sharpe_ranking")
SNAME = "Sharpe排名"

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    if not is_trading_day(today_str): push_daily_report(SNAME, [f"{today_str} 非交易日"]); return
    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None: push_daily_report(SNAME, ["数据库无数据"]); return
    if latest_day != today_str: push_error_alert(SNAME, f"最新数据日{latest_day}非今日"); return
    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback, momentum_window=MOMENTUM_WINDOW)
    if not etf_data: push_error_alert(SNAME, "行情加载失败"); return
    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any() and df.index[mask][0] >= MOMENTUM_WINDOW:
            today_idx = df.index[mask][0]; break
    if today_idx is None: push_daily_report(SNAME, [f"{today_str}数据不足"]); return
    state_mgr = StateManager(str(STATE_FILE_DIR), "sharpe_ranking")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(state_mgr=state_mgr, broker=broker, config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_momentum_signals, rank_func=rank_etfs_by_momentum,
        etf_pool=ETF_POOL, momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION, min_hold_days=MIN_HOLD_DAYS,
        risk_mode=RISK_MODE, stop_loss_pct=0.05, profit_threshold=0.10, drawback_pct=0.05, drawdown_threshold=0.15)
    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report: push_error_alert(SNAME, report["error"]); return
    append_simulation_log("sharpe_ranking", SNAME, report, ETF_POOL)
    lines = [f"{SNAME} | {today_str}", f"操作: {report.get('action','?')}", f"总资产: {report.get('total_value',0):.0f}"]
    push_daily_report(SNAME, lines)
    logger.info(f"{SNAME} 完成")

if __name__ == "__main__": main()
