"""
HS300均线择时策略 — 每日模拟盘运行入口

HS300 > MA10 → 正常动量轮动
HS300 < MA10 → 空仓避险

调用方式：python -m simulation.strategies.hs300_ma_timing.daily

═══════════════════════════════════════════════════════════════════════════
⚠️ 本策略【未纳入 pipeline、从未运行】——原因与启用须知（2026-10-10 留记号）
═══════════════════════════════════════════════════════════════════════════
1) 没跑的原因：**回测已证伪**，不是接线问题。
   ① 它的回测曾是全项目第一（+320.6%），但那个数字来自 **look-ahead bias**——
      用当日收盘价决定当日开盘的交易。改用昨日数据（严格 T+1）后崩到 **+1.8%**，
      低于纯动量 131 个百分点（见 strategies/STRATEGY_COMPARISON.md 第 12 节）。
   ② 2026-10-02 换思路再测一次（不做独立策略、改做"组合层总闸"）→ **再次证伪**：
      熊市回撤更低但牛市收益砍掉 3/4（夏普 0.54）。

2) 接线是**完整**的，别误改：本策略 = "动量选股 + HS300 均线闸门"，
   复用 `compute_momentum_signals` 是**设计如此**（同 momentum_vol_filter），
   闸门由本文件的 check_bull_market() 真实实现，与文件顶部描述一致。

3) 如要启用，先知道这件事：本策略的 ETF 池 / 动量窗口 / 切换阈值
   （0.03 / 10 天）与纯动量**完全相同**，因此**强市期间它的选股决策与
   momentum_rotation 逐日一致**——这是设计产物、不是 bug。
   届时汇总里的"克隆巡检"会把这两个策略报出来（打分+净值都相同），
   应在 `simulation/framework/summary.py` 的 `_INTENTIONAL_SHARED_SCORING`
   里登记，**不要去查 signal_func**。
"""

from __future__ import annotations
import logging, sys
from datetime import date
from pathlib import Path

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

from simulation.strategies.hs300_ma_timing.config import (
    ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, MA_PERIOD,
    MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, DB_PATH,
    INITIAL_CAPITAL, RISK_MODE,
    STOP_LOSS_PCT, PROFIT_THRESHOLD,
    DRAWBACK_PCT, DRAWDOWN_THRESHOLD, STATE_FILE_DIR,
)

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)

logger = logging.getLogger("hs300_ma_timing")
STRATEGY_NAME = "HS300均线择时模拟盘"


def load_hs300_data(db_path=DB_PATH):
    """加载HS300指数数据用于均线计算。"""
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT date, close FROM index_daily WHERE symbol='000300' ORDER BY date",
        conn
    )
    conn.close()
    df["date"] = df["date"].astype(str)
    return df


def check_bull_market(hs300_df, today_str, ma_period=10):
    """检查今日HS300是否站上MA。"""
    if today_str not in hs300_df["date"].values:
        return True  # 无数据时默认允许交易
    idx = hs300_df[hs300_df["date"] == today_str].index[0]
    if idx < ma_period:
        return True
    close = hs300_df.iloc[idx]["close"]
    ma = hs300_df.iloc[idx - ma_period + 1:idx + 1]["close"].mean()
    return close > ma


def build_report(report: dict) -> list[str]:
    state = report.get("state"); lines = []
    def name_of(sym): return f"{ETF_POOL.get(sym, sym[:4])}({sym})"
    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append("  ===========================================")
    bull = report.get("bull_market", True)
    lines.append(f"  HS300 vs MA{MA_PERIOD}: {'牛' if bull else '熊'}市")
    execd = report.get("order_executed")
    blocked = report.get("order_blocked")
    if execd:
        t = execd.get("type", "")
        if t == "buy": lines.append(f"  >> 买入{name_of(execd['symbol'])} {execd['shares']}股")
        elif t == "sell": lines.append(f"  >> 卖出{name_of(execd['symbol'])}")
        elif t == "switch":
            s = execd.get("sell", {}); b = execd.get("buy", {})
            lines.append(f"  >> 切换 {name_of(s.get('symbol',''))} -> {name_of(b.get('symbol',''))}")
    elif blocked: lines.append(f"  >> 取消: {blocked.get('reason','')}")
    elif state and state.pending_order:
        po = state.pending_order
        pa = po.get("action", "?")
        sym = po.get("symbol", po.get("buy_symbol", "?"))
        lines.append(f"  >> {pa}信号 {name_of(sym)}（明日执行）")
    else:
        action = report.get("action", "")
        if action == "hold": lines.append(f"  >> 持有，无新信号")
        elif "cash" in action: lines.append(f"  >> 空仓观望（HS300<MA{MA_PERIOD}）")
        else: lines.append(f"  >> {action}")
    ranking = report.get("ranking", {})
    if ranking:
        parts = [f"#{rk} {name_of(ranking[str(rk)])}" for rk in range(1, min(len(ranking)+1, 4)) if str(rk) in ranking]
        if parts: lines.append(f"      排名: {' > '.join(parts)}")
    lines.append(f"  -------------------------------------------")
    lines.append(f"  账户日结")
    if state:
        pos = state.position
        if pos and pos.shares > 0:
            lines.append(f"    持仓: {name_of(pos.symbol)} {pos.shares}股  均价{pos.avg_cost:.4f}")
            lines.append(f"    市值: {report.get('stock_value', 0):>8.2f}")
        else: lines.append(f"    持仓: 空仓")
        lines.append(f"    现金: {state.cash:>8.2f}")
        tv = report.get("total_value", 0)
        if state.initial_capital > 0:
            lines.append(f"    总资产: {tv:>8.2f}  总收益率: {(tv/state.initial_capital-1)*100:+8.2f}%")
    lines.append(f"  ===========================================")
    return lines


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日"]); return

    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None:
        push_daily_report(STRATEGY_NAME, ["数据库无数据"]); return
    if latest_day != today_str:
        push_error_alert(STRATEGY_NAME, f"最新数据日{latest_day}非今日")
        return

    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback,
                                momentum_window=MOMENTUM_WINDOW)
    if not etf_data:
        push_error_alert(STRATEGY_NAME, "行情加载失败"); return

    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx = df.index[mask][0]
            if idx >= MOMENTUM_WINDOW: today_idx = idx; break
    if today_idx is None:
        push_daily_report(STRATEGY_NAME, [f"{today_str}数据不足"]); return

    # ── HS300均线择时 ──
    hs300_df = load_hs300_data()
    bull_market = check_bull_market(hs300_df, today_str, MA_PERIOD)
    logger.info(f"HS300 {'>' if bull_market else '<'} MA{MA_PERIOD}")

    state_mgr = StateManager(str(STATE_FILE_DIR), "hs300_ma_timing")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)

    if not bull_market:
        state = state_mgr.load()
        has_position = state and state.position and state.position.shares > 0
        if has_position:
            pos = state.position
            if pos.symbol in etf_data:
                open_px = float(etf_data[pos.symbol].iloc[today_idx]["open"])
                result = broker.sell(state, open_px, reason=f"HS300<MA{MA_PERIOD}清仓")
                if result and getattr(result, "success", False):
                    state_mgr.save(state)
        report = {"date": today_str, "action": "hold_cash", "bull_market": False,
                  "state": state, "stock_value": 0,
                  "total_value": state.cash if state else INITIAL_CAPITAL}
        append_simulation_log("hs300_ma_timing", STRATEGY_NAME, report, ETF_POOL)
        for line in build_report(report): logger.info(line)
        push_daily_report(STRATEGY_NAME, build_report(report))
        return

    # ── 牛市中正常动量轮动 ──
    engine = DailySimEngine(
        state_mgr=state_mgr, broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_momentum_signals,
        rank_func=rank_etfs_by_momentum,
        etf_pool=ETF_POOL, momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION,
        min_hold_days=MIN_HOLD_DAYS, risk_mode=RISK_MODE,
        stop_loss_pct=STOP_LOSS_PCT, profit_threshold=PROFIT_THRESHOLD,
        drawback_pct=DRAWBACK_PCT, drawdown_threshold=DRAWDOWN_THRESHOLD,
    )
    report = engine.run_daily(etf_data, today_idx, today_str)
    if "error" in report:
        push_error_alert(STRATEGY_NAME, report["error"]); return
    report["bull_market"] = True
    append_simulation_log("hs300_ma_timing", STRATEGY_NAME, report, ETF_POOL)
    for line in build_report(report): logger.info(line)
    push_daily_report(STRATEGY_NAME, build_report(report))
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
