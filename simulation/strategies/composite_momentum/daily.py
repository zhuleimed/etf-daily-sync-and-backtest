"""
复合动量轮动策略 — 每日模拟盘运行入口

调用方式（由 pipeline.py 在数据同步完成后触发）：
    python -m simulation.strategies.composite_momentum.daily

也可独立运行测试：
    python -m simulation.strategies.composite_momentum.daily
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# ── 确保项目根目录在 path 中 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.engine import DailySimEngine
from simulation.framework.data import (
    load_latest_data,
    get_latest_trading_day,
    is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log

from simulation.strategies.composite_momentum.config import (
    ETF_POOL,
    ETF_SYMBOLS,
    MOMENTUM_WINDOW,
    MIN_SWITCH_CONVICTION,
    MIN_HOLD_DAYS,
    COMMISSION_RATE,
    SLIPPAGE,
    DB_PATH,
    INITIAL_CAPITAL,
    RISK_MODE,
    STOP_LOSS_PCT,
    PROFIT_THRESHOLD,
    DRAWBACK_PCT,
    DRAWDOWN_THRESHOLD,
    STRATEGY_NAME,
    STATE_FILE_DIR,
    MARKET_INDEX,
    MARKET_MA_PERIOD,
)

from strategies.composite_momentum.momentum_signals import (
    compute_composite_score as _compute_composite_score,
    rank_etfs_by_composite,
    judge_market_regime,
)

logger = logging.getLogger("composite_momentum_sim")

# ── 全局缓存：加载的指数数据，供信号函数使用 ──
_index_data: pd.DataFrame | None = None


def compute_composite_signals(
    etf_data: dict[str, pd.DataFrame],
    today_idx: int,
    momentum_window: int = 20,  # 兼容 DailySimEngine 接口，实际不使用
) -> pd.Series:
    """
    计算综合打分信号（包装 compute_composite_score，兼容 DailySimEngine 接口）。

    momentum_window 参数被忽略——复合动量使用内置的多时间尺度加权。
    """
    global _index_data
    return _compute_composite_score(
        etf_data, today_idx,
        index_data=_index_data,
        market_ma_period=MARKET_MA_PERIOD,
    )


def load_etf_and_index_data(
    symbols: list[str],
    db_path: str | Path = DB_PATH,
    lookback_days: int = 90,  # 需要60日动量窗口+余量
    momentum_window: int = MOMENTUM_WINDOW,
) -> dict[str, pd.DataFrame]:
    """
    加载 ETF 数据的同时加载指数数据用于市场状态判断。

    指数数据存入全局 _index_data，供信号函数调用。
    """
    global _index_data

    # 加载 ETF 数据
    etf_data = load_latest_data(
        symbols=symbols,
        db_path=db_path,
        lookback_days=lookback_days,
        momentum_window=momentum_window,
    )

    # 加载指数数据
    import sqlite3
    from datetime import datetime, timedelta

    start_dt = (datetime.today() - timedelta(days=lookback_days + momentum_window + 30)).strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, close
                FROM index_daily
                WHERE symbol = ? AND date >= ?
                ORDER BY date
                """,
                conn,
                params=[MARKET_INDEX, start_dt],
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0)
            df["pct_chg"] = df["close"].pct_change().fillna(0.0)
            _index_data = df
            logger.info(f"指数 {MARKET_INDEX} 数据加载完成：{len(df)}行")
        else:
            _index_data = None
            logger.warning(f"指数 {MARKET_INDEX} 数据为空")
    except Exception as e:
        _index_data = None
        logger.warning(f"指数 {MARKET_INDEX} 加载失败: {e}")

    return etf_data


def build_report(report: dict) -> list[str]:
    """统一格式日结报告。"""
    state = report.get("state")
    lines = []
    action = report.get("action", "unknown")
    pool = ETF_POOL

    def name_of(sym):
        return f"{pool.get(sym, sym[:4])}({sym})" if sym else ""

    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    # ── 第一部分：今日信号 ──
    execd = report.get("order_executed")
    blocked = report.get("order_blocked")
    risk = report.get("risk")
    has_signal = False

    if execd:
        t = execd.get("type", "")
        if t == "buy":
            lines.append(f"  >> 今日信号: 开仓执行 买入{name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f}")
        elif t == "sell":
            lines.append(f"  >> 今日信号: 卖出执行 {name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f} 盈亏{execd.get('pnl', 0):+.2f}")
        elif t == "switch":
            s = execd.get("sell", {}); b = execd.get("buy", {})
            lines.append(f"  >> 今日信号: 切换执行 {name_of(s.get('symbol',''))} -> {name_of(b.get('symbol',''))}")
        has_signal = True

    if blocked:
        lines.append(f"  >> 今日信号: 订单取消: {blocked.get('reason', '')}")
        has_signal = True

    if state and state.pending_order:
        po = state.pending_order
        pa = po.get("action", "?")
        if pa == "buy":
            lines.append(f"  >> 今日信号: 买入信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "sell":
            lines.append(f"  >> 今日信号: 卖出信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "switch":
            lines.append(f"  >> 今日信号: 切换信号 {name_of(po['sell_symbol'])}->{name_of(po['buy_symbol'])}（明日执行）")
        lines.append(f"      原因: {po.get('reason', '')}")
        has_signal = True

    if risk and risk.get("triggered"):
        lines.append(f"  >> 今日信号: {risk['reason']}")
        has_signal = True

    if not has_signal:
        if action == "hold":
            h = name_of(state.position.symbol) if state and state.position.shares > 0 else ""
            lines.append(f"  >> 今日信号: 持有 {h}，无新信号")
        elif action == "hold_cash":
            lines.append(f"  >> 今日信号: 空仓观望，无买入信号")
        else:
            lines.append(f"  >> 今日信号: 无新信号 ({action})")

    # 排名
    ranking = report.get("ranking", {})
    if ranking:
        rank_parts = []
        for rk in range(1, min(len(ranking) + 1, 4)):
            sym = ranking.get(str(rk))
            if sym:
                rank_parts.append(f"#{rk} {name_of(sym)}")
        if rank_parts:
            lines.append(f"      综合排名: {' > '.join(rank_parts)}")

    # ── 第二部分：账户日结 ──
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
            total_return = (total_value / state.initial_capital - 1) * 100
            lines.append(f"    总资产: {total_value:>8.2f}  总收益率: {total_return:+8.2f}%")

    lines.append(f"  ===========================================")
    return lines


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

    # 2. 最新数据检查
    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None:
        msg = "数据库无 ETF 数据，跳过"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    if latest_day != today_str:
        msg = f"最新数据日为 {latest_day}，非今日 {today_str}，跳过（数据未同步）"
        logger.warning(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 3. 加载行情数据（含指数数据）
    lookback = max(MOMENTUM_WINDOW * 3, 90)  # 60日动量需更多历史
    etf_data = load_etf_and_index_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback)
    if not etf_data:
        msg = "行情数据加载失败"
        logger.error(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 4. 找今日索引
    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx = df.index[mask][0]
            if idx >= 60:  # 至少60日数据计算长期动量
                today_idx = idx
                break

    if today_idx is None:
        msg = f"在数据中未找到 {today_str} 的完整行情（动量窗口数据不足）"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # 5. 初始化模拟盘组件
    state_mgr = StateManager(str(STATE_FILE_DIR), "composite_momentum")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(
        state_mgr=state_mgr,
        broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_composite_signals,
        rank_func=rank_etfs_by_composite,
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

    # 6. 运行
    report = engine.run_daily(etf_data, today_idx, today_str)

    if "error" in report:
        logger.error(report["error"])
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 记录模拟盘日志
    append_simulation_log(STRATEGY_NAME, report, ETF_POOL)

    # 7. 推送日报
    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)

    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
