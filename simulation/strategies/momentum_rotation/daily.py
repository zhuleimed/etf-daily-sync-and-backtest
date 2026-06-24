"""
动量轮动策略 — 每日模拟盘运行入口

调用方式（由 pipeline.py 在数据同步完成后触发）：
    python -m simulation.strategies.momentum_rotation.daily

也可以独立运行测试：
    python -m simulation.strategies.momentum_rotation.daily
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

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

from simulation.strategies.momentum_rotation.config import (
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
    STATE_FILE_DIR,
)

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
)

logger = logging.getLogger("momentum_rotation_sim")

STRATEGY_NAME = "动量轮动模拟盘"


def build_report(report: dict) -> list[str]:
    """从引擎日报 dict 构建推送文本行。"""
    state = report.get("state")
    lines = []
    action = report.get("action", "unknown")

    # 操作
    action_map = {
        "open_pending": "📝 待开仓（明日执行）",
        "switch_pending": "📝 待切换（明日执行）",
        "risk_pending": "📝 待风控卖出（明日执行）",
        "hold": "⏸ 持有",
        "hold_cash": "💵 空仓",
        "order_blocked": "🚫 涨跌停封锁",
    }
    lines.append(f"操作: {action_map.get(action, action)}")

    # 已执行的订单
    execd = report.get("order_executed")
    if execd:
        t = execd.get("type", "")
        if t == "buy":
            lines.append(f"✅ 开仓执行: 买入{execd['symbol']} {execd['shares']}股 × {execd['price']:.4f}")
        elif t == "sell":
            lines.append(f"✅ 卖出执行: {execd['symbol']} {execd['shares']}股 × {execd['price']:.4f} 盈亏{execd.get('pnl', 0):+.2f}")
        elif t == "switch":
            s = execd.get("sell", {})
            b = execd.get("buy", {})
            lines.append(f"✅ 切换执行: 卖{s.get('symbol')} {s.get('shares')}股({s.get('pnl',0):+.2f}) → 买{b.get('symbol')} {b.get('shares')}股")

    # 被封锁的订单
    blocked = report.get("order_blocked")
    if blocked:
        lines.append(f"🚫 订单取消: {blocked.get('reason', '未知')}")

    # 风险
    risk = report.get("risk")
    if risk and risk.get("triggered"):
        lines.append(f"⚠️ 风控触发: {risk['reason']}")

    # 持仓 & 资金
    if state:
        pos = state.position
        if pos and pos.shares > 0:
            stock_val = report.get("stock_value", 0)
            lines.append(f"持仓: {pos.symbol} {pos.shares}股 均价{pos.avg_cost:.4f}")
            lines.append(f"市值: ¥{stock_val:.2f}")
        lines.append(f"现金: ¥{state.cash:.2f}")
        lines.append(f"总资产: ¥{report.get('total_value', 0):.2f}")
        total_return = (report["total_value"] / state.initial_capital - 1) * 100
        lines.append(f"总收益率: {total_return:+.2f}%")

        # 待执行订单
        if state.pending_order:
            po = state.pending_order
            lines.append(f"待执行: {po.get('action', '?')} {po.get('symbol', po.get('buy_symbol', ''))}")

    # 动量排名
    ranking = report.get("ranking", {})
    if ranking:
        from strategies.momentum_rotation.config import ETF_POOL as pool
        rank_lines = []
        for rk in range(1, min(len(ranking) + 1, 4)):
            sym = ranking.get(str(rk))
            if sym:
                name = pool.get(sym, sym)
                rank_lines.append(f"  #{rk} {name}")
        if rank_lines:
            lines.append("动量排名:")
            lines.extend(rank_lines)

    return lines


def main():
    # 日志
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

    # 2. 判断最新数据是否已到位
    latest_day = get_latest_trading_day(ETF_SYMBOLS)
    if latest_day is None:
        msg = "数据库无 ETF 数据，跳过"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    if latest_day != today_str:
        msg = f"最新数据日为 {latest_day}，非今日 {today_str}，跳过（可能数据尚未同步）"
        logger.warning(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 3. 加载行情数据（传入动量窗口，确保 momentum 列计算正确）
    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback, momentum_window=MOMENTUM_WINDOW)
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
            idx = df.index[mask][0]  # 原始 DataFrame 中的位置索引
            # 检查是否有足够的历史数据计算动量
            if idx >= MOMENTUM_WINDOW:
                today_idx = idx
                break

    if today_idx is None:
        msg = f"在数据中未找到 {today_str} 的完整行情（可能动量窗口数据不足）"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # 5. 初始化模拟盘组件
    state_mgr = StateManager(str(STATE_FILE_DIR), "momentum_rotation")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
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

    # 6. 运行
    report = engine.run_daily(etf_data, today_idx, today_str)

    if "error" in report:
        logger.error(report["error"])
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 7. 推送日报
    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)

    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
