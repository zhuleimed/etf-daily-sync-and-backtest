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
from simulation.framework.log_writer import append_simulation_log
from simulation.framework.report_builder import build_signal_report

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
    SHORT_TERM_MOMENTUM_CHECK,
)

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
    short_term_momentum_ok,
)

logger = logging.getLogger("momentum_rotation_sim")

STRATEGY_NAME = "动量轮动模拟盘"

# ── 信号上下文（供切换闸门使用，2026-10-02 接线）──
# 回测 _make_decision_single 在切换前有一道"短期动量确认"：
#   目标 ETF 近 5 日跌幅 ≤ -0.5% → 不换（避免追跌）；动能衰减（5日均涨 < 20日均涨）→ 不换。
# 模拟盘此前完全没有这道门 → 09-15 换了 510050（近5日 -1.39%），回测没换，
# 造成 11 个交易日的持仓偏离。上下文在信号函数里记录，离线重放 import 即忠实复现。
_sig_ctx: dict = {}


def compute_momentum_signals_live(
    etf_data: dict,
    today_idx: int,
    momentum_window: int = 20,
):
    """live 信号入口：记录上下文后调用策略信号函数。"""
    _sig_ctx["etf_data"] = etf_data
    _sig_ctx["idx"] = today_idx
    return compute_momentum_signals(etf_data, today_idx, momentum_window)


def momentum_switch_gate(momentum, target_etf: str, hold_sym: str) -> bool:
    """切换闸门：与回测"短期动量确认"同款（False=本次不切换）。

    规则体在 strategies.momentum_rotation.momentum_signals.short_term_momentum_ok
    （vol_filter 共用同一实现，避免两处抄写漂移）。
    模拟盘口径：today_idx 即回测的 signal_idx（同一信息集），故 check_idx = today_idx。
    """
    etf_data, i = _sig_ctx.get("etf_data"), _sig_ctx.get("idx")
    if etf_data is None or i is None:
        return True
    ok = short_term_momentum_ok(etf_data, i, momentum, target_etf, SHORT_TERM_MOMENTUM_CHECK)
    if not ok:
        logger.info(f"切换闸门：目标 {target_etf} 短期动量确认未通过（近5日弱势/动能衰减），不切换")
    return ok


def build_report(report: dict) -> list[str]:
    """统一格式日结报告：信号部分 + 账户日结部分。"""
    state = report.get("state")
    lines = []
    action = report.get("action", "unknown")
    from .config import ETF_POOL as pool

    def name_of(sym):
        return f"{pool.get(sym, sym[:4])}({sym})"

    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    # �� 第一部分：今日信号
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

    # 动量排名
    ranking = report.get("ranking", {})
    if ranking:
        rank_parts = []
        for rk in range(1, min(len(ranking) + 1, 4)):
            sym = ranking.get(str(rk))
            if sym:
                rank_parts.append(f"#{rk} {name_of(sym)}")
        if rank_parts:
            lines.append(f"      动量排名: {' > '.join(rank_parts)}")

    # �� 第二部分：账户日结
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

    # 同日幂等防重：若本交易日已被处理过(state.last_update==today)，跳过。
    # 2026-09-05排查: 08-18 曾多次重跑致同日伪切换(563000→512100)，干净单日引擎不切换。
    _prev = state_mgr.load()
    if _prev is not None and str(getattr(_prev, "last_update", "")) == today_str:
        logger.info(f"{today_str} 本交易日已处理，跳过（防同日重复重跑制造假信号）")
        return

    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    engine = DailySimEngine(
        state_mgr=state_mgr,
        broker=broker,
        config={"initial_capital": INITIAL_CAPITAL},
        signal_func=compute_momentum_signals_live,   # 记录上下文供切换闸门用
        rank_func=rank_etfs_by_momentum,
        etf_pool=ETF_POOL,
        momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION,
        min_hold_days=MIN_HOLD_DAYS,
        switch_gate_func=momentum_switch_gate,       # 回测同款"短期动量确认"（2026-10-02）
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
    append_simulation_log("momentum_rotation", STRATEGY_NAME, report, ETF_POOL)

    # 7. 推送日报
    report_lines = build_signal_report(report, STRATEGY_NAME, ETF_POOL)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)

    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
