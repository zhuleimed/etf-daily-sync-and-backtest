"""
市场宽度择时策略 — 每日模拟盘运行入口

宽度 > 70% → 正常动量轮动
宽度 < 30% → 空仓避险
宽度 30-70% → 空仓避险（NEUTRAL_MODE=cash）

调用方式（由 pipeline.py 在数据同步完成后触发）：
    python -m simulation.strategies.market_breadth.daily
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

from simulation.strategies.market_breadth.config import (
    ETF_POOL, ETF_SYMBOLS,
    MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, DB_PATH,
    INITIAL_CAPITAL, RISK_MODE,
    STOP_LOSS_PCT, PROFIT_THRESHOLD,
    DRAWBACK_PCT, DRAWDOWN_THRESHOLD,
    STATE_FILE_DIR,
    BREADTH_MA_PERIOD, BREADTH_STRONG, BREADTH_WEAK, NEUTRAL_MODE,
)

from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)

logger = logging.getLogger("market_breadth_sim")

STRATEGY_NAME = "市场宽度择时模拟盘"


# ============================================================================
# 宽度计算（与回测引擎保持一致）
# ============================================================================

def compute_breadh(etf_data, date_idx, ma_period=20):
    """计算市场宽度：价格 > MA(N) 的 ETF 占比。"""
    if date_idx < ma_period:
        return 0.5
    above, total = 0, 0
    for sym, df in etf_data.items():
        if sym not in ETF_SYMBOLS:
            continue
        if date_idx >= len(df):
            continue
        total += 1
        close = df.iloc[date_idx]["close"]
        ma = df.iloc[max(0, date_idx - ma_period + 1):date_idx + 1]["close"].mean()
        if close > ma:
            above += 1
    return above / max(total, 1)


def determine_regime(breadth):
    """根据宽度确定市场状态。"""
    if breadth >= BREADTH_STRONG:
        return "bull"
    elif breadth <= BREADTH_WEAK:
        return "bear"
    else:
        return "neutral"


def should_stay_out(regime):
    """当前市场状态是否应该空仓。"""
    if regime == "bear":
        return True
    if regime == "neutral":
        return NEUTRAL_MODE == "cash"
    return False


# ============================================================================
# 日结报告
# ============================================================================

def build_report(report: dict) -> list[str]:
    """统一格式日结报告。"""
    state = report.get("state")
    lines = []

    def name_of(sym):
        return f"{ETF_POOL.get(sym, sym[:4])}({sym})"

    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    # 宽度和状态
    breadth = report.get("breadth", 0)
    regime = report.get("regime", "?")
    lines.append(f"  市场宽度: {breadth:.1%}  状态: {regime}")

    # 信号
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
        lines.append(f"      原因: {po.get('reason', '')}")
        has_signal = True

    if risk and risk.get("triggered"):
        lines.append(f"  >> 今日信号: {risk['reason']}")
        has_signal = True

    if not has_signal:
        action = report.get("action", "unknown")
        if action == "hold":
            h = name_of(state.position.symbol) if state and state.position.shares > 0 else ""
            lines.append(f"  >> 今日信号: 持有 {h}，无新信号")
        elif action == "hold_cash":
            lines.append(f"  >> 今日信号: 空仓观望（宽度{regime}，无买入信号）")
        elif action == "breadth_exit":
            lines.append(f"  >> 今日信号: 宽度{regime}触发清仓，空仓避险")
        else:
            lines.append(f"  >> 今日信号: {action}")

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

    # 账户日结
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


# ============================================================================
# 主流程
# ============================================================================

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

    # 2. 数据检查
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

    # 3. 加载行情数据
    lookback = max(MOMENTUM_WINDOW * 2, 40)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback,
                                momentum_window=MOMENTUM_WINDOW)
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
            if idx >= MOMENTUM_WINDOW:
                today_idx = idx
                break
    if today_idx is None:
        msg = f"在数据中未找到 {today_str} 的完整行情"
        logger.warning(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # 5. 计算市场宽度
    breadth = compute_breadh(etf_data, today_idx, BREADTH_MA_PERIOD)
    regime = determine_regime(breadth)
    logger.info(f"市场宽度: {breadth:.1%}  regime: {regime}")

    # 6. 初始化组件
    state_mgr = StateManager(str(STATE_FILE_DIR), "market_breadth")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)

    # 7. 宽度检查：是否需要清仓避险
    if should_stay_out(regime):
        state = state_mgr.load()
        has_position = state and state.position and state.position.shares > 0

        if has_position:
            # 手动清仓
            pos = state.position
            hold_sym = pos.symbol
            # 用今日开盘价卖出
            if hold_sym in etf_data:
                open_px = float(etf_data[hold_sym].iloc[today_idx]["open"])
                result = broker.sell(state, open_px, reason=f"宽度{regime}清仓避险")
                if result and getattr(result, "success", False):
                    state_mgr.save(state)
                    logger.info(f"宽度{regime}触发清仓: 卖出{hold_sym}")
                    total_value = state.cash
                    report = {
                        "date": today_str, "action": "breadth_exit",
                        "breadth": breadth, "regime": regime,
                        "state": state,
                        "order_executed": {
                            "type": "sell", "symbol": hold_sym,
                            "shares": getattr(result, "shares", 0),
                            "price": open_px,
                            "pnl": getattr(result, "pnl", 0),
                        },
                        "stock_value": 0,
                        "total_value": total_value,
                    }
                else:
                    msg = f"宽度清仓卖出失败: {getattr(result, 'reason', str(result))}"
                    logger.error(msg)
                    report = {"date": today_str, "action": "breadth_exit",
                              "breadth": breadth, "regime": regime,
                              "state": state, "error": msg,
                              "stock_value": 0, "total_value": state.cash + (
                                  state.position.shares * open_px if state.position else 0)}
            else:
                report = {"date": today_str, "action": "breadth_exit",
                          "breadth": breadth, "regime": regime,
                          "state": state,
                          "error": f"{hold_sym} 不在行情数据中",
                          "stock_value": 0, "total_value": state.cash}
        else:
            logger.info(f"宽度{regime}，空仓观望")
            report = {
                "date": today_str, "action": "hold_cash",
                "breadth": breadth, "regime": regime,
                "state": state,
                "stock_value": 0,
                "total_value": state.cash if state else INITIAL_CAPITAL,
            }

        # 记录日志
        append_simulation_log("market_breadth", STRATEGY_NAME, report, ETF_POOL)

        # 推送报告
        report_lines = build_report(report)
        for line in report_lines:
            logger.info(line)
        push_daily_report(STRATEGY_NAME, report_lines)
        logger.info(f"{STRATEGY_NAME} 完成 ✓")
        return

    # 8. 宽度允许 → 正常动量轮动
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

    report = engine.run_daily(etf_data, today_idx, today_str)

    if "error" in report:
        logger.error(report["error"])
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 附加宽度信息
    report["breadth"] = breadth
    report["regime"] = regime

    # 记录日志
    append_simulation_log("market_breadth", STRATEGY_NAME, report, ETF_POOL)

    # 推送报告
    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)

    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
