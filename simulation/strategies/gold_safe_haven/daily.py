"""
黄金避险轮动策略 — 每日模拟盘运行入口

双模式：
  NORMAL → 动量轮动（在7只宽基ETF中选最强）
  GOLD   → 持有黄金ETF(518880)避险

恐慌指数由收盘数据计算，执行用次日开盘价（T+1模型）。
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

import numpy as np
import pandas as pd

from simulation.framework.state import StateManager
from simulation.framework.broker import SimBroker
from simulation.framework.data import (
    load_latest_data,
    get_latest_trading_day,
    is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log
from simulation.framework import sim_db

from simulation.strategies.gold_safe_haven.config import (
    ETF_POOL, ETF_SYMBOLS, BROAD_SYMBOLS, GOLD_SYMBOL,
    MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, INITIAL_CAPITAL,
    RISK_MODE, STOP_LOSS_PCT, PROFIT_THRESHOLD,
    DRAWBACK_PCT, DRAWDOWN_THRESHOLD,
    STRATEGY_NAME, STRATEGY_ID, STATE_FILE_DIR,
    PANIC_THRESHOLD, MIN_GOLD_HOLD, GOLD_MAX_HOLD,
    GOLD_STOP_LOSS, PANIC_EXIT_THRESHOLD, ZSCORE_WINDOW,
)

from strategies.gold_safe_haven.panic_signals import (
    compute_panic_score, get_broad_avg_5d_return, init_panic_history,
)
from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)

logger = logging.getLogger("gold_safe_haven_sim")


# ============================================================================
# 报告生成
# ============================================================================

def build_report(report: dict) -> list[str]:
    """生成微信推送日报。"""
    state = report.get("state")
    lines = []

    def name_of(sym):
        return f"{ETF_POOL.get(sym, sym[:4])}({sym})"

    lines.append("")
    lines.append("  ═══════════════════════════════════════════")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ═══════════════════════════════════════════")

    # 今日信号
    execd = report.get("order_executed")
    blocked = report.get("order_blocked")
    risk = report.get("risk")
    has_signal = False

    if execd:
        t = execd.get("type", "")
        if t == "buy":
            lines.append(f"  >> 今日信号: 买入 {name_of(execd['symbol'])} {execd['shares']}股 @ {execd['price']:.4f}")
        elif t == "sell":
            lines.append(f"  >> 今日信号: 卖出 {name_of(execd['symbol'])} {execd['shares']}股 盈亏{execd.get('pnl', 0):+.2f}")
        elif t == "switch":
            s = execd.get("sell", {}); b = execd.get("buy", {})
            lines.append(f"  >> 今日信号: 切换 {name_of(s.get('symbol',''))} -> {name_of(b.get('symbol',''))}")
        lines.append(f"      原因: {execd.get('reason', '')}")
        has_signal = True

    if blocked:
        lines.append(f"  >> 订单取消: {blocked.get('reason', '')}")
        has_signal = True

    if state and state.pending_order:
        po = state.pending_order
        pa = po.get("action", "?")
        mode_tag = f"[{report.get('mode','?')}] "
        if pa == "buy":
            lines.append(f"  >> {mode_tag}买入信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "sell":
            lines.append(f"  >> {mode_tag}卖出信号 {name_of(po['symbol'])}（明日执行）")
        elif pa == "switch":
            lines.append(f"  >> {mode_tag}切换 {name_of(po['sell_symbol'])} → {name_of(po['buy_symbol'])}（明日执行）")
        lines.append(f"      原因: {po.get('reason', '')}")
        has_signal = True

    if not has_signal:
        h = ""
        if state and state.position.shares > 0:
            h = name_of(state.position.symbol)
        lines.append(f"  >> 今日信号: 持有 {h} [{report.get('mode', '?')}]")

    # panic指标
    panic_info = report.get("panic_info", {})
    if panic_info:
        lines.append(f"      恐慌分={panic_info.get('panic_score', 0):.2f} "
                     f"最大5d跌幅={panic_info.get('max_dd', 0)*100:.1f}% "
                     f"广度={panic_info.get('breadth', 0)*100:.0f}%")

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
    lines.append(f"  ───────────────────────────────────────────")
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

    lines.append(f"  ═══════════════════════════════════════════")
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

    # 1. 交易日检查
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

    # 3. 加载行情数据（加载足够长的历史用于Z-score计算）
    lookback = max(ZSCORE_WINDOW + 100, MOMENTUM_WINDOW * 2)
    etf_data = load_latest_data(ETF_SYMBOLS, DB_PATH, lookback_days=lookback, momentum_window=MOMENTUM_WINDOW)
    if not etf_data:
        msg = "行情数据加载失败"
        logger.error(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # 4. 找到今天的索引
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

    # 5. 初始化组件
    state_mgr = StateManager(str(STATE_FILE_DIR), STRATEGY_ID)
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)

    # 6. 加载/初始化状态
    state = state_mgr.load()
    is_first_run = state is None
    if is_first_run:
        state = state_mgr.init_new(INITIAL_CAPITAL)
        state.strategy_name = STRATEGY_NAME

    # 7. 重建恐慌历史（逐日迭代，回填Z-score计算所需的历史数据）
    panic_history = init_panic_history()
    for i in range(max(1, MOMENTUM_WINDOW), today_idx + 1):
        # 为每一天计算恐慌分（最后一天today_idx用于实际决策）
        signal_i = max(1, i - 1)  # 用前一天数据算，避免look-ahead
        try:
            result = compute_panic_score(etf_data, signal_i, panic_history)
        except Exception:
            continue

    # 8. 获取今日恐慌信号
    # 用today_idx的数据计算（对应T日close）
    # 注意：compute_panic_score已追加到panic_history，所以需要重新计算
    panic_history = init_panic_history()
    signal_idx = max(1, today_idx - 1)
    for i in range(max(1, MOMENTUM_WINDOW), signal_idx + 1):
        try:
            compute_panic_score(etf_data, max(1, i - 1), panic_history)
        except Exception:
            continue

    panic_result = compute_panic_score(etf_data, signal_idx, panic_history)
    is_panic = panic_result["is_panic"]

    # 确定当前持仓模式
    pos = state.position
    is_holding_gold = (pos.symbol == GOLD_SYMBOL and pos.shares > 0)
    is_holding_broad = (pos.symbol in BROAD_SYMBOLS and pos.shares > 0)
    has_position = pos.shares > 0

    # 9. 计算动量排名（正常模式用）
    momentum = compute_momentum_signals(etf_data, signal_idx, MOMENTUM_WINDOW)
    broad_momentum = momentum[momentum.index.isin(BROAD_SYMBOLS)]
    ranking = rank_etfs_by_momentum(broad_momentum)
    top_etf = ranking.get(1) if len(ranking) > 0 else None

    # 10. 构建today_data用于估值
    today_data = {}
    for sym, df in etf_data.items():
        if today_idx < len(df):
            row = df.iloc[today_idx]
            today_data[sym] = {
                "open": float(row["open"]), "high": float(row["high"]),
                "low": float(row["low"]), "close": float(row["close"]),
                "volume": float(row["volume"]),
            }

    # 估值
    stock_value = 0.0
    if has_position and pos.symbol in today_data:
        stock_value = pos.shares * today_data[pos.symbol]["close"]
    total_value = state.cash + stock_value

    # 更新峰值
    if total_value > state.peak_value:
        state.peak_value = total_value

    # 11. 执行昨日的待执行订单
    report = {
        "date": today_str,
        "state": state,
        "action": "hold",
        "stock_value": stock_value,
        "total_value": total_value,
        "mode": "gold" if is_holding_gold else "normal",
        "panic_info": panic_result,
        "ranking": {str(k): v for k, v in ranking.items()} if len(ranking) > 0 else {},
    }

    pending = state.pending_order
    if pending:
        action = pending.get("action", "")
        target_sym = pending.get("symbol", pending.get("buy_symbol", ""))
        sell_sym = pending.get("sell_symbol", "")

        # 检查涨跌停/停牌
        blocked_reason = ""
        if action == "buy" and target_sym in today_data:
            td = today_data[target_sym]
            if td["volume"] == 0:
                blocked_reason = f"{target_sym} 停牌"
            elif target_sym in ("159915", "159949", "588000", "588080", "588050"):
                if td["open"] >= td.get("prev_close", td["open"]) * 1.20:
                    blocked_reason = f"{target_sym} 涨停(20%)"
            else:
                if td["open"] >= td.get("prev_close", td["open"]) * 1.10:
                    blocked_reason = f"{target_sym} 涨停(10%)"

        if action == "sell" and sell_sym in today_data:
            td = today_data[sell_sym]
            if td["volume"] == 0:
                blocked_reason = f"{sell_sym} 停牌"
            elif sell_sym in ("159915", "159949", "588000", "588080", "588050"):
                if td["open"] <= td.get("prev_close", td["open"]) * 0.80:
                    blocked_reason = f"{sell_sym} 跌停(20%)"
            else:
                if td["open"] <= td.get("prev_close", td["open"]) * 0.90:
                    blocked_reason = f"{sell_sym} 跌停(10%)"

        if not blocked_reason:
            # 执行订单
            if action == "buy":
                trade = broker.buy(state, target_sym, today_data[target_sym]["open"], reason=pending.get("reason", ""))
                if trade.success:
                    report["order_executed"] = {"type": "buy", "symbol": target_sym,
                        "shares": trade.shares, "price": trade.price, "reason": pending.get("reason", "")}
                state.pending_order = None
            elif action == "sell":
                trade = broker.sell(state, today_data[sell_sym]["open"], reason=pending.get("reason", ""))
                if trade.success:
                    report["order_executed"] = {"type": "sell", "symbol": sell_sym,
                        "shares": trade.shares, "price": trade.price, "pnl": trade.pnl, "reason": pending.get("reason", "")}
                    sim_db.record_closed_trade(str(STATE_FILE_DIR), STRATEGY_ID, {
                        "sell_date": today_str, "symbol": sell_sym, "shares": trade.shares,
                        "sell_price": trade.price, "pnl": trade.pnl, "exit_reason": pending.get("reason", ""),
                        "buy_date": pos.buy_date, "buy_price": pos.avg_cost,
                    })
                state.pending_order = None
            elif action == "switch":
                # 先卖后买
                sell_trade = broker.sell(state, today_data[sell_sym]["open"], reason=pending.get("reason", ""))
                if sell_trade.success:
                    buy_trade = broker.buy(state, target_sym, today_data[target_sym]["open"], reason=pending.get("reason", ""))
                    if buy_trade.success:
                        report["order_executed"] = {
                            "type": "switch",
                            "sell": {"symbol": sell_sym, "shares": sell_trade.shares, "price": sell_trade.price, "pnl": sell_trade.pnl},
                            "buy": {"symbol": target_sym, "shares": buy_trade.shares, "price": buy_trade.price},
                            "reason": pending.get("reason", ""),
                        }
                        sim_db.record_closed_trade(str(STATE_FILE_DIR), STRATEGY_ID, {
                            "sell_date": today_str, "symbol": sell_sym, "shares": sell_trade.shares,
                            "sell_price": sell_trade.price, "pnl": sell_trade.pnl, "exit_reason": pending.get("reason", ""),
                            "buy_date": pos.buy_date, "buy_price": pos.avg_cost,
                        })
                state.pending_order = None
        else:
            report["order_blocked"] = {"reason": blocked_reason}
            state.pending_order = None

    # 重新估值（执行后）
    pos = state.position
    has_position = pos.shares > 0
    stock_value = 0.0
    if has_position and pos.symbol in today_data:
        stock_value = pos.shares * today_data[pos.symbol]["close"]
    total_value = state.cash + stock_value

    is_holding_gold = (pos.symbol == GOLD_SYMBOL and pos.shares > 0)
    is_holding_broad = (pos.symbol in BROAD_SYMBOLS and pos.shares > 0)
    if total_value > state.peak_value:
        state.peak_value = total_value

    # 12. 信号计算（只在没有待执行订单时）
    if not state.pending_order:
        # 今日开仓保护
        if pos.today_opened:
            report["action"] = "hold"
        else:
            new_order = _generate_signal(
                state, is_holding_gold, is_holding_broad,
                is_panic, panic_result, top_etf, momentum,
                today_data, today_str, today_idx, etf_data,
            )
            state.pending_order = new_order

    # 确定当前模式
    mode = "gold" if is_holding_gold else "normal"
    if state.pending_order:
        pa = state.pending_order.get("action", "")
        if pa in ("buy", "switch") and state.pending_order.get("symbol", state.pending_order.get("buy_symbol", "")) == GOLD_SYMBOL:
            mode = "gold(pending)"
        if pa == "sell" and is_holding_gold:
            mode = "gold(exiting)"

    report.update({
        "state": state,
        "stock_value": stock_value,
        "total_value": total_value,
        "mode": mode,
    })

    # 13. 保存状态
    state_mgr.save(state)

    # 14. 记录CSV日志和SQLite快照
    append_simulation_log(STRATEGY_ID, STRATEGY_NAME, report, ETF_POOL)
    try:
        sim_db.record_account_daily(str(STATE_FILE_DIR), STRATEGY_ID, today_str,
            state.cash, stock_value, total_value, pos.symbol, pos.shares)
    except Exception as e:
        logger.warning(f"SQLite日结记录失败: {e}")

    # 15. 推送日报
    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)

    logger.info(f"{STRATEGY_NAME} 完成 ✓ mode={mode} panic={is_panic}")


# ============================================================================
# 信号生成逻辑
# ============================================================================

def _generate_signal(
    state, is_holding_gold, is_holding_broad,
    is_panic, panic_result, top_etf, momentum,
    today_data, today_str, today_idx, etf_data,
):
    """生成待执行订单（次日开盘执行）。"""
    pos = state.position
    has_position = pos.shares > 0

    # ── 黄金模式下的处理 ──
    if is_holding_gold:
        # 1. 黄金止损检查
        if GOLD_SYMBOL in etf_data and today_idx >= 5:
            gold_ret_5d = float(etf_data[GOLD_SYMBOL].loc[today_idx, "ret_5d"])
            if not pd.isna(gold_ret_5d) and gold_ret_5d <= GOLD_STOP_LOSS:
                return {
                    "action": "sell", "symbol": GOLD_SYMBOL,
                    "sell_symbol": GOLD_SYMBOL,
                    "reason": f"黄金止损(5d={gold_ret_5d*100:.1f}%)",
                    "created": today_str,
                }

        # 2. 最小持仓期未过 → 持有
        days_in_gold = state.days_since_switch
        if days_in_gold < MIN_GOLD_HOLD:
            return None

        # 3. 检查恐慌解除
        avg_5d = get_broad_avg_5d_return(etf_data, today_idx)
        panic_exited = avg_5d >= PANIC_EXIT_THRESHOLD
        force_exit = days_in_gold >= GOLD_MAX_HOLD

        if panic_exited or force_exit:
            reason = f"恐慌解除(avg_5d={avg_5d*100:.1f}%)" if panic_exited else f"黄金到期({GOLD_MAX_HOLD}天)"
            # 如果有宽基信号，切换到宽基
            if top_etf:
                target_mom = momentum.get(top_etf, 0)
                if not pd.isna(target_mom) and target_mom > 0:
                    return {
                        "action": "switch",
                        "sell_symbol": GOLD_SYMBOL,
                        "buy_symbol": top_etf,
                        "reason": f"{reason}→买入{top_etf}",
                        "created": today_str,
                    }
            # 否则只卖黄金
            return {
                "action": "sell", "symbol": GOLD_SYMBOL,
                "sell_symbol": GOLD_SYMBOL,
                "reason": reason,
                "created": today_str,
            }

        # 继续持有黄金
        return None

    # ── 正常模式下的处理 ──
    # 恐慌触发 → 切换到黄金
    if is_panic:
        if is_holding_broad:
            return {
                "action": "switch",
                "sell_symbol": pos.symbol,
                "buy_symbol": GOLD_SYMBOL,
                "reason": f"恐慌触发(score={panic_result['panic_score']:.2f})",
                "created": today_str,
            }
        elif not has_position:
            return {
                "action": "buy", "symbol": GOLD_SYMBOL,
                "reason": f"恐慌触发(score={panic_result['panic_score']:.2f})",
                "created": today_str,
            }

    # 正常动量轮动
    if not has_position:
        if top_etf:
            target_mom = momentum.get(top_etf, 0)
            if not pd.isna(target_mom) and target_mom > 0:
                return {
                    "action": "buy", "symbol": top_etf,
                    "reason": f"动量开仓(动量={target_mom*100:.1f}%)",
                    "created": today_str,
                }
        return None

    # 有宽基持仓 → 检查是否切换
    if is_holding_broad and top_etf and top_etf != pos.symbol:
        if state.days_since_switch < MIN_HOLD_DAYS:
            return None
        target_mom = momentum.get(top_etf, 0)
        hold_mom = momentum.get(pos.symbol, 0)
        if pd.isna(target_mom) or pd.isna(hold_mom):
            return None
        excess = target_mom - hold_mom
        if excess > MIN_SWITCH_CONVICTION:
            return {
                "action": "switch",
                "sell_symbol": pos.symbol,
                "buy_symbol": top_etf,
                "reason": f"动量切换(excess={excess*100:.2f}%)",
                "created": today_str,
            }

    return None


if __name__ == "__main__":
    main()
