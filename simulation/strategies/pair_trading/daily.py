"""
配对交易风格轮动 — 每日模拟盘入口

逻辑：
  利用 z-score 判断大盘价值(上证50/沪深300) vs 成长(创业板/科创50)的风格切换。
  3对 ETF 中取 |z| 最大信号，全仓切换至"便宜方"。

与 momentum_rotation 共享 simulation/framework/ 的状态管理和数据加载。
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.framework.state import StateManager, SimState
from simulation.framework.broker import SimBroker
from simulation.framework.data import (
    load_latest_data, get_latest_trading_day, is_trading_day,
)
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log

from simulation.strategies.pair_trading.config import (
    PAIRS, INITIAL_CAPITAL, ZSCORE_PERIOD, ZSCORE_OPEN,
    ZSCORE_CLOSE, ZSCORE_STOP, COMMISSION_RATE, SLIPPAGE, DB_PATH,
    STRATEGY_NAME, STATE_FILE_DIR,
)
from strategies.pair_trading.engine_switch import compute_pair_signals

logger = logging.getLogger("pair_trading_sim")


def build_report(report: dict) -> list[str]:
    """统一格式日结报告（与momentum_rotation格式一致）。"""
    state = report.get("state")
    lines = []
    action = report.get("action", "unknown")
    # 配对交易只涉及4只ETF，就地定义名称映射
    _pt_names = {"510050": "上证50", "510300": "沪深300", "159915": "创业板", "588000": "科创50"}

    def name_of(sym):
        return f"{_pt_names.get(sym, sym[:4])}({sym})"

    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {report.get('date', '')}")
    lines.append(f"  ===========================================")

    # 第一部分：今日信号
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
        if action == "hold" or action == "hold_cash":
            h = name_of(state.position.symbol) if state and state.position.shares > 0 else ""
            if h:
                lines.append(f"  >> 今日信号: 持有 {h}，无新切换信号")
            else:
                lines.append(f"  >> 今日信号: 空仓观望，无买入信号")
        elif action == "risk_pending":
            pass  # 已在risk中显示
        else:
            lines.append(f"  >> 今日信号: 无新信号 ({action})")

    # 第二部分：账户日结
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


def run_sim_daily(
    state_mgr: StateManager,
    broker: SimBroker,
    etf_data: dict[str, pd.DataFrame],
    today_idx: int,
    today_str: str,
) -> dict:
    """执行一个交易日的配对交易模拟盘流程。

    T+1 待执行订单模式：
      1. 加载状态
      2. 执行昨日待执行订单（今日open）
      3. 计算 z-score 信号
      4. 产生新的待执行订单（明日执行）
      5. 持久化状态
    """
    state = state_mgr.load() or state_mgr.init_new(INITIAL_CAPITAL)
    state.last_update = today_str
    state.days_since_switch += 1

    today_data = {}
    for sym in set(p["a"] for p in PAIRS) | set(p["b"] for p in PAIRS):
        df = etf_data.get(sym)
        if df is not None and today_idx < len(df):
            today_data[sym] = {
                "open": float(df.iloc[today_idx]["open"]),
                "high": float(df.iloc[today_idx]["high"]),
                "low": float(df.iloc[today_idx]["low"]),
                "close": float(df.iloc[today_idx]["close"]),
                "volume": float(df.iloc[today_idx]["volume"]),
            }

    if not today_data:
        return {"error": f"{today_str} 无行情数据", "state": state}

    report = {
        "date": today_str, "action": "hold",
        "hold_symbol": state.position.symbol if state.position.shares > 0 else "",
        "hold_shares": state.position.shares,
        "cash": state.cash, "stock_value": 0.0, "total_value": state.cash,
        "trade": None, "risk": None,
        "order_executed": None, "order_blocked": None,
    }

    hold_sym = state.position.symbol

    # ── 执行昨日待执行订单 ──
    order = state.pending_order
    state.pending_order = None

    if order is not None:
        action_type = order.get("action")
        result, blocked = _execute_order(action_type, order, state, today_data, broker, today_str)
        if result:
            report["order_executed"] = result
            report["action"] = f"exec_{action_type}"
            hold_sym = state.position.symbol
        if blocked:
            report["order_blocked"] = blocked

    # ── 估值（用今日 close） ──
    stock_value = 0.0
    if hold_sym and hold_sym in today_data:
        stock_value = state.position.shares * today_data[hold_sym]["close"]
    total_value = state.cash + stock_value

    report["hold_symbol"] = hold_sym or ""
    report["hold_shares"] = state.position.shares
    report["stock_value"] = stock_value
    report["total_value"] = total_value

    # ── 计算信号（用今日 close，T+1待执行） ──
    # 注意：传入 today_idx+1，因为 compute_pair_signals 内部 signal_idx = idx-1
    # 这样 signal_idx = (today_idx+1)-1 = today_idx，使用今日收盘价计算信号
    signals = compute_pair_signals(etf_data, today_idx + 1)
    ranking = {}
    for i, s in enumerate(signals, 1):
        if s["z"]:
            ranking[str(i)] = s.get("target") or ""
    report["ranking"] = ranking

    has_position = state.position.shares > 0

    # ── 止损检查 ──
    stop_triggered = False
    if has_position and hold_sym and signals:
        for s in signals:
            if hold_sym in (s["pair_cfg"]["a"], s["pair_cfg"]["b"]):
                if abs(s["z"]) > s["stop_threshold"]:
                    stop_triggered = True
                    break
    if stop_triggered:
        report["risk"] = {"triggered": True, "reason": f"止损: |z|>{ZSCORE_STOP}"}
        if state.position.shares > 0:
            state.pending_order = {
                "action": "sell", "symbol": hold_sym,
                "reason": "止损", "created": today_str,
            }
            report["action"] = "risk_pending"
        state_mgr.save(state)
        report["state"] = state
        return report

    # ── 平仓检查 |z| < close ──
    if has_position and hold_sym:
        should_close = False
        for s in signals:
            if hold_sym in (s["pair_cfg"]["a"], s["pair_cfg"]["b"]):
                if abs(s["z"]) < s["close_threshold"] and abs(s["z"]) > 0:
                    should_close = True
                    break
        if should_close:
            state.pending_order = {
                "action": "sell", "symbol": hold_sym,
                "reason": "价差回归平仓", "created": today_str,
            }
            report["action"] = "close_pending"
            state_mgr.save(state)
            report["state"] = state
            return report

    # ── 开仓/切换信号 ──
    valid = [s for s in signals if s["target"] is not None and s["strength"] > 0 and s["volume_ok"]]
    if valid:
        best = max(valid, key=lambda s: s["strength"])
        target = best["target"]

        if not has_position:
            state.pending_order = {
                "action": "buy", "symbol": target,
                "reason": f"z-score={best['z']:.2f} 开仓",
                "created": today_str,
            }
            report["action"] = "open_pending"
        elif target != hold_sym and state.days_since_switch >= 5:
            state.pending_order = {
                "action": "switch",
                "sell_symbol": hold_sym, "buy_symbol": target,
                "reason": f"z-score={best['z']:.2f} 切换",
                "created": today_str,
            }
            report["action"] = "switch_pending"

    state_mgr.save(state)
    report["state"] = state
    return report


def _execute_order(action_type, order, state, today_data, broker, today_str):
    """执行一种订单类型。返回 (executed_info, blocked_info)。"""
    if action_type == "buy":
        sym = order["symbol"]
        if sym not in today_data:
            return None, {"type": "buy", "symbol": sym, "reason": "无行情"}
        # 停牌/零成交量检查
        if today_data[sym].get("volume", 0) == 0:
            return None, {"type": "buy", "symbol": sym, "reason": f"{sym} 停牌/零成交量"}
        open_px = today_data[sym]["open"]
        # broker.buy() 内部会自动加滑点，传原始 open 价即可
        result = broker.buy(state, sym, open_px, reason=order.get("reason", ""))
        if result.success:
            state.days_since_switch = 0
            return {"type": "buy", "symbol": sym, "shares": result.shares,
                    "price": result.price}, None
        return None, {"type": "buy", "symbol": sym, "reason": result.reason}

    if action_type == "sell":
        sym = order["symbol"]
        if sym not in today_data or state.position.shares <= 0:
            return None, {"type": "sell", "symbol": sym, "reason": "无持仓"}
        # 停牌/零成交量检查
        if today_data[sym].get("volume", 0) == 0:
            return None, {"type": "sell", "symbol": sym, "reason": f"{sym} 停牌/零成交量"}
        result = broker.sell(state, today_data[sym]["open"], reason=order.get("reason", ""))
        if result.success:
            return {"type": "sell", "symbol": sym, "shares": result.shares,
                    "price": result.price, "pnl": result.pnl}, None
        return None, {"type": "sell", "symbol": sym, "reason": result.reason}

    if action_type == "switch":
        sell_sym = order["sell_symbol"]
        buy_sym = order["buy_symbol"]
        if sell_sym not in today_data or buy_sym not in today_data:
            return None, {"type": "switch", "reason": "无行情"}
        # 停牌检查（任一停牌则取消）
        if today_data[sell_sym].get("volume", 0) == 0:
            return None, {"type": "switch", "reason": f"{sell_sym} 停牌/零成交量"}
        if today_data[buy_sym].get("volume", 0) == 0:
            return None, {"type": "switch", "reason": f"{buy_sym} 停牌/零成交量"}
        sell_result = broker.sell(state, today_data[sell_sym]["open"], reason="切换卖出")
        if not sell_result.success:
            return None, {"type": "switch", "reason": f"卖出失败: {sell_result.reason}"}
        buy_result = broker.buy(state, buy_sym, today_data[buy_sym]["open"],
                                reason="切换买入")
        state.days_since_switch = 0
        return {
            "type": "switch",
            "sell": {"symbol": sell_sym, "shares": sell_result.shares,
                     "price": sell_result.price, "pnl": sell_result.pnl},
            "buy": {"symbol": buy_sym, "shares": buy_result.shares,
                    "price": buy_result.price},
        }, None

    return None, {"type": "unknown", "reason": f"未知类型: {action_type}"}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
        return

    latest_day = get_latest_trading_day([p["a"] for p in PAIRS] + [p["b"] for p in PAIRS])
    if latest_day is None:
        push_daily_report(STRATEGY_NAME, ["数据库无数据，跳过"])
        return
    if latest_day != today_str:
        push_error_alert(STRATEGY_NAME, f"最新数据 {latest_day}，非今日 {today_str}")
        return

    lookback = max(ZSCORE_PERIOD * 2, 60)
    etf_data = load_latest_data(
        [p["a"] for p in PAIRS] + [p["b"] for p in PAIRS],
        DB_PATH, lookback_days=lookback, momentum_window=ZSCORE_PERIOD,
    )
    if not etf_data:
        push_error_alert(STRATEGY_NAME, "数据加载失败")
        return

    today_idx = None
    for sym, df in etf_data.items():
        mask = df["date"] == today_str
        if mask.any():
            idx = df.index[mask][0]
            if idx >= ZSCORE_PERIOD:
                today_idx = idx
                break
    if today_idx is None:
        push_daily_report(STRATEGY_NAME, ["今日数据不足（需要ZSCORE_PERIOD天历史）"])
        return

    state_mgr = StateManager(str(STATE_FILE_DIR), "pair_trading")
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    report = run_sim_daily(state_mgr, broker, etf_data, today_idx, today_str)

    if "error" in report:
        push_error_alert(STRATEGY_NAME, report["error"])
        return

    # 记录模拟盘日志
    append_simulation_log(STRATEGY_NAME, report, ETF_POOL)

    report_lines = build_report(report)
    for line in report_lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, report_lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
