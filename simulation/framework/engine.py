"""
模拟盘核心引擎 — 每日流程编排（T+1 待执行订单模式）

市场正确的时间线：
  T日 20:20（数据同步后）
    → 用 T日 close 计算信号
    → 产生"待执行订单"（买/卖/切换）
    → 保存到状态文件，今日不交易

  T+1日 20:20
    → 执行昨日待执行订单（用 T+1日 open，检查涨停/跌停）
    → 用 T+1日 close 算新信号
    → 产生新的待执行订单...
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Callable, Optional

import pandas as pd

from .state import SimState, StateManager, PositionState
from .broker import SimBroker
from .risk import RiskResult, run_all_risk_checks

from simulation.framework import sim_db

logger = logging.getLogger(__name__)

# ── 涨跌停限制（不同类型 ETF 不同） ──
LIMIT_20PCT_SYMBOLS = {"159915", "588000", "159949", "588080", "588050"}


def _check_limit_open(
    symbol: str,
    open_price: float,
    prev_close: float,
) -> tuple[bool, str]:
    """检查开盘价是否涨停或跌停。

    Args:
        symbol: ETF 代码。
        open_price: 开盘价。
        prev_close: 前收盘价。

    Returns:
        (is_blocked, reason) — 是否被封锁及原因。
    """
    if prev_close <= 0:
        return False, ""
    limit_pct = 0.20 if symbol in LIMIT_20PCT_SYMBOLS else 0.10
    upper = round(prev_close * (1 + limit_pct), 4)
    lower = round(prev_close * (1 - limit_pct), 4)
    if open_price >= upper:
        return True, f"{symbol} 涨停（前收{prev_close:.4f}→开盘{open_price:.4f}≥{upper:.4f}）"
    if open_price <= lower:
        return True, f"{symbol} 跌停（前收{prev_close:.4f}→开盘{open_price:.4f}≤{lower:.4f}）"
    return False, ""


def _check_suspended(today_data: dict, symbol: str) -> tuple[bool, str]:
    """检查标的当日是否停牌（成交量=0）。"""
    if symbol not in today_data:
        return True, f"{symbol} 无行情数据"
    volume = today_data[symbol].get("volume", 0)
    if volume == 0:
        return True, f"{symbol} 停牌/零成交量"
    return False, ""


def _risk_cooldown_active(engine: "DailySimEngine") -> bool:
    """风控清仓后回补冷却是否仍生效（仅离线实验，默认 cooldown=0 恒 False）。"""
    c = engine.risk_exit_reentry_cooldown
    if c <= 0:
        return False
    bar0 = engine._risk_exit_bar
    if bar0 is None:
        return False
    return (engine._bar - bar0) < c


class DailySimEngine:
    """每日模拟盘引擎（T+1 待执行订单模式）。"""

    def __init__(
        self,
        state_mgr: StateManager,
        broker: SimBroker,
        config: dict[str, Any],
        signal_func: Callable,
        rank_func: Callable,
        etf_pool: dict[str, str],
        momentum_window: int = 20,
        min_switch_conviction: float = 0.03,
        min_hold_days: int = 10,
        risk_mode: str = "A",
        stop_loss_pct: float = 0.05,
        profit_threshold: float = 0.10,
        drawback_pct: float = 0.05,
        drawdown_threshold: float = 0.15,
        exit_when_signal_dead: bool = False,
        switch_threshold_func: Optional[Callable] = None,
        confirm_days: int = 1,
        risk_exit_reentry_cooldown: int = 0,
    ):
        self.state_mgr = state_mgr
        self.broker = broker
        self.config = config
        self.signal_func = signal_func
        self.rank_func = rank_func
        self.etf_pool = etf_pool
        self.momentum_window = momentum_window
        self.min_switch_conviction = min_switch_conviction
        self.min_hold_days = min_hold_days
        self.risk_mode = risk_mode
        # 持仓信号消失即卖出（与回测引擎"持仓得分归零→平仓"行为对齐，
        # 2026-08-03 新增：RSI 等策略启用，修复模拟盘不回测不一致）
        self.exit_when_signal_dead = exit_when_signal_dead
        # 自定义切换阈值函数（momentum, target, current)->float，
        # None=用绝对差 min_switch_conviction（默认行为不变）
        self.switch_threshold_func = switch_threshold_func
        self.stop_loss_pct = stop_loss_pct
        self.profit_threshold = profit_threshold
        self.drawback_pct = drawback_pct
        self.drawdown_threshold = drawdown_threshold

        # ── 修复摩擦的离线实验杠杆（默认 OFF，live 字节不变）──
        # 促销到真盘需持久化计数（SimState 版本化字段）；此处仅引擎实例级，
        # 供 simulation.analysis.repair_bt 的单进程回放计数使用。
        # confirm_days: 开仓/切换目标需连续 N 日保持 rank1/动量>0 才发 pending
        #               （平滑单日假 leader）。=1 即当日行为，不改变现状。
        # risk_exit_reentry_cooldown: 风控触发卖出后，需在现金里至少 N 巴才允许
        #               再次开仓（抑制"风控清仓→立刻追高回补"摩擦）。=0 不生效。
        self.confirm_days = max(int(confirm_days), 1)
        self.risk_exit_reentry_cooldown = max(int(risk_exit_reentry_cooldown), 0)
        # 进程内跨日计数（单一引擎复用时确定）：
        self._bar = 0                       # 已运行交易日计数
        self._confirm = {"sym": None, "days": 0}   # 当前连续确认的目标
        self._risk_exit_bar: Optional[int] = None  # 风控卖出 pending 创建的巴号

    # ── 主入口 ──

    def run_daily(
        self,
        etf_data: dict[str, pd.DataFrame],
        today_idx: int,
        today_str: str,
    ) -> dict[str, Any]:
        """执行一个交易日的模拟盘流程。

        步骤顺序：
          1. 加载/初始化状态
          2. 重置 today_opened（昨天的持仓今天可以卖了）
          3. 构建今日行情 + 前日收盘价
          4. 执行昨日待执行订单（用今日 open，检查涨跌停）
          5. 更新最高价 + 总资产（用今日 close 估值）
          6. 风控检查 → 产生风险待执行订单
          7. 计算动量信号 → 产生信号待执行订单
          8. 持久化状态
          9. 生成日报
        """
        # ── 1. 加载状态 ──
        state = self.state_mgr.load()
        if state is None:
            state = self.state_mgr.init_new(self.config.get("initial_capital", 10000))
        state.last_update = today_str
        state.days_since_switch += 1
        self._bar += 1

        # ── 2. 重置 T+1 标记（昨天的持仓今天可以卖了） ──
        state.position.today_opened = False

        # ── 3. 构建今日行情 ──
        today_data = {}
        prev_close = {}
        for sym in self.etf_pool:
            df = etf_data.get(sym)
            if df is not None and today_idx < len(df):
                today_data[sym] = {
                    "open": df.iloc[today_idx]["open"],
                    "high": df.iloc[today_idx]["high"],
                    "low": df.iloc[today_idx]["low"],
                    "close": df.iloc[today_idx]["close"],
                    "volume": df.iloc[today_idx]["volume"],
                }
                if today_idx > 0:
                    prev_close[sym] = df.iloc[today_idx - 1]["close"]

        if not today_data:
            return {"error": f"{today_str} 无行情数据", "state": state}

        report = {
            "date": today_str,
            "action": "hold",
            "hold_symbol": state.position.symbol if state.position.shares > 0 else "",
            "hold_shares": state.position.shares,
            "cash": state.cash,
            "stock_value": 0.0,
            "total_value": state.cash,
            "trade": None,
            "risk": None,
            "order_executed": None,
            "order_blocked": None,
        }

        hold_sym = state.position.symbol

        # ── 4. 执行昨日的待执行订单（用今日 open，检查涨跌停） ──
        order_executed, order_blocked = self._execute_pending_order(
            state, today_data, prev_close, today_str,
        )
        if order_executed:
            report["order_executed"] = order_executed
            report["action"] = {
                "buy": "open", "sell": "risk_sell", "switch": "switch",
            }.get(order_executed.get("type", ""), "trade")
            # 切换后可能已有新持仓
            hold_sym = state.position.symbol
        if order_blocked:
            report["order_blocked"] = order_blocked
            report["action"] = "order_blocked"

        # ── 5. 更新最高价 + 总资产（用今日 close 估值） ──
        if hold_sym and hold_sym in today_data:
            today_close = today_data[hold_sym]["close"]
            if today_close > state.position.highest_price:
                state.position.highest_price = today_close

        stock_value = 0.0
        if hold_sym and hold_sym in today_data:
            stock_value = state.position.shares * today_data[hold_sym]["close"]
        total_value = state.cash + stock_value
        if total_value > state.peak_value:
            state.peak_value = total_value
        state.total_value = total_value  # 供 combined 读取最新估值

        report["hold_symbol"] = hold_sym if hold_sym else ""
        report["hold_shares"] = state.position.shares
        report["stock_value"] = stock_value
        report["total_value"] = total_value

        # ── 6. 风控检查（RISK_MODE B/C 时触发） ──
        risk_result = run_all_risk_checks(
            state=state,
            current_price=today_data[hold_sym]["close"] if hold_sym and hold_sym in today_data else 0,
            current_value=total_value,
            mode=self.risk_mode,
            stop_loss_pct=self.stop_loss_pct,
            profit_threshold=self.profit_threshold,
            drawback_pct=self.drawback_pct,
            drawdown_threshold=self.drawdown_threshold,
            peak_value=state.peak_value,
        )
        if risk_result and risk_result.triggered:
            report["risk"] = {"triggered": True, "reason": risk_result.reason}
            if state.position.shares > 0:
                # 风控产生待执行卖出订单（明天以 open 执行）
                state.pending_order = {
                    "action": "sell",
                    "symbol": hold_sym,
                    "reason": risk_result.reason,
                    "created": today_str,
                }
                report["action"] = "risk_pending"
                # 记录风控卖出 pending 的巴号，作为回补冷却计时起点（默认不启用）
                if self.risk_exit_reentry_cooldown > 0:
                    self._risk_exit_bar = self._bar
            self.state_mgr.save(state)
            report["state"] = state
            # 写入每日快照（风控触发时也有估值数据）
            try:
                sim_db.record_account_daily({
                    "date": today_str,
                    "strategy": self.state_mgr.strategy_name,
                    "strategy_name": type(self).__name__,
                    "cash": round(state.cash, 2),
                    "stock_value": round(stock_value, 2),
                    "total_value": round(total_value, 2),
                    "total_return": round((total_value / state.initial_capital - 1), 6)
                        if state.initial_capital > 0 else 0,
                    "position_symbol": state.position.symbol if state.position.shares > 0 else "",
                    "position_shares": state.position.shares,
                })
            except Exception:
                logger.exception("写入每日快照失败")
            return report

        # ── 7. 计算今日信号（明日开盘执行的依据） ──
        # 信号计算永远执行——今日算出的信号是给明天用的，
        # 与"今日执行了什么"（第4步）是两件独立的事。
        momentum = self.signal_func(etf_data, today_idx, self.momentum_window)
        ranking = self.rank_func(momentum)
        target_etf = ranking.get(1) if len(ranking) > 0 else None
        target_mom = momentum.get(target_etf, float("nan")) if target_etf else float("nan")

        report["ranking"] = {str(k): str(v) for k, v in ranking.items()} if len(ranking) > 0 else {}

        # ── 计算信号动作（供日报展示），但不一定产生 pending_order ──
        has_position = state.position.shares > 0
        current_mom = momentum.get(hold_sym, float("nan")) if has_position else float("nan")

        # ── 持仓信号消失 → 卖出（可选模式 exit_when_signal_dead）──
        # 与回测引擎"持仓得分归零→平仓"行为对齐；不受 min_hold 限制（同回测）。
        # RSI 等策略启用：2026-07 模拟盘教训——持仓得分归零时模拟盘只"持有"，
        # 回测却已平仓避险，导致模拟盘多亏 8~10pp。
        if has_position and self.exit_when_signal_dead and (
            pd.isna(current_mom) or current_mom <= 0
        ):
            report["signal"] = "exit_pending"
            report["signal_note"] = "持仓信号消失（得分≤0），明日卖出"
            report["signal_target"] = hold_sym

        elif not has_position:
            if target_etf is not None and not pd.isna(target_mom) and target_mom > 0:
                report["signal"] = "open_pending"
                report["signal_target"] = target_etf
            else:
                report["signal"] = "hold_cash"

        elif state.days_since_switch < self.min_hold_days:
            # min_hold 期内：不产生切换 pending_order，但信号本身有效
            # 卖出信号（风控触发/信号消失）不受 min_hold 限制
            report["signal"] = "hold"
            report["signal_note"] = f"min_hold={self.min_hold_days}天（已持{state.days_since_switch}天）"

        elif target_etf is None or pd.isna(target_mom):
            report["signal"] = "hold"

        elif hold_sym in today_data:
            if not pd.isna(current_mom) and not pd.isna(target_mom):
                # 切换阈值：默认绝对差；可自定义（RSI 用 2.0×截面标准差，与回测一致）
                if self.switch_threshold_func is not None:
                    threshold = self.switch_threshold_func(momentum, target_etf, hold_sym)
                else:
                    threshold = self.min_switch_conviction
                excess = target_mom - current_mom
                if excess > threshold:
                    report["signal"] = "switch_pending"
                    report["signal_target"] = target_etf
                else:
                    report["signal"] = "hold"
        else:
            report["signal"] = "hold"

        # ── 产生待执行订单（仅当日无已执行订单且无被封锁订单时才创建） ──
        # "已执行订单"指第4步执行的昨信号，不影响今日信号的计算但影响
        # 是否创建新的 pending_order（避免同一天既有执行又产生新订单）
        can_create_pending = (
            not report.get("order_executed")
            and not report.get("order_blocked")
        )

        if can_create_pending:
            signal = report["signal"]
            # 开仓/切换目标需连续 confirm_days 日保持同一目标才发（平滑单日假 leader）。
            # confirm_days==1(默认)：不进入计数，恒 confirmed=True，行为与现状字节一致。
            # confirm_days>1 ：仅 signal 为 open/switch 当日推进 streak，目标一变即清零。
            target_today = signal in ("open_pending", "switch_pending")
            confirmed = True
            if target_today and target_etf is not None and self.confirm_days > 1:
                if self._confirm["sym"] == target_etf:
                    self._confirm["days"] += 1
                else:
                    self._confirm = {"sym": target_etf, "days": 1}
                confirmed = self._confirm["days"] >= self.confirm_days

            if signal == "exit_pending":
                state.pending_order = {
                    "action": "sell", "symbol": hold_sym,
                    "reason": "持仓信号消失", "created": today_str,
                }
            # open/switch 需过 confirm；且现金开的 open 还要过风控清仓回补冷却
            elif signal == "open_pending" and confirmed and not _risk_cooldown_active(self):
                state.pending_order = {
                    "action": "buy", "symbol": target_etf,
                    "reason": "动量信号开仓", "created": today_str,
                }
            elif signal == "switch_pending" and confirmed:
                state.pending_order = {
                    "action": "switch", "sell_symbol": hold_sym,
                    "buy_symbol": target_etf, "reason": "动量切换", "created": today_str,
                }
        # 当 can_create_pending=False 时：
        #   - 信号已计算（report["signal"]）供日报展示
        #   - pending_order 不创建（今天已执行了订单，明天不应再执行）
        #   - 下一次运行（明天）会重新计算信号

        # ── 8. 持久化状态 ──
        self.state_mgr.save(state)
        report["state"] = state

        # ── 9. 写入每日资产快照（sim_trading.db） ──
        try:
            sim_db.record_account_daily({
                "date": today_str,
                "strategy": self.state_mgr.strategy_name,
                "strategy_name": type(self).__name__,
                "cash": round(state.cash, 2),
                "stock_value": round(stock_value, 2),
                "total_value": round(total_value, 2),
                "total_return": round((total_value / state.initial_capital - 1), 6)
                    if state.initial_capital > 0 else 0,
                "position_symbol": state.position.symbol if state.position.shares > 0 else "",
                "position_shares": state.position.shares,
            })
        except Exception:
            logger.exception("写入每日快照失败")

        return report

    # ── 待执行订单处理 ──

    # ── 已平仓交易记录 ──

    def _record_closed_trade(
        self,
        state: SimState,
        order: dict,
        snapshot: dict,
        result,
        today_str: str,
    ) -> None:
        """记录已平仓交易到 sim_trading.db。"""
        try:
            cost = snapshot.get("total_cost", 0)
            pnl = getattr(result, "pnl", 0) if hasattr(result, "pnl") else result.get("pnl", 0)
            trade_record = {
                "strategy": self.state_mgr.strategy_name,
                "symbol": snapshot.get("symbol", ""),
                "etf_name": self.etf_pool.get(snapshot.get("symbol", ""),
                                              snapshot.get("symbol", "")[:4]),
                "action": order.get("action", "sell"),
                "buy_date": snapshot.get("buy_date", ""),
                "sell_date": today_str,
                "hold_days": snapshot.get("hold_days", 0),
                "buy_price": snapshot.get("avg_cost", 0),
                "sell_price": getattr(result, "price", 0) if hasattr(result, "price") else result.get("price", 0),
                "shares": snapshot.get("shares", 0),
                "total_cost": round(cost, 2),
                "net_revenue": round(cost + pnl, 2),
                "commission": getattr(result, "commission", 0) if hasattr(result, "commission") else result.get("commission", 0),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl / cost, 6) if cost > 0 else 0,
                "exit_reason": order.get("reason", ""),
            }
            sim_db.record_closed_trade(trade_record)
        except Exception:
            logger.exception("记录已平仓交易失败")

    # ── 待执行订单处理 ──

    def _execute_pending_order(
        self,
        state: SimState,
        today_data: dict,
        prev_close: dict[str, float],
        today_str: str,
    ) -> tuple[Optional[dict], Optional[dict]]:
        """执行昨日的待执行订单。

        执行规则：
          - 用今日开盘价交易
          - 涨跌停时取消订单（不能交易）
          - 切换订单：卖旧和买新任一被封锁 → 全部取消

        Returns:
            (executed_info, blocked_info)，两者互斥。
        """
        order = state.pending_order
        if order is None:
            return None, None

        # 原子性：取出即清除（执行或取消都不再保留）
        state.pending_order = None
        action = order.get("action")

        if action == "buy":
            sym = order["symbol"]
            if sym not in today_data:
                return None, {"type": "buy", "symbol": sym, "reason": "无行情数据"}
            # 停牌检查
            suspended, susp_reason = _check_suspended(today_data, sym)
            if suspended:
                return None, {"type": "buy", "symbol": sym, "reason": susp_reason}
            open_px = today_data[sym]["open"]
            pc = prev_close.get(sym, 0)
            blocked, reason = _check_limit_open(sym, open_px, pc)
            if blocked:
                return None, {"type": "buy", "symbol": sym, "reason": reason}
            result = self.broker.buy(state, sym, open_px, reason=order.get("reason", ""))
            if result.success:
                state.days_since_switch = 0
                return {"type": "buy", "symbol": sym, "shares": result.shares,
                        "price": result.price}, None
            return None, {"type": "buy", "symbol": sym, "reason": result.reason}

        if action == "sell":
            sym = order["symbol"]
            if sym not in today_data:
                return None, {"type": "sell", "symbol": sym, "reason": "无行情数据"}
            if state.position.shares <= 0:
                return None, {"type": "sell", "symbol": sym, "reason": "无持仓"}
            # 停牌检查
            suspended, susp_reason = _check_suspended(today_data, sym)
            if suspended:
                return None, {"type": "sell", "symbol": sym, "reason": susp_reason}
            open_px = today_data[sym]["open"]
            pc = prev_close.get(sym, 0)
            blocked, reason = _check_limit_open(sym, open_px, pc)
            if blocked:
                return None, {"type": "sell", "symbol": sym, "reason": reason}
            # ── 保存持仓快照（broker.sell 会清空 position） ──
            sell_snapshot = {
                "symbol": sym,
                "shares": state.position.shares,
                "avg_cost": state.position.avg_cost,
                "total_cost": state.position.total_cost,
                "hold_days": state.days_since_switch,
                "buy_date": state.position.buy_date,
            }
            result = self.broker.sell(state, open_px, reason=order.get("reason", ""))
            if result.success:
                self._record_closed_trade(state, order, sell_snapshot, result, today_str)
                return {"type": "sell", "symbol": sym, "shares": result.shares,
                        "price": result.price, "pnl": result.pnl}, None
            return None, {"type": "sell", "symbol": sym, "reason": result.reason}

        if action == "switch":
            sell_sym = order["sell_symbol"]
            buy_sym = order["buy_symbol"]
            if sell_sym not in today_data or buy_sym not in today_data:
                return None, {"type": "switch", "reason": "标的无行情数据"}

            # 停牌检查（任一停牌则取消）
            susp_sell, reason_sell_s = _check_suspended(today_data, sell_sym)
            susp_buy, reason_buy_s = _check_suspended(today_data, buy_sym)
            if susp_sell or susp_buy:
                reasons = []
                if susp_sell: reasons.append(reason_sell_s)
                if susp_buy: reasons.append(reason_buy_s)
                return None, {"type": "switch", "reason": "；".join(reasons)}

            # 涨跌停检查
            open_sell = today_data[sell_sym]["open"]
            open_buy = today_data[buy_sym]["open"]
            pc_sell = prev_close.get(sell_sym, 0)
            pc_buy = prev_close.get(buy_sym, 0)

            blocked_sell, reason_sell = _check_limit_open(sell_sym, open_sell, pc_sell)
            blocked_buy, reason_buy = _check_limit_open(buy_sym, open_buy, pc_buy)
            if blocked_sell or blocked_buy:
                reasons = []
                if blocked_sell:
                    reasons.append(reason_sell)
                if blocked_buy:
                    reasons.append(reason_buy)
                return None, {"type": "switch", "reason": "；".join(reasons)}

            # 保存卖出快照（broker.sell 会清空 position）
            sell_snapshot = {
                "symbol": sell_sym,
                "shares": state.position.shares,
                "avg_cost": state.position.avg_cost,
                "total_cost": state.position.total_cost,
                "hold_days": state.days_since_switch,
                "buy_date": state.position.buy_date,
            }
            # 卖旧
            sell_result = self.broker.sell(state, open_sell, reason="动量切换卖出")
            if not sell_result.success:
                return None, {"type": "switch", "reason": f"卖出失败: {sell_result.reason}"}
            # 记录已平仓交易（卖出部分）
            self._record_closed_trade(state, {
                "action": "switch_sell", "reason": "动量切换",
            }, sell_snapshot, sell_result, today_str)
            # 买新
            buy_result = self.broker.buy(state, buy_sym, open_buy, reason="动量切换买入")
            state.days_since_switch = 0
            return {
                "type": "switch",
                "sell": {"symbol": sell_sym, "shares": sell_result.shares, "price": sell_result.price, "pnl": sell_result.pnl},
                "buy": {"symbol": buy_sym, "shares": buy_result.shares, "price": buy_result.price},
            }, None

        return None, {"type": "unknown", "reason": f"未知订单类型: {action}"}
