"""
回测引擎 — 均值回归轮动

与趋势策略完全相反的决策逻辑：
  1. 市场过滤器：沪深300 > MA200 才交易（不做熊市抄底）
  2. 只有超卖（RSI低+%B低+缩量）的ETF才进入候选池
  3. 买入最超卖的ETF → 等待价格回归均值
  4. 回归完成（%B回升/RSI上升）→ 获利了结
  5. 继续下跌（%B跌破止损线）→ 止损
  6. 持有超期 → 时间止损
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    ETF_POOL, ETF_SYMBOLS, INITIAL_CAPITAL,
    COMMISSION_RATE, SLIPPAGE, TAX_RATE,
    ADJUSTMENT_DAYS, DB_PATH, RISK_MODE,
    MIN_HOLD_DAYS, MAX_HOLD_DAYS, SWITCH_CONVICTION_STD,
    OVERSOLD_RSI_THRESHOLD, OVERSOLD_PCT_B_THRESHOLD,
    REVERT_PCT_B_THRESHOLD, REVERT_RSI_THRESHOLD,
    STOP_LOSS_PCT_B,
    MARKET_INDEX, MARKET_MA_PERIOD,
)
from .data import load_all_etf_data, load_index_data, compute_equal_weight_benchmark
from .momentum_signals import (
    compute_reversion_scores,
    rank_etfs_by_reversion,
    compute_pct_b,
    compute_rsi,
    judge_market_regime,
)
from .risk import RiskState, run_all_risk_checks


@dataclass
class DailyRecord:
    date: str = ""
    hold_symbol: str = ""
    hold_shares: int = 0
    hold_close: float = 0.0
    cash: float = 0.0
    stock_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    action: str = "hold"
    regime: str = "neutral"
    top_etf: str = ""
    score: float = 0.0
    pct_b: float = 0.0
    rsi: float = 0.0


@dataclass
class BuyLot:
    date: str = ""
    symbol: str = ""
    shares: int = 0
    price: float = 0.0
    total_cost: float = 0.0


@dataclass
class TradeRecord:
    date: str = ""
    symbol: str = ""
    trade_type: str = ""
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    profit: float = 0.0
    days_held: int = 0
    return_rate: float = 0.0
    reason: str = ""


class BacktestEngine:
    """均值回归轮动回测引擎。"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, risk_mode: str = ""):
        self.initial_capital = initial_capital
        self.risk_mode = risk_mode or RISK_MODE
        self._days_since_last_switch = 999
        self._entry_day = -999  # 记录持仓起始日（用于时间止损）
        self._entry_pct_b = 0.0  # 记录入场时的 %B（用于止损参考）
        self.etf_data: dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.index_data: pd.DataFrame = pd.DataFrame()
        self.equal_weight_data: pd.DataFrame = pd.DataFrame()
        self.positions: dict[str, int] = {}
        self.cash: float = initial_capital
        self.open_buys: list[BuyLot] = []
        self.adjustment_from: str = ""
        self.adjustment_to: str = ""
        self.adjustment_days_left: int = 0
        self.adjustment_total_days: int = 0
        self.risk_state = RiskState()
        self.daily_records: list[DailyRecord] = []
        self.trade_records: list[TradeRecord] = []
        self.total_trade_cost: float = 0.0

    def load_data(self, start_date: str = "2024-01-01", end_date: str = "",
                  db_path: str = DB_PATH) -> "BacktestEngine":
        self.etf_data, self.dates = load_all_etf_data(
            symbols=ETF_SYMBOLS, start_date=start_date, end_date=end_date, db_path=db_path,
        )
        self.index_data = load_index_data(
            symbol=MARKET_INDEX, start_date=start_date, end_date=end_date, db_path=db_path,
        )
        self.equal_weight_data = compute_equal_weight_benchmark(self.etf_data)
        return self

    def run(self) -> "BacktestEngine":
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")
        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(ETF_POOL.get(s, s) for s in ETF_SYMBOLS)}")
        print(f"  超卖阈值: RSI<{OVERSOLD_RSI_THRESHOLD}, %B<{OVERSOLD_PCT_B_THRESHOLD}")
        print(f"  回归了结: %B>{REVERT_PCT_B_THRESHOLD} or RSI>{REVERT_RSI_THRESHOLD}")
        print(f"  市场过滤: HS300>MA{MARKET_MA_PERIOD}")
        mode_names = {"A": "纯信号", "B": "风控全开", "C": "仅极端回撤"}
        print(f"  风控模式: {self.risk_mode} = {mode_names.get(self.risk_mode, self.risk_mode)}")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── 风控检查 ──
            if self.risk_mode != "A" and has_position and hold_symbol:
                hold_row = today_data[hold_symbol]
                total_value = self._calc_total_value(today_data)
                self.risk_state.update_peak(hold_row["high"])
                self.risk_state.update_peak_total_value(total_value)
                risk_action, risk_reason = run_all_risk_checks(
                    self.risk_state, total_value, has_position,
                    hold_symbol, hold_row["high"], hold_row["low"],
                    hold_row["close"], hold_row["atr"], mode=self.risk_mode,
                )
                if risk_action != "none":
                    self._execute_risk_exit(idx, today_data, risk_action, risk_reason)
                    self._record_day(idx, today_data, action_override=risk_action)
                    continue

            # ── 渐进调仓 ──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── 均值回归信号（用 idx-1 避免 look-ahead bias） ──
            signal_idx = max(0, idx - 1)
            index_aligned = None
            if not self.index_data.empty and signal_idx < len(self.index_data):
                index_aligned = self.index_data

            scores = compute_reversion_scores(self.etf_data, signal_idx)
            ranking = rank_etfs_by_reversion(scores)
            spread = compute_score_spread(scores) if 'compute_score_spread' in dir() else scores.std()
            regime_info = judge_market_regime(index_aligned, signal_idx, MARKET_MA_PERIOD)

            target_etf = ranking.get(1) if len(ranking) > 0 else None
            target_score = scores.get(target_etf, np.nan) if target_etf else np.nan

            # 计算当前持仓的实时 %B 和 RSI（用 idx-1，与信号一致）
            hold_pct_b = np.nan
            hold_rsi = np.nan
            if hold_symbol and hold_symbol in today_data:
                close_vals = self.etf_data[hold_symbol]["close"].values[:max(1, signal_idx) + 1]
                if len(close_vals) > 20:
                    hold_pct_b_arr = compute_pct_b(close_vals, 20, 2.0)
                    hold_pct_b = hold_pct_b_arr[-1] if len(hold_pct_b_arr) > 0 else np.nan
                    hold_rsi_arr = compute_rsi(close_vals, 14)
                    hold_rsi = hold_rsi_arr[-1] if len(hold_rsi_arr) > 0 else np.nan

            # ── 决策 ──
            if self.adjustment_days_left <= 0:
                self._make_decision(
                    idx, today_data, hold_symbol, target_etf, target_score,
                    scores, spread, regime_info,
                    hold_pct_b, hold_rsi,
                )

            top_name = target_etf if target_etf else ""
            top_score_val = round(target_score, 4) if not pd.isna(target_score) else 0.0
            self._record_day(
                idx, today_data,
                regime=regime_info["regime"],
                top_etf=top_name,
                score=top_score_val,
                pct_b=hold_pct_b if not pd.isna(hold_pct_b) else 0.0,
                rsi=hold_rsi if not pd.isna(hold_rsi) else 0.0,
            )
            self._days_since_last_switch += 1

        self._close_remaining_positions()
        print(f"  回测完成 ✓")
        return self

    def _get_hold_symbol(self) -> Optional[str]:
        if not self.positions:
            return None
        valid = {k: v for k, v in self.positions.items() if v > 0}
        return max(valid, key=valid.get) if valid else None

    def _calc_total_value(self, today_data: dict) -> float:
        sv = sum(sh * today_data[sym]["close"] for sym, sh in self.positions.items()
                 if sh > 0 and sym in today_data)
        return self.cash + sv

    def _calc_stock_value(self, today_data: dict) -> float:
        sv = sum(sh * today_data[sym]["close"] for sym, sh in self.positions.items()
                 if sh > 0 and sym in today_data)
        return sv

    def _buy(self, symbol: str, amount: float, idx: int, today_data: dict,
             trade_type: str = "买入", reason: str = "") -> int:
        price = today_data[symbol]["open"] * (1 + SLIPPAGE)
        max_shares = int(amount // price // 100) * 100
        if max_shares <= 0:
            return 0
        cost = max_shares * price
        commission = max(cost * COMMISSION_RATE, 0.0)
        total_cost = cost + commission
        if total_cost > self.cash:
            max_shares = int((self.cash * 0.99) // price // 100) * 100
            if max_shares <= 0:
                return 0
            cost = max_shares * price
            commission = max(cost * COMMISSION_RATE, 0.0)
            total_cost = cost + commission
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + max_shares
        self.open_buys.append(BuyLot(
            date=str(today_data[symbol]["date"]),
            symbol=symbol, shares=max_shares, price=price, total_cost=total_cost,
        ))
        self.total_trade_cost += commission
        self.trade_records.append(TradeRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"), symbol=symbol,
            trade_type=trade_type, price=price, shares=max_shares,
            amount=cost, commission=commission, tax=0.0, reason=reason,
        ))
        return max_shares

    def _sell(self, symbol: str, idx: int, today_data: dict,
              trade_type: str = "卖出", reason: str = "") -> tuple[float, float]:
        shares = self.positions.get(symbol, 0)
        if shares <= 0:
            return 0.0, 0.0
        price = today_data[symbol]["open"] * (1 - SLIPPAGE)
        amount = shares * price
        commission = max(amount * COMMISSION_RATE, 0.0)
        tax = amount * TAX_RATE
        net_amount = amount - commission - tax
        remaining = shares
        total_cost_basis = 0.0
        for buy in self.open_buys[:]:
            if buy.symbol != symbol:
                continue
            if remaining <= 0:
                break
            used = min(buy.shares, remaining)
            total_cost_basis += buy.total_cost * (used / buy.shares)
            buy.shares -= used
            remaining -= used
            if buy.shares <= 0:
                self.open_buys.remove(buy)
        profit = net_amount - total_cost_basis
        self.cash += net_amount
        self.positions[symbol] = 0
        if symbol in self.positions:
            del self.positions[symbol]
        self.total_trade_cost += commission
        self.trade_records.append(TradeRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"), symbol=symbol,
            trade_type=trade_type, price=price, shares=shares,
            amount=amount, commission=commission, tax=tax,
            profit=profit, reason=reason,
        ))
        return profit, net_amount

    def _start_adjustment(self, from_sym: str, to_sym: str, total_days: int = ADJUSTMENT_DAYS):
        self.adjustment_from = from_sym
        self.adjustment_to = to_sym
        self.adjustment_days_left = total_days
        self.adjustment_total_days = total_days

    def _execute_adjustment_step(self, idx: int, today_data: dict):
        if self.adjustment_days_left <= 0:
            return
        day_num = self.adjustment_total_days - self.adjustment_days_left + 1
        sell_sym = self.adjustment_from
        buy_sym = self.adjustment_to
        sell_shares = self.positions.get(sell_sym, 0)
        if sell_shares > 0:
            sell_qty = max(sell_shares // self.adjustment_days_left // 100 * 100, 100)
            if sell_qty > sell_shares:
                sell_qty = sell_shares
            price_s = today_data[sell_sym]["open"] * (1 - SLIPPAGE)
            amount_s = sell_qty * price_s
            commission_s = max(amount_s * COMMISSION_RATE, 0.0)
            self.cash += amount_s - commission_s
            self.positions[sell_sym] = sell_shares - sell_qty
            if self.positions[sell_sym] <= 0:
                del self.positions[sell_sym]
            self.trade_records.append(TradeRecord(
                date=self.dates[idx].strftime("%Y-%m-%d"), symbol=sell_sym,
                trade_type="调仓卖出", price=price_s, shares=sell_qty,
                amount=amount_s, commission=commission_s, tax=0.0,
                reason=f"渐进调仓第{day_num}天",
            ))
        buy_amount = self.cash / self.adjustment_days_left
        if buy_amount > 100:
            price_b = today_data[buy_sym]["open"] * (1 + SLIPPAGE)
            buy_qty = int(buy_amount // price_b // 100) * 100
            if buy_qty > 0:
                cost_b = buy_qty * price_b
                commission_b = max(cost_b * COMMISSION_RATE, 0.0)
                self.cash -= cost_b + commission_b
                self.positions[buy_sym] = self.positions.get(buy_sym, 0) + buy_qty
                self.trade_records.append(TradeRecord(
                    date=self.dates[idx].strftime("%Y-%m-%d"), symbol=buy_sym,
                    trade_type="调仓买入", price=price_b, shares=buy_qty,
                    amount=cost_b, commission=commission_b, tax=0.0,
                    reason=f"渐进调仓第{day_num}天",
                ))
        self.adjustment_days_left -= 1

    def _execute_risk_exit(self, idx: int, today_data: dict, risk_action: str, risk_reason: str):
        for sym in list(self.positions.keys()):
            if self.positions[sym] > 0:
                tt = "止损卖出" if "止损" in risk_reason else "止盈卖出" if "止盈" in risk_reason else "极端回撤清仓"
                self._sell(sym, idx, today_data, trade_type=tt, reason=risk_reason)

    def _make_decision(self, idx: int, today_data: dict,
                       hold_symbol: Optional[str], target_etf: Optional[str],
                       target_score: float, scores: pd.Series, spread: float,
                       regime_info: dict, hold_pct_b: float, hold_rsi: float):
        """
        均值回归决策逻辑：

        入场条件：
          - 市场非熊市（HS300 > MA200）
          - 目标ETF超卖（评分>0 = 通过了RSI+%B双重过滤）
          - 得分+阈值检查

        出场条件（任一触发）：
          1. %B回升至 REVERT_PCT_B_THRESHOLD → 价格回归均值，获利了结
          2. RSI回升至 REVERT_RSI_THRESHOLD → 趋势已反转，不再需要抄底
          3. %B跌破 STOP_LOSS_PCT_B → 趋势性下跌，止损
          4. 持仓超过 MAX_HOLD_DAYS → 时间止损
        """
        has_position = hold_symbol is not None

        # ── 有持仓：先检查出场条件 ──
        if has_position:
            exit_reason = self._check_exit_conditions(
                hold_symbol, hold_pct_b, hold_rsi, idx
            )
            if exit_reason:
                self._sell(hold_symbol, idx, today_data, reason=exit_reason)
                has_position = False
                hold_symbol = None

        # ── 入场逻辑 ──
        if not has_position:
            regime = regime_info["regime"]

            # 熊市过滤：不做熊市抄底
            if regime == "bear":
                return

            # 有超卖候选
            if target_etf is not None and not pd.isna(target_score) and target_score > 0:
                self._buy(target_etf, self.cash * 0.98, idx, today_data,
                         reason=f"均值回归买入 {target_etf}（评分={target_score:.4f}）")
                self.risk_state.on_open_position(today_data[target_etf]["open"])
                self._entry_day = idx
                self._days_since_last_switch = 0
                # 记录入场 %B
                close_vals = self.etf_data[target_etf]["close"].values[:idx + 1]
                if len(close_vals) > 20:
                    pct_b_arr = compute_pct_b(close_vals, 20, 2.0)
                    self._entry_pct_b = pct_b_arr[-1] if not np.isnan(pct_b_arr[-1]) else 0.0

    def _check_exit_conditions(self, hold_symbol: str, hold_pct_b: float,
                                hold_rsi: float, idx: int) -> Optional[str]:
        """检查出场条件。返回出场原因字符串，None=不出场。"""

        # 条件1: %B回升至获利线（价格回归均值）
        if not np.isnan(hold_pct_b) and hold_pct_b >= REVERT_PCT_B_THRESHOLD:
            return f"均值回归目标达成：%B={hold_pct_b:.2f}≥{REVERT_PCT_B_THRESHOLD}→获利了结"

        # 条件2: RSI回升至趋势反转线（趋势已逆转）
        if not np.isnan(hold_rsi) and hold_rsi >= REVERT_RSI_THRESHOLD:
            return f"趋势反转确认：RSI={hold_rsi:.1f}≥{REVERT_RSI_THRESHOLD}→卖出"

        # 条件3: %B继续跌破止损线（趋势性下跌）
        if not np.isnan(hold_pct_b) and hold_pct_b <= STOP_LOSS_PCT_B:
            return f"趋势性下跌止损：%B={hold_pct_b:.2f}≤{STOP_LOSS_PCT_B}→止损"

        # 条件4: 持仓超期
        days_held = idx - self._entry_day
        if days_held >= MAX_HOLD_DAYS:
            return f"时间止损：持仓{days_held}天超过{MAX_HOLD_DAYS}天→平仓"

        return None

    def _record_day(self, idx: int, today_data: dict, action_override: str = "",
                    regime: str = "neutral", top_etf: str = "", score: float = 0.0,
                    pct_b: float = 0.0, rsi: float = 0.0):
        total_value = self._calc_total_value(today_data)
        stock_value = self._calc_stock_value(today_data)
        hold_symbol = self._get_hold_symbol() or ""
        hold_close = 0.0
        if hold_symbol and hold_symbol in today_data:
            hold_close = today_data[hold_symbol]["close"]
        prev_total = self.daily_records[-1].total_value if self.daily_records else self.initial_capital
        daily_return = (total_value - prev_total) / prev_total if prev_total > 0 else 0.0
        cumulative_return = total_value / self.initial_capital - 1
        action = action_override or ("hold" if hold_symbol else "hold_cash")
        self.daily_records.append(DailyRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"),
            hold_symbol=hold_symbol,
            hold_shares=self.positions.get(hold_symbol, 0) if hold_symbol else 0,
            hold_close=hold_close, cash=self.cash, stock_value=stock_value,
            total_value=total_value, daily_return=daily_return,
            cumulative_return=cumulative_return, action=action,
            regime=regime, top_etf=top_etf, score=score,
            pct_b=pct_b, rsi=rsi,
        ))

    def _close_remaining_positions(self):
        for sym in list(self.positions.keys()):
            if self.positions[sym] > 0:
                last_date = self.dates[-1].strftime("%Y-%m-%d")
                last_close = self.etf_data[sym].iloc[-1]["close"]
                price = last_close * (1 - SLIPPAGE)
                shares = self.positions[sym]
                amount = shares * price
                commission = max(amount * COMMISSION_RATE, 0.0)
                self.cash += amount - commission
                self.trade_records.append(TradeRecord(
                    date=last_date, symbol=sym, trade_type="虚拟卖出",
                    price=price, shares=shares, amount=amount,
                    commission=commission, tax=0.0, profit=0.0,
                    reason="期末虚拟平仓",
                ))
                self.positions[sym] = 0
                del self.positions[sym]

    def get_daily_df(self) -> pd.DataFrame:
        if not self.daily_records:
            return pd.DataFrame()
        records = [{
            "date": r.date, "hold_symbol": r.hold_symbol,
            "hold_shares": r.hold_shares, "hold_close": r.hold_close,
            "cash": r.cash, "stock_value": r.stock_value,
            "total_value": r.total_value, "daily_return": r.daily_return,
            "cumulative_return": r.cumulative_return, "action": r.action,
            "regime": r.regime, "top_etf": r.top_etf, "score": r.score,
        } for r in self.daily_records]
        df = pd.DataFrame(records)
        df["benchmark_return"] = 0.0
        df["excess_return"] = 0.0
        return df

    def get_trade_df(self) -> pd.DataFrame:
        if not self.trade_records:
            return pd.DataFrame()
        records = [{
            "date": t.date, "symbol": t.symbol, "trade_type": t.trade_type,
            "price": t.price, "shares": t.shares, "amount": t.amount,
            "commission": t.commission, "tax": t.tax, "profit": t.profit,
            "days_held": t.days_held, "return_rate": t.return_rate, "reason": t.reason,
        } for t in self.trade_records]
        return pd.DataFrame(records)


def compute_score_spread(scores: pd.Series) -> float:
    """得分截面标准差。"""
    valid = scores.dropna()
    return float(valid.std()) if len(valid) > 1 else 0.0
