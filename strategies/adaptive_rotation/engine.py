"""
回测引擎 — 自适应轮动

核心创新：根据市场状态动态切换交易逻辑。

牛市 → 动量模式（买强势、追趋势）
  与 momentum_rotation 一致的逻辑

震荡 → 均值回归模式（买超卖、等回归）
  与 mean_reversion_rotation 一致的逻辑

熊市 → 空仓模式（保护本金）
  强制平仓，不再开仓
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .config import (
    ETF_POOL, ETF_SYMBOLS, INITIAL_CAPITAL,
    COMMISSION_RATE, SLIPPAGE, TAX_RATE,
    ADJUSTMENT_DAYS, DB_PATH, RISK_MODE,
    MOM_WINDOW, MOM_MIN_HOLD_DAYS, MOM_SWITCH_CONVICTION,
    REV_OVERSOLD_RSI, REV_OVERSOLD_PCT_B,
    REV_REVERT_PCT_B, REV_REVERT_RSI,
    REV_STOP_LOSS_PCT_B, REV_MAX_HOLD_DAYS, REV_MIN_HOLD_DAYS,
    MARKET_INDEX,
)
from .data import load_all_etf_data, load_index_data, compute_equal_weight_benchmark
from .regime import detect_regime, regime_description
from .signals import (
    compute_adaptive_scores,
    rank_etfs_by_adaptive,
    _compute_pct_b,
    _compute_rsi,
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
    entry_day: int = 0


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
    reason: str = ""


class BacktestEngine:
    """自适应轮动回测引擎。"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, risk_mode: str = ""):
        self.initial_capital = initial_capital
        self.risk_mode = risk_mode or RISK_MODE
        self._days_since_last_switch = 999
        self._entry_day = -999
        self._current_regime = "neutral"
        self.etf_data = {}
        self.dates = pd.DatetimeIndex([])
        self.index_data = pd.DataFrame()
        self.equal_weight_data = pd.DataFrame()
        self.positions: dict[str, int] = {}
        self.cash = initial_capital
        self.open_buys: list[BuyLot] = []
        self.adjustment_from = ""
        self.adjustment_to = ""
        self.adjustment_days_left = 0
        self.adjustment_total_days = 0
        self.risk_state = RiskState()
        self.daily_records: list[DailyRecord] = []
        self.trade_records: list[TradeRecord] = []
        self.total_trade_cost = 0.0

    def load_data(self, start_date="2024-01-01", end_date="", db_path=DB_PATH) -> "BacktestEngine":
        self.etf_data, self.dates = load_all_etf_data(ETF_SYMBOLS, start_date, end_date, db_path)
        self.index_data = load_index_data(MARKET_INDEX, start_date, end_date, db_path)
        self.equal_weight_data = compute_equal_weight_benchmark(self.etf_data)
        return self

    def run(self) -> "BacktestEngine":
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")
        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(ETF_POOL.get(s, s) for s in ETF_SYMBOLS)}")
        print(f"  核心: 牛市→动量  |  震荡→均值回归  |  熊市→空仓")
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
                ra, rr = run_all_risk_checks(self.risk_state, total_value, has_position,
                    hold_symbol, hold_row["high"], hold_row["low"],
                    hold_row["close"], hold_row["atr"], mode=self.risk_mode)
                if ra != "none":
                    self._execute_risk_exit(idx, today_data, ra, rr)
                    self._record_day(idx, today_data, action_override=ra, regime=self._current_regime)
                    continue

            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── 市场状态检测（用T-1数据避免look-ahead） ──
            signal_idx = max(0, idx - 1)
            index_aligned = None
            if not self.index_data.empty and signal_idx < len(self.index_data):
                index_aligned = self.index_data

            regime_info = detect_regime(index_aligned, signal_idx)
            self._current_regime = regime_info["regime"]

            # ── 自适应信号计算 ──
            scores = compute_adaptive_scores(self.etf_data, signal_idx, index_aligned)
            ranking = rank_etfs_by_adaptive(scores)
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            target_score = scores.get(target_etf, np.nan) if target_etf else np.nan

            # ── 决策 ──
            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target_etf,
                                   target_score, scores, regime_info)

            self._record_day(idx, today_data, regime=self._current_regime,
                            top_etf=target_etf or "",
                            score=round(target_score, 4) if not pd.isna(target_score) else 0.0)
            self._days_since_last_switch += 1

        self._close_remaining_positions()
        print(f"  回测完成 ✓")
        return self

    def _get_hold_symbol(self):
        if not self.positions:
            return None
        valid = {k: v for k, v in self.positions.items() if v > 0}
        return max(valid, key=valid.get) if valid else None

    def _calc_total_value(self, today_data):
        sv = sum(sh * today_data[sym]["close"] for sym, sh in self.positions.items() if sh > 0 and sym in today_data)
        return self.cash + sv

    def _calc_stock_value(self, today_data):
        return sum(sh * today_data[sym]["close"] for sym, sh in self.positions.items() if sh > 0 and sym in today_data)

    def _buy(self, symbol, amount, idx, today_data, trade_type="买入", reason=""):
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
        self.open_buys.append(BuyLot(date=str(today_data[symbol]["date"]), symbol=symbol,
                                      shares=max_shares, price=price, total_cost=total_cost))
        self.total_trade_cost += commission
        self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"),
            symbol=symbol, trade_type=trade_type, price=price, shares=max_shares,
            amount=cost, commission=commission, tax=0.0, reason=reason))
        return max_shares

    def _sell(self, symbol, idx, today_data, trade_type="卖出", reason=""):
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
        self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"),
            symbol=symbol, trade_type=trade_type, price=price, shares=shares,
            amount=amount, commission=commission, tax=tax, profit=profit, reason=reason))
        return profit, net_amount

    def _execute_risk_exit(self, idx, today_data, risk_action, risk_reason):
        for sym in list(self.positions.keys()):
            if self.positions[sym] > 0:
                tt = "止损卖出" if "止损" in risk_reason else "止盈卖出" if "止盈" in risk_reason else "极端回撤清仓"
                self._sell(sym, idx, today_data, trade_type=tt, reason=risk_reason)

    def _start_adjustment(self, from_sym, to_sym, total_days=ADJUSTMENT_DAYS):
        self.adjustment_from = from_sym
        self.adjustment_to = to_sym
        self.adjustment_days_left = total_days
        self.adjustment_total_days = total_days

    def _execute_adjustment_step(self, idx, today_data):
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
            self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"),
                symbol=sell_sym, trade_type="调仓卖出", price=price_s, shares=sell_qty,
                amount=amount_s, commission=commission_s, tax=0.0, reason=f"渐进调仓第{day_num}天"))
        buy_amount = self.cash / self.adjustment_days_left
        if buy_amount > 100:
            price_b = today_data[buy_sym]["open"] * (1 + SLIPPAGE)
            buy_qty = int(buy_amount // price_b // 100) * 100
            if buy_qty > 0:
                cost_b = buy_qty * price_b
                commission_b = max(cost_b * COMMISSION_RATE, 0.0)
                self.cash -= cost_b + commission_b
                self.positions[buy_sym] = self.positions.get(buy_sym, 0) + buy_qty
                self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"),
                    symbol=buy_sym, trade_type="调仓买入", price=price_b, shares=buy_qty,
                    amount=cost_b, commission=commission_b, tax=0.0, reason=f"渐进调仓第{day_num}天"))
        self.adjustment_days_left -= 1

    def _make_decision(self, idx, today_data, hold_symbol, target_etf,
                       target_score, scores, regime_info):
        """自适应决策：根据市场状态选择逻辑。"""
        has_position = hold_symbol is not None
        regime = regime_info["regime"]

        # ── 熊市模式：强制平仓，不再开仓 ──
        if regime == "bear":
            if has_position:
                reason = f"市场转熊（{regime_info.get('ratio', 0):+.2%}），强制平仓"
                self._sell(hold_symbol, idx, today_data, reason=reason)
            return

        # ── 牛市模式：动量逻辑 ──
        if regime == "bull":
            if has_position:
                current_score = scores.get(hold_symbol, np.nan)
                has_candidate = target_etf is not None and not pd.isna(target_score) and target_score > 0

                if self._days_since_last_switch < MOM_MIN_HOLD_DAYS:
                    return
                if not has_candidate or target_etf == hold_symbol:
                    return
                if not pd.isna(current_score):
                    excess = target_score - current_score
                    if excess > MOM_SWITCH_CONVICTION:
                        self._sell(hold_symbol, idx, today_data, reason=f"动量切换 {hold_symbol}→{target_etf}")
                        self._buy(target_etf, self.cash * 0.98, idx, today_data,
                                 reason=f"买入 {target_etf}（动量,评分={target_score:.4f}）")
                        self._days_since_last_switch = 0
                return

            # 无持仓：开仓
            if target_etf is not None and not pd.isna(target_score) and target_score > 0:
                self._buy(target_etf, self.cash * 0.98, idx, today_data,
                         reason=f"动量开仓 {target_etf}（评分={target_score:.4f}）")
                self._entry_day = idx
                self._days_since_last_switch = 0
            return

        # ── 震荡模式：均值回归逻辑 ──
        if regime == "neutral":
            if has_position:
                # 检查出场条件
                exit_reason = self._check_reversion_exit(hold_symbol, idx)
                if exit_reason:
                    self._sell(hold_symbol, idx, today_data, reason=exit_reason)
                    return

                # 检查是否切换到更超卖的标的
                if target_etf and hold_symbol and target_etf != hold_symbol and self._days_since_last_switch >= REV_MIN_HOLD_DAYS:
                    current_score = scores.get(hold_symbol, 0.0)
                    if not pd.isna(current_score) and not pd.isna(target_score):
                        if target_score > current_score + 0.5:  # 显著更超卖
                            self._sell(hold_symbol, idx, today_data, reason=f"均值回归切换 {hold_symbol}→{target_etf}")
                            self._buy(target_etf, self.cash * 0.98, idx, today_data,
                                     reason=f"买入 {target_etf}（回归,评分={target_score:.4f}）")
                            self._entry_day = idx
                            self._days_since_last_switch = 0
                return

            # 无持仓
            if target_etf is not None and not pd.isna(target_score) and target_score > 0:
                self._buy(target_etf, self.cash * 0.98, idx, today_data,
                         reason=f"均值回归开仓 {target_etf}（评分={target_score:.4f}）")
                self._entry_day = idx
                self._days_since_last_switch = 0
            return

    def _check_reversion_exit(self, hold_symbol, idx) -> Optional[str]:
        """均值回归出场条件检查。"""
        date_idx = max(0, idx - 1)  # 用T-1数据
        close_arr = self.etf_data[hold_symbol]["close"].values[:date_idx + 1]
        if len(close_arr) < 20:
            return None

        pct_b_arr = _compute_pct_b(close_arr, 20, 2.0)
        rsi_arr = _compute_rsi(close_arr, 14)
        hold_pct_b = pct_b_arr[-1] if len(pct_b_arr) > 0 else np.nan
        hold_rsi = rsi_arr[-1] if len(rsi_arr) > 0 else np.nan

        # %B回升至获利线
        if not np.isnan(hold_pct_b) and hold_pct_b >= REV_REVERT_PCT_B:
            return f"均值回归目标达成：%B={hold_pct_b:.2f}→获利了结"

        # RSI回升至反转线
        if not np.isnan(hold_rsi) and hold_rsi >= REV_REVERT_RSI:
            return f"趋势反转确认：RSI={hold_rsi:.1f}→卖出"

        # %B继续跌破止损线
        if not np.isnan(hold_pct_b) and hold_pct_b <= REV_STOP_LOSS_PCT_B:
            return f"趋势性下跌止损：%B={hold_pct_b:.2f}→止损"

        # 时间止损
        days_held = idx - self._entry_day
        if days_held >= REV_MAX_HOLD_DAYS:
            return f"时间止损：持仓{days_held}天超过{REV_MAX_HOLD_DAYS}天→平仓"

        return None

    def _record_day(self, idx, today_data, action_override="", regime="neutral", top_etf="", score=0.0):
        total_value = self._calc_total_value(today_data)
        stock_value = self._calc_stock_value(today_data)
        hold_symbol = self._get_hold_symbol() or ""
        hold_close = 0.0
        if hold_symbol and hold_symbol in today_data:
            hold_close = today_data[hold_symbol]["close"]
        prev = self.daily_records[-1].total_value if self.daily_records else self.initial_capital
        daily_return = (total_value - prev) / prev if prev > 0 else 0.0
        cum_ret = total_value / self.initial_capital - 1
        action = action_override or ("hold" if hold_symbol else "hold_cash")
        self.daily_records.append(DailyRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"), hold_symbol=hold_symbol,
            hold_shares=self.positions.get(hold_symbol, 0) if hold_symbol else 0,
            hold_close=hold_close, cash=self.cash, stock_value=stock_value,
            total_value=total_value, daily_return=daily_return,
            cumulative_return=cum_ret, action=action, regime=regime,
            top_etf=top_etf, score=score, entry_day=idx - self._entry_day,
        ))

    def _close_remaining_positions(self):
        for sym in list(self.positions.keys()):
            if self.positions[sym] > 0:
                last_close = self.etf_data[sym].iloc[-1]["close"]
                price = last_close * (1 - SLIPPAGE)
                shares = self.positions[sym]
                amount = shares * price
                commission = max(amount * COMMISSION_RATE, 0.0)
                self.cash += amount - commission
                self.trade_records.append(TradeRecord(date=self.dates[-1].strftime("%Y-%m-%d"),
                    symbol=sym, trade_type="虚拟卖出", price=price, shares=shares,
                    amount=amount, commission=commission, tax=0.0, profit=0.0, reason="期末虚拟平仓"))
                self.positions[sym] = 0
                del self.positions[sym]

    def get_daily_df(self) -> pd.DataFrame:
        if not self.daily_records:
            return pd.DataFrame()
        records = [{"date": r.date, "hold_symbol": r.hold_symbol, "hold_shares": r.hold_shares,
                     "hold_close": r.hold_close, "cash": r.cash, "stock_value": r.stock_value,
                     "total_value": r.total_value, "daily_return": r.daily_return,
                     "cumulative_return": r.cumulative_return, "action": r.action,
                     "regime": r.regime, "top_etf": r.top_etf, "score": r.score}
                   for r in self.daily_records]
        df = pd.DataFrame(records)
        df["benchmark_return"] = 0.0
        df["excess_return"] = 0.0
        return df

    def get_trade_df(self) -> pd.DataFrame:
        if not self.trade_records:
            return pd.DataFrame()
        records = [{"date": t.date, "symbol": t.symbol, "trade_type": t.trade_type,
                     "price": t.price, "shares": t.shares, "amount": t.amount,
                     "commission": t.commission, "tax": t.tax, "profit": t.profit, "reason": t.reason}
                   for t in self.trade_records]
        return pd.DataFrame(records)
