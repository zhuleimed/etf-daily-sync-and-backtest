"""
回测引擎核心模块

在 momentum_rotation 引擎基础上增加市场状态识别：
- BULL 状态：锁定持有，不执行切换
- BEAR 状态：正常动量轮动
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    ETF_SYMBOLS,
    ETF_POOL,
    INITIAL_CAPITAL,
    MOMENTUM_WINDOW,
    COMMISSION_RATE,
    SLIPPAGE,
    TAX_RATE,
    ADJUSTMENT_DAYS,
    DB_PATH,
    MIN_SWITCH_CONVICTION,
    BULL_MA_PERIOD,
    BULL_MOMENTUM_WINDOW,
    BULL_MOMENTUM_THRESHOLD,
    REGIME_CONFIRM_DAYS,
    BULL_ENTRY_BUFFER,
    BULL_EXIT_BUFFER,
)
from .data import (
    load_all_etf_data,
    load_benchmark_data,
    compute_equal_weight_benchmark,
)
from .signal import (
    compute_momentum_signals,
    rank_etfs_by_momentum,
)
from .cost import compute_total_friction_cost
from .regime import compute_regime, RegimeDetector


@dataclass
class DailyRecord:
    """单日账户状态快照"""
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
    target_etf: str = ""
    momentum_str: str = ""
    regime: str = ""           # 新增：当日市场状态


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
    """市场状态感知动量轮动回测引擎"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.etf_data: Dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        self.equal_weight_data: pd.DataFrame = pd.DataFrame()
        self.positions: Dict[str, int] = {}
        self.cash: float = initial_capital
        self.open_buys: List[BuyLot] = []
        self.adjustment_from: str = ""
        self.adjustment_to: str = ""
        self.adjustment_days_left: int = 0
        self.adjustment_total_days: int = 0
        self.daily_records: List[DailyRecord] = []
        self.trade_records: List[TradeRecord] = []
        self.total_trade_cost: float = 0.0
        self._last_reason: str = ""

        # 市场状态检测器（带迟滞和确认窗口）
        self.regime_detector = RegimeDetector(
            ma_period=BULL_MA_PERIOD,
            mom_window=BULL_MOMENTUM_WINDOW,
            mom_threshold=BULL_MOMENTUM_THRESHOLD,
            confirm_days=REGIME_CONFIRM_DAYS,
            bull_entry_buffer=BULL_ENTRY_BUFFER,
            bull_exit_buffer=BULL_EXIT_BUFFER,
        )

    def load_data(self, start_date: str = "2024-01-01",
                  end_date: str = "", db_path: str = DB_PATH
                  ) -> "BacktestEngine":
        """加载ETF数据和基准数据。"""
        self.etf_data, self.dates = load_all_etf_data(
            symbols=ETF_SYMBOLS, start_date=start_date,
            end_date=end_date, db_path=db_path,
            momentum_window=MOMENTUM_WINDOW,
        )
        try:
            # 为计算长期均线，额外前载数据
            import datetime
            start_dt = pd.to_datetime(start_date)
            extended_start = (start_dt - datetime.timedelta(days=BULL_MA_PERIOD * 3)
                             ).strftime("%Y-%m-%d")
            self.benchmark_data = load_benchmark_data(
                start_date=extended_start, end_date=end_date, db_path=db_path,
                momentum_window=MOMENTUM_WINDOW,
            )
        except ValueError as e:
            print(f"  ⚠ 基准指数加载失败: {e}")
            self.benchmark_data = pd.DataFrame()

        self.equal_weight_data = compute_equal_weight_benchmark(self.etf_data)
        return self

    def run(self) -> "BacktestEngine":
        """执行完整回测。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(ETF_POOL.get(s, s) for s in ETF_SYMBOLS)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  BULL判定：{BULL_MA_PERIOD}日均线+{BULL_MOMENTUM_WINDOW}日动量>{BULL_MOMENTUM_THRESHOLD:.0%}")
        print(f"  BULL时锁仓(BEAR时正常轮动)")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── 检测市场状态（核心创新）──
            regime = self._detect_market_regime(idx)

            # ── 动量信号 ──
            momentum = compute_momentum_signals(self.etf_data, idx, MOMENTUM_WINDOW)
            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = self._format_ranking(ranking, momentum)

            # ── 决策（受市场状态影响）──
            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target_etf,
                                    momentum, regime)

            # ── 记录 ──
            self._record_day(idx, today_data, target_etf=target_etf or "",
                             momentum_str=momentum_str, regime=regime)

            # 持仓天数递增
            if hasattr(self, '_days_since_last_switch'):
                self._days_since_last_switch += 1
            else:
                self._days_since_last_switch = 0

        self._close_remaining_positions()
        print(f"  回测完成 ✓")
        return self

    def _detect_market_regime(self, idx: int) -> str:
        """检测市场状态（使用 RegimeDetector — 带迟滞+确认窗口）。"""
        if self.benchmark_data.empty:
            return "BEAR"
        target_date = self.dates[idx]
        try:
            bm_idx = self.benchmark_data[
                self.benchmark_data["date"] == target_date
            ].index[0]
        except (IndexError, KeyError):
            return "BEAR"
        return self.regime_detector.update(
            self.benchmark_data, bm_idx,
        )

    def _make_decision(self, idx: int, today_data: Dict,
                       hold_symbol: Optional[str],
                       target_etf: Optional[str],
                       momentum_series: pd.Series,
                       regime: str):
        """核心决策：受市场状态影响。"""
        if target_etf is None:
            return
        target_mom = momentum_series.get(target_etf, np.nan)
        if np.isnan(target_mom):
            return

        has_position = hold_symbol is not None

        # ── 无持仓：开仓（不受市场状态影响）──
        if not has_position:
            if target_mom > 0:
                self._buy(target_etf, self.cash, idx, today_data,
                          trade_type="买入", reason="动量信号开仓")
            return

        # ── BULL 状态：锁定持有，不做切换 ──
        if regime == "BULL":
            return

        # ── BEAR 状态：正常动量轮动 ──
        if hold_symbol == target_etf:
            return
        current_mom = momentum_series.get(hold_symbol, np.nan)
        if np.isnan(current_mom):
            return

        # 动量减速检查
        if idx >= 5:
            tgt_5d = (today_data[target_etf]["close"] /
                      self.etf_data[target_etf].iloc[idx - 5]["close"] - 1)
            if tgt_5d <= -0.005:
                return
            tgt_15d = momentum_series.get(target_etf, np.nan)
            if not pd.isna(tgt_15d) and tgt_15d > 0:
                if tgt_5d / 5 < tgt_15d / 15:
                    return

        # 摩擦成本校验
        hold_shares = self.positions.get(hold_symbol, 0)
        sell_amt = hold_shares * today_data[hold_symbol]["close"]
        buy_amt = sell_amt
        friction, _ = compute_total_friction_cost(
            hold_symbol, target_etf, sell_amt, buy_amt,
            today_data[hold_symbol]["close"],
            today_data[target_etf]["close"],
            self.etf_data, idx,
        )
        excess = target_mom - current_mom
        total_trade_amt = sell_amt + buy_amt
        friction_ratio = friction / total_trade_amt if total_trade_amt > 0 else 1.0
        switch_threshold = max(friction_ratio, MIN_SWITCH_CONVICTION)
        if excess > switch_threshold:
            self._start_adjustment(hold_symbol, target_etf, idx, today_data)

    # ═══════════════════════════════════════════════
    # 以下方法与 momentum_rotation 一致
    # ═══════════════════════════════════════════════

    def _get_hold_symbol(self) -> Optional[str]:
        if not self.positions:
            return None
        valid = {k: v for k, v in self.positions.items() if v > 0}
        return max(valid, key=valid.get) if valid else None

    def _calc_total_value(self, today_data: Dict) -> float:
        sv = sum(sh * today_data[sym]["close"]
                 for sym, sh in self.positions.items() if sh > 0 and sym in today_data)
        return self.cash + sv

    def _calc_stock_value(self, today_data: Dict) -> float:
        return sum(sh * today_data[sym]["close"]
                   for sym, sh in self.positions.items() if sh > 0 and sym in today_data)

    def _buy(self, symbol: str, amount: float, idx: int, today_data: Dict,
             trade_type: str = "买入", reason: str = "") -> int:
        price = today_data[symbol]["close"] * (1 + SLIPPAGE)
        max_shares = int(amount // price // 100) * 100
        if max_shares <= 0:
            return 0
        cost = max_shares * price
        commission = max(cost * COMMISSION_RATE, 0.0)
        total_cost = cost + commission
        if total_cost > self.cash:
            max_shares = int(self.cash // price // 100) * 100
            if max_shares <= 0:
                return 0
            cost = max_shares * price
            commission = max(cost * COMMISSION_RATE, 0.0)
            total_cost = cost + commission
        self.positions[symbol] = self.positions.get(symbol, 0) + max_shares
        self.cash -= total_cost
        self.total_trade_cost += commission
        self.open_buys.append(BuyLot(
            date=str(self.dates[idx].date()),
            symbol=symbol, shares=max_shares,
            price=price, total_cost=total_cost,
        ))
        self.trade_records.append(TradeRecord(
            date=str(self.dates[idx].date()), symbol=symbol,
            trade_type=trade_type, price=round(price, 4),
            shares=max_shares, amount=round(cost, 2),
            commission=round(commission, 2), tax=0.0,
            reason=reason,
        ))
        return max_shares

    def _sell(self, symbol: str, shares: int, idx: int, today_data: Dict,
              trade_type: str = "卖出", reason: str = "") -> float:
        if symbol not in self.positions or shares <= 0:
            return 0.0
        actual = min(shares, self.positions[symbol])
        if actual <= 0:
            return 0.0
        sell_price = today_data[symbol]["close"] * (1 - SLIPPAGE)
        revenue = actual * sell_price
        commission = max(revenue * COMMISSION_RATE, 0.0)
        tax = revenue * TAX_RATE
        net_revenue = revenue - commission - tax
        self.positions[symbol] -= actual
        if self.positions[symbol] <= 0:
            del self.positions[symbol]
        self.cash += net_revenue
        self.total_trade_cost += commission
        remaining = actual
        total_buy_cost = 0.0
        total_days_held = 0
        max_days = 0
        sell_date = pd.Timestamp(self.dates[idx].date())
        new_open_buys = []
        for lot in self.open_buys:
            if lot.symbol != symbol:
                new_open_buys.append(lot)
                continue
            if remaining <= 0:
                new_open_buys.append(lot)
                continue
            batch = min(lot.shares, remaining)
            batch_cost = lot.total_cost * (batch / lot.shares)
            total_buy_cost += batch_cost
            buy_date = pd.Timestamp(lot.date)
            days_held = (sell_date - buy_date).days
            max_days = max(max_days, days_held)
            total_days_held += days_held * batch
            lot.shares -= batch
            if lot.shares > 0:
                lot.total_cost -= batch_cost
                new_open_buys.append(lot)
            remaining -= batch
        self.open_buys = new_open_buys
        profit = revenue - total_buy_cost - commission - tax
        avg_days = total_days_held // max(actual, 1)
        return_rate = profit / total_buy_cost if total_buy_cost > 0 else 0.0
        self.trade_records.append(TradeRecord(
            date=str(self.dates[idx].date()), symbol=symbol,
            trade_type=trade_type, price=round(sell_price, 4),
            shares=actual, amount=round(revenue, 2),
            commission=round(commission, 2), tax=round(tax, 2),
            profit=round(profit, 2), days_held=avg_days,
            return_rate=round(return_rate, 4), reason=reason,
        ))
        return net_revenue

    def _sell_all(self, idx: int, today_data: Dict,
                  trade_type: str = "卖出", reason: str = ""):
        for sym in list(self.positions.keys()):
            sh = self.positions.get(sym, 0)
            if sh > 0:
                self._sell(sym, sh, idx, today_data, trade_type, reason)

    def _start_adjustment(self, from_symbol: str, to_symbol: str,
                          idx: int, today_data: Dict):
        from_shares = self.positions.get(from_symbol, 0)
        if from_shares <= 0:
            self._buy(to_symbol, self.cash, idx, today_data,
                      trade_type="买入", reason="动量信号开仓")
            return
        self.adjustment_from = from_symbol
        self.adjustment_to = to_symbol
        self.adjustment_days_left = ADJUSTMENT_DAYS
        self.adjustment_total_days = ADJUSTMENT_DAYS
        self._execute_adjustment_step(idx, today_data)

    def _execute_adjustment_step(self, idx: int, today_data: Dict):
        if self.adjustment_days_left <= 0:
            self._finish_adjustment(idx, today_data)
            return
        from_shares = self.positions.get(self.adjustment_from, 0)
        if from_shares > 0 and self.adjustment_days_left > 0:
            sell_shares = from_shares // self.adjustment_days_left
            if sell_shares > 0:
                day_num = self.adjustment_total_days - self.adjustment_days_left + 1
                net = self._sell(self.adjustment_from, sell_shares, idx, today_data,
                                 trade_type="调仓卖出", reason=f"渐进调仓第{day_num}天")
                if net > 0 and self.cash > 0:
                    self._buy(self.adjustment_to, self.cash, idx, today_data,
                              trade_type="调仓买入", reason=f"渐进调仓第{day_num}天")
        self.adjustment_days_left -= 1
        if self.adjustment_days_left <= 0:
            self._finish_adjustment(idx, today_data)

    def _finish_adjustment(self, idx: int = 0, today_data: Dict = None):
        self.adjustment_from = ""
        self.adjustment_to = ""
        self.adjustment_days_left = 0
        self.adjustment_total_days = 0

    def _close_remaining_positions(self):
        if not self.positions:
            return
        last_idx = len(self.dates) - 1
        last_date = self.dates[last_idx]
        td = {sym: self.etf_data[sym].iloc[last_idx] for sym in ETF_SYMBOLS}
        for sym in list(self.positions.keys()):
            shares = self.positions[sym]
            if shares <= 0:
                continue
            cp = td[sym]["close"]
            revenue = shares * cp
            self.cash += revenue
            del self.positions[sym]
            remaining = shares
            total_buy = 0.0
            buy_date_str = ""
            new_opens = []
            for lot in self.open_buys:
                if lot.symbol != sym:
                    new_opens.append(lot)
                    continue
                if remaining <= 0:
                    new_opens.append(lot)
                    continue
                batch = min(lot.shares, remaining)
                batch_cost = lot.total_cost * (batch / lot.shares)
                total_buy += batch_cost
                buy_date_str = lot.date
                lot.shares -= batch
                if lot.shares > 0:
                    lot.total_cost -= batch_cost
                    new_opens.append(lot)
                remaining -= batch
            self.open_buys = new_opens
            profit = revenue - total_buy
            return_rate = profit / total_buy if total_buy > 0 else 0.0
            days_held = (pd.Timestamp(last_date.date()) -
                        pd.Timestamp(buy_date_str)).days if buy_date_str else 0
            self.trade_records.append(TradeRecord(
                date=str(last_date.date()), symbol=sym,
                trade_type="虚拟卖出", price=round(cp, 4),
                shares=shares, amount=round(revenue, 2),
                commission=0.0, tax=0.0,
                profit=round(profit, 2), days_held=days_held,
                return_rate=round(return_rate, 4),
                reason="回测结束虚拟卖出",
            ))

    def _record_day(self, idx: int, today_data: Dict,
                    target_etf: str = "", momentum_str: str = "",
                    regime: str = ""):
        hold_sym = self._get_hold_symbol()
        hold_shares = self.positions.get(hold_sym, 0) if hold_sym else 0
        hold_close = today_data[hold_sym]["close"] if hold_sym else 0.0
        stock_value = self._calc_stock_value(today_data)
        total_value = self.cash + stock_value
        if not self.daily_records:
            daily_ret = 0.0
        else:
            prev = self.daily_records[-1].total_value
            daily_ret = (total_value - prev) / prev if prev > 0 else 0.0
        cum_ret = total_value / self.initial_capital - 1
        self.daily_records.append(DailyRecord(
            date=str(self.dates[idx].date()),
            hold_symbol=hold_sym or "",
            hold_shares=hold_shares,
            hold_close=round(hold_close, 4),
            cash=round(self.cash, 2),
            stock_value=round(stock_value, 2),
            total_value=round(total_value, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            action="持有" if hasattr(self, '_days_since_last_switch') else "空仓",
            target_etf=target_etf,
            momentum_str=momentum_str,
            regime=regime,
        ))

    def _format_ranking(self, ranking: pd.Series, momentum: pd.Series) -> str:
        parts = []
        for rk in range(1, min(len(ranking) + 1, 6)):
            sym = ranking.get(rk)
            if sym:
                mom = momentum.get(sym, np.nan)
                if not np.isnan(mom):
                    parts.append(f"#{rk}{ETF_POOL.get(sym, sym)}({mom:.4f})")
        return " > ".join(parts)

    def get_daily_df(self) -> pd.DataFrame:
        rows = []
        for r in self.daily_records:
            rows.append({
                "date": r.date, "hold_symbol": r.hold_symbol,
                "hold_shares": r.hold_shares, "hold_close": r.hold_close,
                "cash": r.cash, "stock_value": r.stock_value,
                "total_value": r.total_value, "daily_return": r.daily_return,
                "cumulative_return": r.cumulative_return, "action": r.action,
                "target_etf": r.target_etf, "momentum_rank": r.momentum_str,
                "regime": r.regime,
            })
        return pd.DataFrame(rows)

    def get_trade_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trade_records:
            rows.append({
                "date": t.date,
                "symbol": ETF_POOL.get(t.symbol, t.symbol),
                "trade_type": t.trade_type, "price": t.price,
                "shares": t.shares, "amount": t.amount,
                "commission": t.commission, "tax": t.tax,
                "profit": t.profit, "days_held": t.days_held,
                "return_rate": t.return_rate, "reason": t.reason,
            })
        return pd.DataFrame(rows)
