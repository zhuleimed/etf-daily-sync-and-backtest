"""
双均线交叉轮动策略 — 回测引擎

每只ETF独立判断MA交叉：
  - 快线上穿慢线（上升趋势）→ 买入/持有
  - 快线下穿慢线（下降趋势）→ 卖出
  所有上升趋势ETF等权持有，全部下降时全仓现金。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import (
    ETF_SYMBOLS,
    ETF_POOL,
    INITIAL_CAPITAL,
    COMMISSION_RATE,
    SLIPPAGE,
    TAX_RATE,
    ADJUSTMENT_DAYS,
    DB_PATH,
    RISK_MODE,
    FAST_MA_PERIOD,
    SLOW_MA_PERIOD,
    MAX_HOLD_ETFS,
)
from .data import (
    load_all_etf_data,
    load_benchmark_data,
    compute_equal_weight_benchmark,
)


# ── 数据类 ──


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
    reason: str = ""


@dataclass
class BuyLot:
    """单笔买入记录（用于 FIFO 成本核算）"""
    date: str = ""
    symbol: str = ""
    shares: int = 0
    price: float = 0.0
    total_cost: float = 0.0     # 总成本（含佣金）


@dataclass
class TradeRecord:
    """单笔交易记录"""
    date: str = ""
    symbol: str = ""
    trade_type: str = ""
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0
    commission: float = 0.0
    tax: float = 0.0
    profit: float = 0.0          # 卖出时计算盈亏
    days_held: int = 0           # 持仓天数
    return_rate: float = 0.0     # 收益率
    reason: str = ""


# ── 回测引擎 ──


class BacktestEngine:
    """
    宽基ETF动量轮动回测引擎。
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL,
                 risk_mode: str = ""):
        self.initial_capital = initial_capital
        self.risk_mode = risk_mode or RISK_MODE

        # 数据
        self.etf_data: Dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.benchmark_data: pd.DataFrame = pd.DataFrame()
        self.equal_weight_data: pd.DataFrame = pd.DataFrame()

        # 持仓状态
        self.positions: Dict[str, int] = {}   # {symbol: shares}
        self.cash: float = initial_capital
        self.open_buys: List[BuyLot] = []     # 未平仓买入记录（FIFO）

        # 渐进调仓调度
        self.adjustment_from: str = ""
        self.adjustment_to: str = ""
        self.adjustment_days_left: int = 0
        self.adjustment_total_days: int = 0

        # 风控

        # 结果
        self.daily_records: List[DailyRecord] = []
        self.trade_records: List[TradeRecord] = []
        self.total_trade_cost: float = 0.0

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_data(self, start_date: str = "2024-01-01",
                  end_date: str = "", db_path: str = DB_PATH
                  ) -> "BacktestEngine":
        """加载ETF数据和基准数据。"""
        self.etf_data, self.dates = load_all_etf_data(
            symbols=ETF_SYMBOLS, start_date=start_date,
            end_date=end_date, db_path=db_path,
            momentum_window=20,
        )
        try:
            self.benchmark_data = load_benchmark_data(
                start_date=start_date, end_date=end_date, db_path=db_path,
                momentum_window=20,
            )
        except ValueError as e:
            print(f"  ⚠ 基准指数加载失败: {e}")
            self.benchmark_data = pd.DataFrame()


        self.equal_weight_data = compute_equal_weight_benchmark(self.etf_data)
        return self

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> "BacktestEngine":
        """执行完整回测。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据，请先调用 load_data()")

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(ETF_POOL.get(s, s) for s in ETF_SYMBOLS)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  快线：{FAST_MA_PERIOD}日  慢线：{SLOW_MA_PERIOD}日")
        print(f"  最多持有：{MAX_HOLD_ETFS} 只")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── Step 2: 渐进调仓 ──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── Step 3: 均线交叉判断（用 idx-1 避免 look-ahead）──
            signal_idx = max(1, idx - 1)
            uptrend_etfs = []
            for sym in ETF_SYMBOLS:
                df = self.etf_data.get(sym)
                if df is None or signal_idx < SLOW_MA_PERIOD:
                    continue
                fast_ma = df.iloc[signal_idx - FAST_MA_PERIOD + 1: signal_idx + 1]["close"].mean()
                slow_ma = df.iloc[signal_idx - SLOW_MA_PERIOD + 1: signal_idx + 1]["close"].mean()
                if fast_ma > slow_ma:
                    uptrend_etfs.append(sym)

            # ── Step 4: 等权分配 ──
            if self.adjustment_days_left <= 0:
                self._rebalance_by_ma(uptrend_etfs, idx, today_data)

            # ── 记录 ──
            self._record_day(idx, today_data)

        # ── 期末：未平仓虚拟卖出 ──
        self._close_remaining_positions()

        print(f"  回测完成 ✓")
        return self

    # ------------------------------------------------------------------
    # 持仓与市值
    # ------------------------------------------------------------------

    def _get_hold_symbol(self) -> Optional[str]:
        if not self.positions:
            return None
        valid = {k: v for k, v in self.positions.items() if v > 0}
        if not valid:
            return None
        return max(valid, key=valid.get)

    def _calc_total_value(self, today_data: Dict) -> float:
        sv = 0.0
        for sym, sh in self.positions.items():
            if sh > 0 and sym in today_data:
                sv += sh * today_data[sym]["close"]
        return self.cash + sv

    def _calc_stock_value(self, today_data: Dict) -> float:
        sv = 0.0
        for sym, sh in self.positions.items():
            if sh > 0 and sym in today_data:
                sv += sh * today_data[sym]["close"]
        return sv

    # ------------------------------------------------------------------
    # 交易执行（带 FIFO 成本核算）
    # ------------------------------------------------------------------

    def _buy(self, symbol: str, amount: float, idx: int, today_data: Dict,
             trade_type: str = "买入", reason: str = "") -> int:
        """买入 ETF，记录 BuyLot 用于后续 FIFO 成本核算。"""
        price = today_data[symbol]["close"] * (1 + SLIPPAGE)
        max_shares = int(amount // price // 100) * 100
        if max_shares <= 0:
            return 0

        # 如果现金不够，用实际现金重新算
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

        # 记录买入批次（FIFO）
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
        """
        卖出 ETF，使用 FIFO 匹配买入批次计算盈亏和持仓天数。
        返回卖出净收入。
        """
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

        # FIFO 匹配买入批次，计算盈亏
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
            # 成本比例分摊
            batch_cost = lot.total_cost * (batch / lot.shares)
            total_buy_cost += batch_cost
            buy_date = pd.Timestamp(lot.date)
            days_held = (sell_date - buy_date).days
            max_days = max(max_days, days_held)
            total_days_held += days_held * batch

            # 更新买入批次剩余份额
            lot.shares -= batch
            if lot.shares > 0:
                lot.total_cost -= batch_cost
                new_open_buys.append(lot)
            remaining -= batch

        self.open_buys = new_open_buys

        # 计算盈亏
        sell_commission_part = commission * (actual / max(actual, 1))
        sell_tax_part = tax * (actual / max(actual, 1))
        profit = revenue - total_buy_cost - sell_commission_part - sell_tax_part
        avg_days = total_days_held // max(actual, 1)
        return_rate = profit / total_buy_cost if total_buy_cost > 0 else 0.0

        self.trade_records.append(TradeRecord(
            date=str(self.dates[idx].date()), symbol=symbol,
            trade_type=trade_type, price=round(sell_price, 4),
            shares=actual, amount=round(revenue, 2),
            commission=round(commission, 2), tax=round(tax, 2),
            profit=round(profit, 2), days_held=avg_days,
            return_rate=round(return_rate, 4),
            reason=reason,
        ))

        return net_revenue

    def _sell_all(self, idx: int, today_data: Dict,
                  trade_type: str = "卖出", reason: str = "",
                  symbol: Optional[str] = None):
        """
        清空持仓。指定 symbol 则只卖该品种，否则清空全部。
        """
        syms = [symbol] if symbol else list(self.positions.keys())
        for sym in syms:
            sh = self.positions.get(sym, 0)
            if sh > 0:
                self._sell(sym, sh, idx, today_data, trade_type, reason)

    # ------------------------------------------------------------------
    # 渐进调仓
    # ------------------------------------------------------------------

    def _start_adjustment(self, from_symbol: str, to_symbol: str,
                          idx: int, today_data: Dict):
        """启动渐进调仓。"""
        from_shares = self.positions.get(from_symbol, 0)
        if from_shares <= 0:
            self._buy(to_symbol, self.cash, idx, today_data,
                      trade_type="买入", reason="动量信号开仓")
            return

        self.adjustment_from = from_symbol
        self.adjustment_to = to_symbol
        self.adjustment_days_left = ADJUSTMENT_DAYS
        self.adjustment_total_days = ADJUSTMENT_DAYS
        self._days_since_last_switch = 0  # 重置持仓计时

        self._execute_adjustment_step(idx, today_data)

    def _execute_adjustment_step(self, idx: int, today_data: Dict):
        """执行一步渐进调仓。"""
        if self.adjustment_days_left <= 0:
            self._finish_adjustment(idx, today_data)
            return

        from_shares = self.positions.get(self.adjustment_from, 0)
        if from_shares > 0 and self.adjustment_days_left > 0:
            sell_shares = from_shares // self.adjustment_days_left
            if sell_shares > 0:
                day_num = self.adjustment_total_days - self.adjustment_days_left + 1
                net = self._sell(self.adjustment_from, sell_shares, idx, today_data,
                                 trade_type="调仓卖出",
                                 reason=f"渐进调仓第{day_num}天")
                if net > 0 and self.cash > 0:
                    self._buy(self.adjustment_to, self.cash, idx, today_data,
                              trade_type="调仓买入",
                              reason=f"渐进调仓第{day_num}天")

        self.adjustment_days_left -= 1
        if self.adjustment_days_left <= 0:
            self._finish_adjustment(idx, today_data)

    def _finish_adjustment(self, idx: int = 0, today_data: Dict = None):
        """完成调仓：清理调度状态。"""
        self.adjustment_from = ""
        self.adjustment_to = ""
        self.adjustment_days_left = 0
        self.adjustment_total_days = 0

    # ------------------------------------------------------------------
    # 均线交叉重平衡
    # ------------------------------------------------------------------

    def _rebalance_by_ma(self, uptrend_etfs, idx, today_data):
        """根据均线趋势重平衡持仓。"""
        if MAX_HOLD_ETFS > 0 and len(uptrend_etfs) > MAX_HOLD_ETFS:
            ranked = sorted(uptrend_etfs,
                key=lambda s: today_data[s]["close"] / self.etf_data[s].iloc[max(0, idx-20)]["close"] - 1,
                reverse=True)
            uptrend_etfs = ranked[:MAX_HOLD_ETFS]
        target_set = set(uptrend_etfs)
        current_set = set(s for s, sh in self.positions.items() if sh > 0)
        to_sell = current_set - target_set
        for sym in to_sell:
            self._sell_all(idx, today_data, reason=f"{sym}跌破慢线卖出", symbol=sym)
        to_buy = target_set - current_set
        if to_buy and self.cash > 0:
            cash_per = self.cash / len(to_buy)
            for sym in to_buy:
                self._buy(sym, cash_per, idx, today_data,
                          trade_type="买入", reason="均线交叉买入")

    # ------------------------------------------------------------------
    # 期末处理
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def _close_remaining_positions(self):
        """回测结束，未平仓仓位以收盘价虚拟卖出。"""
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

            # FIFO 匹配计算收益
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
            days_held = (pd.Timestamp(last_date.date()) - pd.Timestamp(buy_date_str)).days if buy_date_str else 0

            self.trade_records.append(TradeRecord(
                date=str(last_date.date()), symbol=sym,
                trade_type="虚拟卖出", price=round(cp, 4),
                shares=shares, amount=round(revenue, 2),
                commission=0.0, tax=0.0,
                profit=round(profit, 2), days_held=days_held,
                return_rate=round(return_rate, 4),
                reason="回测结束虚拟卖出",
            ))

    # ------------------------------------------------------------------
    # 每日记录
    # ------------------------------------------------------------------

    def _record_day(self, idx: int, today_data: Dict,
                    action_override: str = ""):
        """记录当日账户状态。

        Parameters
        ----------
        action_override : str
            风控触发时传入风控动作名（stop_loss/stop_profit/extreme_drawdown），
            覆盖 _get_action_desc() 的自动判断。
        """
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
        action = action_override or self._get_action_desc()

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
            action=action,

        ))

    def _get_action_desc(self) -> str:
        if self.adjustment_days_left > 0:
            done = self.adjustment_total_days - self.adjustment_days_left
            return f"调仓中({done}/{self.adjustment_total_days})"
        if not self.positions:
            return "空仓"
        return "持有"

    def get_daily_df(self) -> pd.DataFrame:
        rows = []
        for r in self.daily_records:
            rows.append({
                "date": r.date, "hold_symbol": r.hold_symbol,
                "hold_shares": r.hold_shares, "hold_close": r.hold_close,
                "cash": r.cash, "stock_value": r.stock_value,
                "total_value": r.total_value, "daily_return": r.daily_return,
                "cumulative_return": r.cumulative_return, "action": r.action,
            })
        return pd.DataFrame(rows)

    def get_trade_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trade_records:
            rows.append({
                "date": t.date, "symbol": ETF_POOL.get(t.symbol, t.symbol),
                "trade_type": t.trade_type, "price": t.price,
                "shares": t.shares, "amount": t.amount,
                "commission": t.commission, "tax": t.tax,
                "profit": t.profit, "days_held": t.days_held,
                "return_rate": t.return_rate, "reason": t.reason,
            })
        return pd.DataFrame(rows)


