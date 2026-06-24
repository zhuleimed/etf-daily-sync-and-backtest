"""
动量+均线过滤轮动策略 — 回测引擎核心模块

在 momentum_rotation 引擎基础上增加均线过滤：
  - 沪深300在60日均线上方 → 正常动量轮动
  - 沪深300在60日均线下方 → 全部空仓（规避下跌风险）
"""

from dataclasses import dataclass, field
from datetime import datetime
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
    RISK_MODE,
    MIN_SWITCH_CONVICTION,
    TOP_N,
    DYNAMIC_WINDOW_ENABLED,
    WINDOW_SWITCH_THRESHOLD,
    MIN_HOLD_DAYS,
    USE_RELATIVE_MOMENTUM,
    ETF_BENCHMARK_MAP,
    RELATIVE_MOMENTUM_FACTOR,
    SHORT_TERM_MOMENTUM_CHECK,
    MA_FILTER_ENABLED,
    MA_FILTER_PERIOD,
    MA_FILTER_BENCHMARK,
)
from .data import (
    load_all_etf_data,
    load_benchmark_data,
    compute_equal_weight_benchmark,
)
from .momentum_signals import (
    compute_momentum_signals,
    compute_momentum_signals_dynamic,
    compute_momentum_spread,
    rank_etfs_by_momentum,
)
from .cost import compute_total_friction_cost
from .risk import RiskState, run_all_risk_checks


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
    target_etf: str = ""
    momentum_str: str = ""
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
                 risk_mode: str = "",
                 momentum_window: int = MOMENTUM_WINDOW,
                 top_n: int = TOP_N,
                 dynamic_window: bool = DYNAMIC_WINDOW_ENABLED):
        self.initial_capital = initial_capital
        # 风控模式: "" 用 config 默认, "A"=纯信号, "B"=全开, "C"=仅极端回撤
        self.risk_mode = risk_mode or RISK_MODE
        self.momentum_window = momentum_window
        self.top_n = top_n
        self.dynamic_window = dynamic_window
        # 最小持仓天数追踪
        self._days_since_last_switch = 999  # 初始大值，允许首次开仓
        # 切换冷却期（连续亏损后主动停手）
        self._bad_switch_streak = 0
        self._switch_cooldown = 0

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
        self.risk_state = RiskState()

        # 结果
        self.daily_records: List[DailyRecord] = []
        self.trade_records: List[TradeRecord] = []
        self.total_trade_cost: float = 0.0
        self._last_reason: str = ""

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
            momentum_window=self.momentum_window,
        )
        # 提前基准数据起始日，确保有足够历史计算均线
        try:
            from datetime import timedelta
            start_dt = pd.to_datetime(start_date)
            bench_start = (start_dt - timedelta(days=MA_FILTER_PERIOD * 3)).strftime("%Y-%m-%d")
            self.benchmark_data = load_benchmark_data(
                start_date=bench_start, end_date=end_date, db_path=db_path,
                momentum_window=self.momentum_window,
            )
        except ValueError as e:
            print(f"  ⚠ 基准指数加载失败: {e}")
            self.benchmark_data = pd.DataFrame()

        # 加载各ETF对应的基准指数（用于相对动量）
        self.etf_benchmark_data: Dict[str, pd.DataFrame] = {}
        if USE_RELATIVE_MOMENTUM:
            unique_indices = set(ETF_BENCHMARK_MAP.values())
            for idx_code in unique_indices:
                try:
                    df_idx = load_benchmark_data(
                        symbol=idx_code, start_date=start_date,
                        end_date=end_date, db_path=db_path,
                        momentum_window=self.momentum_window,
                    )
                    self.etf_benchmark_data[idx_code] = df_idx
                except ValueError:
                    print(f"  ⚠ 指数 {idx_code} 加载失败，相对动量可能不完整")

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
        if self.dynamic_window:
            print(f"  动量窗口：动态(10/20日, 阈值{int(WINDOW_SWITCH_THRESHOLD*100)}%)")
        else:
            print(f"  动量窗口：{self.momentum_window} 日")
        print(f"  调仓周期：{ADJUSTMENT_DAYS} 日")
        if self.top_n > 1:
            print(f"  TOP-N：持有前{self.top_n}名等权")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── Step 2: 风控检查（A=纯信号模式跳过）──
            if self.risk_mode != "A" and has_position and hold_symbol:
                hold_row = today_data[hold_symbol]
                total_value = self._calc_total_value(today_data)
                self.risk_state.update_peak(hold_row["high"])
                self.risk_state.update_peak_total_value(total_value)
                risk_action, risk_reason = run_all_risk_checks(
                    self.risk_state, total_value, has_position,
                    hold_symbol, hold_row["high"], hold_row["low"], hold_row["close"], hold_row["atr"],
                    self.etf_data, idx, mode=self.risk_mode,
                )
                if risk_action != "none":
                    self._execute_risk_exit(idx, today_data, risk_action, risk_reason)
                    self._record_day(idx, today_data, action_override=risk_action)
                    continue

            # ── Step 3: 渐进调仓 ──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── Step 4: 动量信号（用 idx-1 避免 look-ahead）──
            # 信号用前一日收盘数据计算，交易用当日收盘执行
            signal_idx = max(0, idx - 1)
            if self.dynamic_window:
                momentum = compute_momentum_signals_dynamic(
                    self.etf_data, signal_idx, threshold=WINDOW_SWITCH_THRESHOLD,
                )
            else:
                momentum = compute_momentum_signals(self.etf_data, signal_idx, self.momentum_window)

            # 相对动量：每只ETF减去各自基准指数的动量
            if USE_RELATIVE_MOMENTUM and self.etf_benchmark_data:
                for sym in momentum.index:
                    if sym in ETF_BENCHMARK_MAP:
                        idx_code = ETF_BENCHMARK_MAP[sym]
                        idx_df = self.etf_benchmark_data.get(idx_code)
                        if idx_df is not None and signal_idx < len(idx_df):
                            idx_mom = idx_df.iloc[signal_idx].get("momentum", np.nan)
                            if not pd.isna(idx_mom) and not pd.isna(momentum.get(sym)):
                                momentum[sym] = momentum[sym] - idx_mom * RELATIVE_MOMENTUM_FACTOR

            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = self._format_ranking(ranking, momentum)

            # ── Step 5: 决策（支持 TOP-N）──
            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)

            # ── Step 6: 记录 ──
            self._record_day(idx, today_data, target_etf=target_etf or "",
                             momentum_str=momentum_str)

            # 持仓天数递增 + 冷却期倒数
            self._days_since_last_switch += 1
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1

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
        price = today_data[symbol]["open"] * (1 + SLIPPAGE)
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
        self._last_reason = reason
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

        sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)
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
            self.risk_state.on_open_position(today_data[to_symbol]["close"])
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
        """完成调仓：清理调度状态，重置风控以新持仓为准。"""
        self.adjustment_from = ""
        self.adjustment_to = ""
        self.adjustment_days_left = 0
        self.adjustment_total_days = 0

        # 重置风控：以调整后的新持仓为基准
        hold_sym = self._get_hold_symbol()
        if hold_sym and today_data is not None and hold_sym in today_data:
            self.risk_state.on_open_position(today_data[hold_sym]["close"])

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------

    def _is_market_above_ma(self, idx: int) -> bool:
        """判断沪深300收盘价是否在均线上方。"""
        if self.benchmark_data.empty:
            return True
        target_date = self.dates[idx]
        try:
            bm_idx_pos = self.benchmark_data[
                self.benchmark_data["date"] == target_date
            ].index[0]
        except (IndexError, KeyError):
            return True  # 无法判断时默认允许交易
        close = self.benchmark_data.iloc[bm_idx_pos]["close"]
        if bm_idx_pos < MA_FILTER_PERIOD - 1:
            return True  # 数据不足时默认允许
        ma = self.benchmark_data.iloc[
            bm_idx_pos - MA_FILTER_PERIOD + 1: bm_idx_pos + 1
        ]["close"].mean()
        return close > ma

    # ------------------------------------------------------------------

    def _make_decision(self, idx: int, today_data: Dict,
                       hold_symbol: Optional[str],
                       target_etf: Optional[str],
                       momentum_series: pd.Series):
        """核心决策：开仓 / 切换 / 持有（支持 TOP-N）。"""
        # ── 均线过滤：沪深300在均线下方 → 全部空仓 ──
        if MA_FILTER_ENABLED and not self._is_market_above_ma(idx):
            if hold_symbol is not None:
                self._sell_all(idx, today_data, reason="均线过滤空仓")
            return

        if target_etf is None:
            return
        target_mom = momentum_series.get(target_etf, np.nan)
        if np.isnan(target_mom):
            return

        if self.top_n > 1:
            self._make_decision_top_n(idx, today_data, momentum_series)
        else:
            self._make_decision_single(idx, today_data, hold_symbol, target_etf, momentum_series)

    # ------------------------------------------------------------------
    # 决策：单只持有（原逻辑，保留兼容）
    # ------------------------------------------------------------------

    def _make_decision_single(self, idx: int, today_data: Dict,
                               hold_symbol: Optional[str],
                               target_etf: Optional[str],
                               momentum_series: pd.Series):
        """单只持有的决策逻辑（TOP_N=1 时使用）。"""
        target_mom = momentum_series.get(target_etf, np.nan)
        has_position = hold_symbol is not None

        if not has_position:
            if target_mom > 0:
                self._buy(target_etf, self.cash, idx, today_data,
                          trade_type="买入", reason="动量信号开仓")
                self.risk_state.on_open_position(today_data[target_etf]["close"])
            return

        if hold_symbol == target_etf:
            return
        current_mom = momentum_series.get(hold_symbol, np.nan)
        if np.isnan(current_mom):
            return

        # 最小持仓天数过滤：刚切换不久，不再次切换
        if MIN_HOLD_DAYS > 0 and self._days_since_last_switch < MIN_HOLD_DAYS:
            return

        # 短期动量确认：目标ETF的5日动量不能为负（用 idx-1 保持与动量信号一致）
        if SHORT_TERM_MOMENTUM_CHECK and idx >= 6:
            check_idx = idx - 1  # 与前一日收盘数据比较，避免 look-ahead
            tgt_5d = (self.etf_data[target_etf].iloc[check_idx]["close"] /
                      self.etf_data[target_etf].iloc[check_idx - 5]["close"] - 1)
            if tgt_5d <= -0.005:
                return
            # 动量减速检查：目标短期涨幅 < 中期日均涨幅 → 动能衰减 → 不买
            tgt_15d = momentum_series.get(target_etf, np.nan)
            if not pd.isna(tgt_15d) and tgt_15d > 0:
                if tgt_5d / 5 < tgt_15d / 15:
                    return
        # 摩擦成本校验
        hold_shares = self.positions.get(hold_symbol, 0)
        sell_amt = hold_shares * today_data[hold_symbol]["close"]
        buy_amt = sell_amt
        friction, _ = compute_total_friction_cost(
            hold_symbol, target_etf,
            sell_amt, buy_amt,
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

    # ------------------------------------------------------------------
    # 决策：TOP-N 持有（新逻辑）
    # ------------------------------------------------------------------

    def _make_decision_top_n(self, idx: int, today_data: Dict,
                              momentum_series: pd.Series):
        """
        TOP-N 决策逻辑。

        获取动量排名前 TOP_N 的 ETF，与当前持仓对比：
        - 无持仓 → 买入前 TOP_N（需动量 > 0）
        - 有持仓 → 如果当前持有的 ETF 不在前 TOP_N，调仓
        """
        ranking = rank_etfs_by_momentum(momentum_series)
        targets = [ranking.get(i) for i in range(1, self.top_n + 1)]
        targets = [t for t in targets if t is not None]

        if not targets:
            return

        has_position = bool(self.positions)
        target_syms = set(targets)

        # ── 无持仓：开仓 ──
        if not has_position:
            top_mom = momentum_series.get(targets[0], -np.inf)
            if top_mom > 0:
                cash_per = self.cash / len(targets)
                for sym in targets:
                    self._buy(sym, cash_per, idx, today_data,
                              trade_type="买入",
                              reason=f"TOP-{len(targets)}动量开仓")
                self.risk_state.on_open_position(today_data[targets[0]]["close"])
            return

        # ── 有持仓：检查是否需要调仓 ──
        current_syms = set(self.positions.keys())

        # 需要卖出的 = 当前持仓不在目标集中
        to_sell = current_syms - target_syms
        # 需要买入的 = 目标集但当前未持有
        to_buy = target_syms - current_syms

        if not to_sell and not to_buy:
            return  # 已完美对齐

        # 简单检查：被替换的ETF确实弱于新目标
        # （TOP-N模式下不用MIN_SWITCH_CONVICTION，因为部分换仓风险分散，
        #  且动态窗口已降低了噪音。只要排名变化就执行。）
        if to_sell and to_buy:
            displaced_best = max(
                momentum_series.get(s, -np.inf) for s in to_sell
            )
            new_worst = min(
                momentum_series.get(s, -np.inf) for s in to_buy
            )
            if new_worst <= displaced_best:
                return  # 新目标并不更好，不切换

        # 执行调仓：卖出不在目标中的，买入缺失的目标
        for sym in list(self.positions.keys()):
            if sym not in target_syms:
                self._sell_all(idx, today_data,
                               trade_type="调仓卖出",
                               reason=f"TOP-{self.top_n}调仓卖出",
                               symbol=sym)

        # 均分现金买入目标中缺失的
        if to_buy and self.cash > 0:
            cash_per = self.cash / len(to_buy)
            for sym in to_buy:
                self._buy(sym, cash_per, idx, today_data,
                          trade_type="调仓买入",
                          reason=f"TOP-{self.top_n}调仓买入")

    # ------------------------------------------------------------------
    # 风控执行
    # ------------------------------------------------------------------

    def _execute_risk_exit(self, idx: int, today_data: Dict,
                           risk_action: str, reason: str):
        """执行风控平仓。"""
        ttm = {"stop_loss": "止损卖出", "stop_profit": "止盈卖出",
               "extreme_drawdown": "极端回撤清仓"}
        tt = ttm.get(risk_action, "风控卖出")
        self._sell_all(idx, today_data, trade_type=tt, reason=reason)
        self.adjustment_days_left = 0

        # 极端回撤触发后，将峰值重置为当前净值（避免死亡螺旋）
        if risk_action == "extreme_drawdown":
            current_value = self.cash  # 清仓后只剩现金
            self.risk_state.peak_total_value = current_value

    # ------------------------------------------------------------------
    # 期末处理
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
                    target_etf: str = "", momentum_str: str = "",
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
            target_etf=target_etf,
            momentum_str=momentum_str,
        ))

    def _get_action_desc(self) -> str:
        if self.adjustment_days_left > 0:
            done = self.adjustment_total_days - self.adjustment_days_left
            return f"调仓中({done}/{self.adjustment_total_days})"
        if not self.positions:
            return "空仓"
        return "持有"

    def _format_ranking(self, ranking: pd.Series, momentum: pd.Series) -> str:
        parts = []
        for rk in range(1, min(len(ranking) + 1, 6)):
            sym = ranking.get(rk)
            if sym:
                mom = momentum.get(sym, np.nan)
                if not np.isnan(mom):
                    parts.append(f"#{rk}{ETF_POOL.get(sym, sym)}({mom:.4f})")
        return " > ".join(parts)

    # ------------------------------------------------------------------
    # 结果导出
    # ------------------------------------------------------------------

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
