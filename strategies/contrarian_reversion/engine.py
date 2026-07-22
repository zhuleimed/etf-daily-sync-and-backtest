"""
反转/均值回归回测引擎

继承 BacktestEngine，重写决策逻辑：
  - 动量排名升序（最弱排第一）
  - 入场：最弱ETF收益 < ENTRY_THRESHOLD
  - 离场：止盈 / 止损 / 到期 / 切换到更弱者
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.momentum_rotation.engine import (
    BacktestEngine, DailyRecord, TradeRecord, BuyLot,
)
from strategies.momentum_rotation.data import (
    load_all_etf_data, load_benchmark_data, compute_equal_weight_benchmark,
)
from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)
from strategies.momentum_rotation.cost import compute_total_friction_cost
from strategies.momentum_rotation.risk import RiskState, run_all_risk_checks

from . import config as cfg


# ============================================================================
# 反转专用：升序排名（最弱=第1名）
# ============================================================================

def rank_etfs_by_weakness(momentum_series: pd.Series) -> pd.Series:
    """
    按动量值升序排列 ETF（最弱排第一）。

    Returns
    -------
    pd.Series
        index = 排名（1=最弱）, values = ETF 代码
    """
    valid = momentum_series.dropna()
    if valid.empty:
        return pd.Series(dtype=str)
    sorted_etfs = valid.sort_values(ascending=True)  # 升序：最弱排第一
    return pd.Series(sorted_etfs.index.values, index=range(1, len(sorted_etfs) + 1))


# ============================================================================
# 反转引擎
# ============================================================================

class ContrarianEngine(BacktestEngine):
    """
    反转/均值回归回测引擎。

    与动量引擎的核心区别：
      1. 排名升序——买最弱的，而非最强的
      2. 入场条件——最弱ETF必须真的在跌（收益 < 阈值）
      3. 离场条件——止盈/止损/到期，而非"持有最强"
      4. 切换条件——新目标"显著更弱"才切换（带摩擦成本校验）
      5. 默认开启风控（RISK_MODE=B），因为反转策略风险更高
    """

    def __init__(self,
                 initial_capital: float = cfg.INITIAL_CAPITAL,
                 risk_mode: str = "",
                 momentum_window: int = cfg.REVERSION_WINDOW,
                 reversion_window: int = None,
                 entry_threshold: float = None,
                 profit_target: float = None,
                 stop_loss: float = None,
                 max_hold_days: int = None,
                 min_hold_days: int = None,
                 min_switch_conviction: float = None,
                 ):
        # 调用父类构造（父类用 momentum_window 参数名，但对我们来说是 reversion_window）
        super().__init__(
            initial_capital=initial_capital,
            risk_mode=risk_mode or cfg.RISK_MODE,
            momentum_window=momentum_window,
            top_n=1,  # 反转策略每次只持有一只
            dynamic_window=False,
        )
        # 反转专属参数
        self.reversion_window = reversion_window or cfg.REVERSION_WINDOW
        self.entry_threshold = entry_threshold if entry_threshold is not None else cfg.ENTRY_THRESHOLD
        self.profit_target = profit_target if profit_target is not None else cfg.PROFIT_TARGET
        self.stop_loss = stop_loss if stop_loss is not None else cfg.STOP_LOSS
        self.max_hold_days = max_hold_days if max_hold_days is not None else cfg.MAX_HOLD_DAYS
        self.min_hold_days = min_hold_days if min_hold_days is not None else cfg.MIN_HOLD_DAYS
        self.min_switch_conviction = (min_switch_conviction
                                      if min_switch_conviction is not None
                                      else cfg.MIN_SWITCH_CONVICTION)

        # 追踪持仓以来的盈亏（用于止盈/止损判断）
        self._entry_price: float = 0.0       # 当前持仓的买入均价
        self._entry_date_idx: int = 0         # 当前持仓的买入日期索引
        self._holding_return: float = 0.0     # 当前持仓的浮动收益率

    # ------------------------------------------------------------------
    # 主流程（重写，核心变化在 Step 4 & 5）
    # ------------------------------------------------------------------

    def run(self) -> "ContrarianEngine":
        """执行完整回测（反转逻辑）。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据，请先调用 load_data()")

        etf_syms = cfg.ETF_SYMBOLS

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(cfg.ETF_POOL.get(s, s) for s in etf_syms)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  反转窗口：{self.reversion_window} 日")
        print(f"  入场阈值：{self.entry_threshold:.1%}  止盈：{self.profit_target:.1%}")
        print(f"  止损：{self.stop_loss:.1%}  最大持有：{self.max_hold_days} 天")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in etf_syms}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── 更新持仓收益率 ──
            if has_position and hold_symbol and self._entry_price > 0:
                current_price = today_data[hold_symbol]["close"]
                self._holding_return = current_price / self._entry_price - 1

            # ── Step 2: 风控检查 ──
            if self.risk_mode != "A" and has_position and hold_symbol:
                hold_row = today_data[hold_symbol]
                total_value = self._calc_total_value(today_data)
                self.risk_state.update_peak(hold_row["high"])
                self.risk_state.update_peak_total_value(total_value)
                risk_action, risk_reason = run_all_risk_checks(
                    self.risk_state, total_value, has_position,
                    hold_symbol, hold_row["high"], hold_row["low"],
                    hold_row["close"], hold_row["atr"],
                    self.etf_data, idx, mode=self.risk_mode,
                )
                if risk_action != "none":
                    self._execute_risk_exit(idx, today_data, risk_action, risk_reason)
                    self._record_day(idx, today_data, action_override=risk_action)
                    continue

            # ── Step 3: 渐进调仓（反转策略用 ADJUSTMENT_DAYS=1，即立即执行）──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── Step 4: 反转信号 —— 升序排名（最弱=第1）──
            signal_idx = max(0, idx - 1)  # 用前一日收盘数据，避免 look-ahead
            momentum = compute_momentum_signals(
                self.etf_data, signal_idx, self.reversion_window,
            )
            ranking = rank_etfs_by_weakness(momentum)  # ★ 升序：最弱第一
            weakest_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = self._format_ranking(ranking, momentum)

            # ── Step 5: 反转决策 ──
            if self.adjustment_days_left <= 0:
                self._make_decision_contrarian(
                    idx, today_data, hold_symbol, weakest_etf, momentum,
                )

            # ── Step 6: 记录 ──
            self._record_day(idx, today_data,
                             target_etf=weakest_etf or "",
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
    # 反转决策（核心！）
    # ------------------------------------------------------------------

    def _make_decision_contrarian(
        self,
        idx: int,
        today_data: Dict,
        hold_symbol: Optional[str],
        weakest_etf: Optional[str],
        momentum_series: pd.Series,
    ):
        """
        反转策略决策逻辑。

        状态机：
          [空仓] ──最弱ETF收益<阈值──→ [持仓]
          [持仓] ──止盈/止损/到期──→ [空仓]
          [持仓] ──新目标显著更弱──→ [切换]
          [持仓] ──继续等待──────→ [持有]
        """
        if weakest_etf is None:
            return

        weakest_mom = momentum_series.get(weakest_etf, np.nan)
        if np.isnan(weakest_mom):
            return

        has_position = hold_symbol is not None

        # ═══════════════════════════════════════════════════════
        # 情况 A：空仓 → 检查入场条件
        # ═══════════════════════════════════════════════════════
        if not has_position:
            if weakest_mom < self.entry_threshold:
                # 最弱ETF确实跌够了 → 开仓买入
                self._buy(
                    weakest_etf, self.cash, idx, today_data,
                    trade_type="买入",
                    reason=f"反转入场：{weakest_etf} {self.reversion_window}日收益={weakest_mom:.4f} < {self.entry_threshold:.1%}",
                )
                self.risk_state.on_open_position(today_data[weakest_etf]["close"])
                # 记录入场价和日期，用于后续止盈/止损计算
                self._entry_price = today_data[weakest_etf]["close"]
                self._entry_date_idx = idx
                self._holding_return = 0.0
                self._days_since_last_switch = 0
            return

        # ═══════════════════════════════════════════════════════
        # 情况 B：有持仓 → 检查离场条件
        # ═══════════════════════════════════════════════════════

        hold_mom = momentum_series.get(hold_symbol, np.nan)
        if np.isnan(hold_mom):
            return

        # B1. 止盈检查：持仓收益达到目标
        if self._holding_return >= self.profit_target:
            self._sell_all(
                idx, today_data,
                trade_type="止盈卖出",
                reason=f"止盈：{hold_symbol} 收益={self._holding_return:.2%} >= {self.profit_target:.1%}",
                symbol=hold_symbol,
            )
            self._entry_price = 0.0
            self._holding_return = 0.0
            return

        # B2. 止损检查：持仓亏损超过阈值
        if self._holding_return <= self.stop_loss:
            self._sell_all(
                idx, today_data,
                trade_type="止损卖出",
                reason=f"止损：{hold_symbol} 亏损={self._holding_return:.2%} <= {self.stop_loss:.1%}",
                symbol=hold_symbol,
            )
            self._entry_price = 0.0
            self._holding_return = 0.0
            return

        # B3. 到期检查：持仓超过最大天数
        days_held = idx - self._entry_date_idx
        if days_held >= self.max_hold_days:
            self._sell_all(
                idx, today_data,
                trade_type="到期卖出",
                reason=f"到期：{hold_symbol} 持仓{days_held}天 >= {self.max_hold_days}天，收益={self._holding_return:.2%}",
                symbol=hold_symbol,
            )
            self._entry_price = 0.0
            self._holding_return = 0.0
            return

        # B4. 同一标的：继续持有
        if weakest_etf == hold_symbol:
            return

        # B5. 切换检查：新目标是否"显著更弱"？
        # 最小持仓天数过滤
        if self._days_since_last_switch < self.min_hold_days:
            return

        # 摩擦成本校验
        hold_shares = self.positions.get(hold_symbol, 0)
        sell_amt = hold_shares * today_data[hold_symbol]["close"]
        buy_amt = sell_amt
        friction, _ = compute_total_friction_cost(
            hold_symbol, weakest_etf,
            sell_amt, buy_amt,
            today_data[hold_symbol]["close"],
            today_data[weakest_etf]["close"],
            self.etf_data, idx,
        )
        # extra_weakness = 新目标比当前持仓"弱"多少（负值更负 = 更弱）
        extra_weakness = hold_mom - weakest_mom  # 正数表示新目标更弱
        total_trade_amt = sell_amt + buy_amt
        friction_ratio = friction / total_trade_amt if total_trade_amt > 0 else 1.0
        switch_threshold = max(friction_ratio, self.min_switch_conviction)

        if extra_weakness > switch_threshold:
            self._start_adjustment(
                hold_symbol, weakest_etf, idx, today_data,
            )
            self._entry_price = today_data[weakest_etf]["close"]
            self._entry_date_idx = idx
            self._holding_return = 0.0

    # ------------------------------------------------------------------
    # 主流程用到的复用方法（从父类继承，无需重写）
    # ------------------------------------------------------------------
    # _get_hold_symbol, _calc_total_value, _calc_stock_value,
    # _buy, _sell, _sell_all, _start_adjustment, _execute_adjustment_step,
    # _finish_adjustment, _execute_risk_exit, _close_remaining_positions,
    # _record_day, _format_ranking, get_daily_df, get_trade_df
    # 均继承自 BacktestEngine，无需改动。
