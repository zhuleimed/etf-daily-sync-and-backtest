"""
市场宽度择时回测引擎

继承 BacktestEngine，在动量决策前加入宽度过滤：
  宽度 > STRONG → 正常动量轮动
  宽度在 WEAK-STRONG 之间 → 中性（空仓/半仓）
  宽度 < WEAK → 全部空仓
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)
from strategies.momentum_rotation.cost import compute_total_friction_cost
from strategies.momentum_rotation.risk import run_all_risk_checks

from . import config as cfg


# ============================================================================
# 宽度计算（核心！）
# ============================================================================

def compute_breadth(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    ma_period: int = 20,
) -> float:
    """
    计算市场宽度：价格 > MA(N) 的 ETF 占比。

    Parameters
    ----------
    etf_data : dict
        {symbol: DataFrame}，每只 ETF 的日线数据
    date_idx : int
        当前日期索引
    ma_period : int
        均线周期

    Returns
    -------
    float : 0.0-1.0，站上均线的ETF比例
    """
    if date_idx < ma_period:
        return 0.5  # 数据不足时返回中性

    above = 0
    total = 0
    for sym, df in etf_data.items():
        if sym not in cfg.ETF_SYMBOLS:
            continue
        if date_idx >= len(df):
            continue
        total += 1
        close = df.iloc[date_idx]["close"]
        # 用前一日收盘计算 MA（避免 look-ahead）
        # 实际上站上均线的判断用当日收盘 vs 前N日均值，不存在look-ahead
        ma = df.iloc[max(0, date_idx - ma_period + 1):date_idx + 1]["close"].mean()
        if close > ma:
            above += 1

    return above / max(total, 1)


def determine_regime(breadth: float, strong: float = None, weak: float = None) -> str:
    """
    根据宽度确定市场状态。

    Returns
    -------
    str : "bull" | "neutral" | "bear"
    """
    st = strong if strong is not None else cfg.BREADTH_STRONG
    wk = weak if weak is not None else cfg.BREADTH_WEAK
    if breadth >= st:
        return "bull"
    elif breadth <= wk:
        return "bear"
    else:
        return "neutral"


def should_exit_regime(regime: str, neutral_mode: str) -> bool:
    """
    当前市场状态下是否应该清仓（或至少不参与动量轮动）。

    强市 → 不禁（正常交易）
    弱市 → 清仓
    中性 → 按 NEUTRAL_MODE 决定
    """
    if regime == "bear":
        return True
    if regime == "neutral":
        return neutral_mode == "cash"
    return False


# ============================================================================
# 宽度择时引擎
# ============================================================================

class BreadthTimingEngine(BacktestEngine):
    """
    市场宽度择时引擎。

    在父类的动量决策之上增加宽度过滤层：
      每日先计算市场宽度 → 确定市场状态 → 决定是否允许交易
    """

    def __init__(self,
                 initial_capital: float = cfg.INITIAL_CAPITAL,
                 risk_mode: str = "",
                 momentum_window: int = cfg.MOMENTUM_WINDOW,
                 ma_period: int = None,
                 breadth_strong: float = None,
                 breadth_weak: float = None,
                 neutral_mode: str = None,
                 ):
        super().__init__(
            initial_capital=initial_capital,
            risk_mode=risk_mode or cfg.RISK_MODE,
            momentum_window=momentum_window,
            top_n=cfg.TOP_N,
            dynamic_window=cfg.DYNAMIC_WINDOW_ENABLED,
        )
        self.ma_period = ma_period if ma_period is not None else cfg.BREADTH_MA_PERIOD
        self.breadth_strong = breadth_strong if breadth_strong is not None else cfg.BREADTH_STRONG
        self.breadth_weak = breadth_weak if breadth_weak is not None else cfg.BREADTH_WEAK
        self.neutral_mode = neutral_mode or cfg.NEUTRAL_MODE

    # ------------------------------------------------------------------
    # 主流程（重写 run，在动量决策前加入宽度过滤）
    # ------------------------------------------------------------------

    def run(self) -> "BreadthTimingEngine":
        """执行完整回测（带宽度择时）。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据，请先调用 load_data()")

        etf_syms = cfg.ETF_SYMBOLS

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(cfg.ETF_POOL.get(s, s) for s in etf_syms)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  宽度MA周期：{self.ma_period}日")
        print(f"  强市阈值：{self.breadth_strong:.0%}  弱市阈值：{self.breadth_weak:.0%}")
        print(f"  中性模式：{self.neutral_mode}")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in etf_syms}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── 计算市场宽度 ──
            breadth = compute_breadth(self.etf_data, idx, self.ma_period)
            regime = determine_regime(breadth, self.breadth_strong, self.breadth_weak)

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
                    self._record_day(idx, today_data, action_override=risk_action,
                                     momentum_str=f"宽度={breadth:.1%}({regime})")
                    continue

            # ── Step 3: 渐进调仓 ──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── Step 4: 宽度过滤（核心！）──
            if should_exit_regime(regime, self.neutral_mode):
                # 弱市或中性（空仓模式）→ 清仓并跳过动量决策
                if has_position:
                    self._sell_all(idx, today_data,
                                   trade_type="宽度清仓",
                                   reason=f"市场宽度={breadth:.1%}({regime})，清仓避险")
                self._record_day(idx, today_data, action_override=f"空仓({regime})",
                                 momentum_str=f"宽度={breadth:.1%}({regime})")
                self._days_since_last_switch += 1
                if self._switch_cooldown > 0:
                    self._switch_cooldown -= 1
                continue

            # ── Step 5: 动量信号（强市或中性半仓模式）──
            signal_idx = max(0, idx - 1)
            momentum = compute_momentum_signals(self.etf_data, signal_idx, self.momentum_window)
            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = f"宽度={breadth:.1%}({regime}) " + self._format_ranking(ranking, momentum)

            # ── Step 6: 决策 ──
            if self.adjustment_days_left <= 0:
                if regime == "neutral" and self.neutral_mode == "half":
                    # 半仓模式：仓位不超过总资产的一半
                    total_val = self._calc_total_value(today_data)
                    stock_val = self._calc_stock_value(today_data)
                    # 允许买入的金额上限 = half_target - 当前持仓市值
                    max_buy = max(0.0, total_val / 2 - stock_val)
                    pre_cash = self.cash
                    self.cash = min(self.cash, max_buy)
                    self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)
                    # self.cash 现在是 spending_limit - spent
                    # 恢复 = 原现金 - 实际花费
                    actual_spent = min(pre_cash, max_buy) - self.cash
                    self.cash = pre_cash - actual_spent
                else:
                    self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)

            # ── Step 7: 记录 ──
            self._record_day(idx, today_data,
                             target_etf=target_etf or "",
                             momentum_str=momentum_str)

            self._days_since_last_switch += 1
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1

        # ── 期末处理 ──
        self._close_remaining_positions()
        print(f"  回测完成 ✓")
        return self
