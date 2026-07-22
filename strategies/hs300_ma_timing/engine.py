"""
沪深300均线择时回测引擎

HS300 > MA(N) → 正常动量轮动
HS300 < MA(N) → 全部空仓
"""

from typing import Dict
import numpy as np
import pandas as pd

from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.momentum_signals import (
    compute_momentum_signals, rank_etfs_by_momentum,
)
from strategies.momentum_rotation.risk import run_all_risk_checks
from . import config as cfg


class HS300MATimingEngine(BacktestEngine):
    """HS300均线择时引擎。"""

    def __init__(self, initial_capital=cfg.INITIAL_CAPITAL,
                 risk_mode="", momentum_window=cfg.MOMENTUM_WINDOW,
                 ma_period=None):
        super().__init__(initial_capital=initial_capital,
                         risk_mode=risk_mode or cfg.RISK_MODE,
                         momentum_window=momentum_window,
                         top_n=cfg.TOP_N, dynamic_window=False)
        self.ma_period = ma_period or cfg.MA_PERIOD
        self.hs300_data: pd.DataFrame = None  # 需在 run() 前设置

    def run(self) -> "HS300MATimingEngine":
        n = len(self.dates)
        if n == 0 or self.hs300_data is None:
            raise RuntimeError("无数据")

        syms = cfg.ETF_SYMBOLS
        # 对齐HS300数据到ETF的交易日
        hs300_close = self.hs300_data.set_index("date")["close"]

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  HS300均线：{self.ma_period}日  动量窗口：{self.momentum_window}日")
        print(f"  {'=' * 40}")

        in_market_days = 0; out_market_days = 0

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in syms}
            date_str = str(self.dates[idx].date())
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── HS300均线择时 ──
            if date_str in hs300_close.index:
                hs300_val = hs300_close.loc[date_str]
                # 找HS300数据中对应日期的索引
                hs_idx = hs300_close.index.get_loc(date_str)
                if isinstance(hs_idx, slice): hs_idx = hs_idx.start
                if hs_idx >= self.ma_period:
                    ma_val = hs300_close.iloc[hs_idx - self.ma_period + 1:hs_idx + 1].mean()
                else:
                    ma_val = hs300_val  # 数据不足，默认持有（不择时）
            else:
                hs300_val = ma_val = 0
                in_market_days += 1
                # fallback: 无HS300数据时默认正常交易
                self._run_momentum_day(idx, today_data, has_position, hold_symbol, syms)
                continue

            bull_market = hs300_val > ma_val
            if bull_market:
                in_market_days += 1
            else:
                out_market_days += 1

            # ── 风控 ──
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
                                     momentum_str=f"HS300={hs300_val:.0f} MA={ma_val:.0f} {'牛' if bull_market else '熊'}")
                    continue

            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── 择时判断 ──
            if not bull_market:
                if has_position:
                    self._sell_all(idx, today_data, trade_type="择时清仓",
                                   reason=f"HS300({hs300_val:.0f})<MA{self.ma_period}({ma_val:.0f})")
                self._record_day(idx, today_data, action_override="空仓(HS300<MA)",
                                 momentum_str=f"HS300={hs300_val:.0f}<MA={ma_val:.0f} 空仓")
                self._days_since_last_switch += 1
                continue

            # ── 牛市中正常动量轮动 ──
            self._run_momentum_day(idx, today_data, has_position, hold_symbol, syms,
                                   extra_str=f"HS300>{self.ma_period}MA ")

        self._close_remaining_positions()
        print(f"  在场: {in_market_days}天  空仓: {out_market_days}天 "
              f"({out_market_days/(in_market_days+out_market_days)*100:.0f}%)")
        print(f"  回测完成 ✓")
        return self

    def _run_momentum_day(self, idx, today_data, has_position, hold_symbol, syms, extra_str=""):
        """正常动量轮动一天（与父类run逻辑相同）。"""
        signal_idx = max(0, idx - 1)
        momentum = compute_momentum_signals(self.etf_data, signal_idx, self.momentum_window)
        ranking = rank_etfs_by_momentum(momentum)
        target = ranking.get(1) if len(ranking) > 0 else None
        mom_str = extra_str + self._format_ranking(ranking, momentum)

        if self.adjustment_days_left <= 0:
            self._make_decision(idx, today_data, hold_symbol, target, momentum)

        self._record_day(idx, today_data, target_etf=target or "", momentum_str=mom_str)
        self._days_since_last_switch += 1
        if self._switch_cooldown > 0:
            self._switch_cooldown -= 1
