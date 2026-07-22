"""
相对强度持久性回测引擎

继承 BacktestEngine，在动量信号上叠加"跑赢同类的持续性"过滤：
  1. 计算每只ETF过去N天跑赢等权平均的天数占比
  2. 持续性不达标的ETF不参与排名
  3. 通过过滤的ETF按绝对动量排名
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


def compute_persistence(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    window: int = 20,
) -> pd.Series:
    """
    计算每只ETF的相对强度持续性：过去N天中跑赢等权平均的天数占比。

    Returns
    -------
    pd.Series
        index = ETF代码, values = 持续性 (0.0-1.0)
    """
    if date_idx < window:
        return pd.Series({sym: 0.5 for sym in cfg.ETF_SYMBOLS})

    syms = cfg.ETF_SYMBOLS
    persistence = {}

    for sym in syms:
        if sym not in etf_data:
            persistence[sym] = 0.5
            continue
        df = etf_data[sym]
        beat_count = 0
        total = 0
        for i in range(date_idx - window + 1, date_idx + 1):
            if i <= 0:
                continue
            # 该ETF的日收益率
            etf_ret = df.iloc[i]["pct_chg"]
            # 等权平均（所有ETF日收益率的均值）
            avg_ret = np.mean([
                etf_data[s].iloc[i]["pct_chg"]
                for s in syms if s in etf_data and i < len(etf_data[s])
            ])
            total += 1
            if etf_ret > avg_ret:
                beat_count += 1
        persistence[sym] = beat_count / max(total, 1)

    return pd.Series(persistence)


class RelativeStrengthEngine(BacktestEngine):
    """相对强度持久性引擎。"""

    def __init__(self,
                 initial_capital: float = cfg.INITIAL_CAPITAL,
                 risk_mode: str = "",
                 momentum_window: int = cfg.MOMENTUM_WINDOW,
                 relative_window: int = None,
                 min_persistence: float = None,
                 ):
        super().__init__(
            initial_capital=initial_capital,
            risk_mode=risk_mode or cfg.RISK_MODE,
            momentum_window=momentum_window,
            top_n=cfg.TOP_N,
            dynamic_window=False,
        )
        self.rel_window = relative_window if relative_window is not None else cfg.RELATIVE_WINDOW
        self.min_persistence = (min_persistence if min_persistence is not None
                                else cfg.MIN_PERSISTENCE)

    def run(self) -> "RelativeStrengthEngine":
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")

        etf_syms = cfg.ETF_SYMBOLS
        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(cfg.ETF_POOL.get(s, s) for s in etf_syms)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  相对强度窗口：{self.rel_window}日  最小持续性：{self.min_persistence:.0%}")
        print(f"  动量窗口：{self.momentum_window}日")
        print(f"  {'=' * 40}")

        total_signals = 0; filtered_signals = 0

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in etf_syms}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # 风控
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

            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # 动量信号 + 持续性过滤
            signal_idx = max(0, idx - 1)
            momentum = compute_momentum_signals(self.etf_data, signal_idx, self.momentum_window)
            persistence = compute_persistence(self.etf_data, signal_idx, self.rel_window)

            # ★ 持续性过滤：不达标的ETF不参与排名
            for sym in momentum.index:
                if sym in persistence:
                    total_signals += 1
                    if persistence[sym] < self.min_persistence:
                        momentum[sym] = np.nan
                        filtered_signals += 1

            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = self._format_with_persistence(ranking, momentum, persistence)

            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)

            self._record_day(idx, today_data,
                             target_etf=target_etf or "",
                             momentum_str=momentum_str)
            self._days_since_last_switch += 1
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1

        self._close_remaining_positions()
        if total_signals > 0:
            print(f"  持续性过滤: {filtered_signals}/{total_signals} "
                  f"({filtered_signals/total_signals*100:.1f}%) ETF-日被过滤")
        print(f"  回测完成 ✓")
        return self

    def _format_with_persistence(self, ranking, momentum, persistence):
        parts = []
        for rk in range(1, min(len(ranking) + 1, 6)):
            sym = ranking.get(rk)
            if sym is None:
                continue
            mom = momentum.get(sym, np.nan)
            if np.isnan(mom):
                continue
            p = persistence.get(sym, np.nan)
            p_str = f"{p:.0%}" if not np.isnan(p) else "?"
            parts.append(f"#{rk}{cfg.ETF_POOL.get(sym, sym)}({mom:.4f},p{p_str})")
        return " > ".join(parts)
