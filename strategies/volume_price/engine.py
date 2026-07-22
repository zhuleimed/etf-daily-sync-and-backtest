"""
量价配合增强回测引擎

继承 BacktestEngine，在动量信号计算中加入成交量过滤：
  量比 = 近N日均量 / 近M日均量
  仅当量比 > 阈值时，该ETF才参与动量排名。

效果：过滤掉"缩量上涨"（可能是假突破），只追"放量上涨"（真金白银）。
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


def compute_volume_ratios(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    short_period: int = 5,
    long_period: int = 20,
) -> pd.Series:
    """
    计算每只ETF的量比（短期均量 / 长期均量）。

    Returns
    -------
    pd.Series
        index = ETF代码, values = 量比
        >1 = 放量, <1 = 缩量
    """
    ratios = {}
    for sym in cfg.ETF_SYMBOLS:
        if sym not in etf_data:
            ratios[sym] = np.nan
            continue
        df = etf_data[sym]
        if date_idx < long_period:
            ratios[sym] = np.nan
            continue
        # 短期均量
        short_vol = df.iloc[max(0, date_idx - short_period + 1):date_idx + 1]["volume"].mean()
        # 长期均量
        long_vol = df.iloc[max(0, date_idx - long_period + 1):date_idx + 1]["volume"].mean()
        ratios[sym] = short_vol / long_vol if long_vol > 0 else 1.0
    return pd.Series(ratios)


class VolumePriceEngine(BacktestEngine):
    """
    量价配合增强引擎。

    在父类动量决策之上增加成交量过滤层：
      每日先计算量比 → 过滤缩量ETF → 动量排名 → 正常决策
    """

    def __init__(self,
                 initial_capital: float = cfg.INITIAL_CAPITAL,
                 risk_mode: str = "",
                 momentum_window: int = cfg.MOMENTUM_WINDOW,
                 vol_short: int = None,
                 vol_long: int = None,
                 vol_threshold: float = None,
                 ):
        super().__init__(
            initial_capital=initial_capital,
            risk_mode=risk_mode or cfg.RISK_MODE,
            momentum_window=momentum_window,
            top_n=cfg.TOP_N,
            dynamic_window=cfg.DYNAMIC_WINDOW_ENABLED,
        )
        self.vol_short = vol_short if vol_short is not None else cfg.VOL_SHORT_PERIOD
        self.vol_long = vol_long if vol_long is not None else cfg.VOL_LONG_PERIOD
        self.vol_threshold = vol_threshold if vol_threshold is not None else cfg.VOL_THRESHOLD

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> "VolumePriceEngine":
        """执行完整回测（带量价过滤）。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")

        etf_syms = cfg.ETF_SYMBOLS

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(cfg.ETF_POOL.get(s, s) for s in etf_syms)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  量比周期：{self.vol_short}日/{self.vol_long}日  阈值：{self.vol_threshold:.1%}")
        print(f"  动量窗口：{self.momentum_window}日")
        print(f"  {'=' * 40}")

        # 统计量价过滤效果
        total_signals = 0
        filtered_signals = 0

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in etf_syms}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

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

            # ── Step 3: 渐进调仓 ──
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # ── Step 4: 动量信号 + 量价过滤 ──
            signal_idx = max(0, idx - 1)
            momentum = compute_momentum_signals(self.etf_data, signal_idx, self.momentum_window)

            # ★ 量价过滤：缩量ETF不参与排名
            vol_ratios = compute_volume_ratios(
                self.etf_data, signal_idx,
                short_period=self.vol_short,
                long_period=self.vol_long,
            )
            for sym in momentum.index:
                if sym in vol_ratios and not pd.isna(vol_ratios[sym]):
                    total_signals += 1
                    if vol_ratios[sym] < self.vol_threshold:
                        momentum[sym] = np.nan  # 量比不足 → 不参与排名
                        filtered_signals += 1

            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1) if len(ranking) > 0 else None

            # 格式化排名（含量比信息）
            momentum_str = self._format_ranking_with_vol(ranking, momentum, vol_ratios)

            # ── Step 5: 决策 ──
            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)

            # ── Step 6: 记录 ──
            self._record_day(idx, today_data,
                             target_etf=target_etf or "",
                             momentum_str=momentum_str)

            self._days_since_last_switch += 1
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1

        # ── 期末处理 ──
        self._close_remaining_positions()

        # 输出过滤统计
        if total_signals > 0:
            pct = filtered_signals / total_signals * 100
            print(f"  量价过滤: {filtered_signals}/{total_signals} ({pct:.1f}%) ETF-日被过滤")

        print(f"  回测完成 ✓")
        return self

    def _format_ranking_with_vol(self, ranking, momentum, vol_ratios):
        """格式化排名，附带量比信息。"""
        parts = []
        for rk in range(1, min(len(ranking) + 1, 6)):
            sym = ranking.get(rk)
            if sym:
                mom = momentum.get(sym, np.nan)
                vr = vol_ratios.get(sym, np.nan)
                if not np.isnan(mom):
                    vol_flag = "✓" if not np.isnan(vr) and vr >= self.vol_threshold else "✗"
                    parts.append(f"#{rk}{cfg.ETF_POOL.get(sym, sym)}({mom:.4f},量{vol_flag})")
        return " > ".join(parts)
