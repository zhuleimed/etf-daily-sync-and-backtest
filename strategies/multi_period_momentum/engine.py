"""
多周期动量共振回测引擎

继承 BacktestEngine，重写信号计算逻辑：
  1. 计算短/中/长三期动量
  2. 检查共振：至少 MIN_RESONANCE 期动量 > 0
  3. 用加权综合得分排序

效果：过滤"昙花一现"的短期反弹，只在趋势确认后才入场。
"""

from typing import Dict

import numpy as np
import pandas as pd

from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.momentum_signals import rank_etfs_by_momentum
from strategies.momentum_rotation.risk import run_all_risk_checks

from . import config as cfg


def compute_multi_momentum(
    etf_data: Dict[str, pd.DataFrame],
    date_idx: int,
    short: int = 5,
    medium: int = 10,
    long: int = 20,
) -> pd.DataFrame:
    """
    计算每只ETF在指定日期的三期动量。

    Returns
    -------
    pd.DataFrame
        columns: ['short', 'medium', 'long'], index = ETF代码
    """
    result = {}
    for sym in cfg.ETF_SYMBOLS:
        if sym not in etf_data:
            continue
        df = etf_data[sym]
        if date_idx < long:
            result[sym] = {"short": np.nan, "medium": np.nan, "long": np.nan}
            continue
        close = df.iloc[date_idx]["close"]
        result[sym] = {
            "short": close / df.iloc[max(0, date_idx - short)]["close"] - 1
            if date_idx >= short else np.nan,
            "medium": close / df.iloc[max(0, date_idx - medium)]["close"] - 1
            if date_idx >= medium else np.nan,
            "long": close / df.iloc[max(0, date_idx - long)]["close"] - 1
            if date_idx >= long else np.nan,
        }
    return pd.DataFrame(result).T


def compute_resonance_score(multi_mom: pd.DataFrame,
                            min_resonance: int = 3,
                            weights: tuple = (0.2, 0.3, 0.5)) -> pd.Series:
    """
    计算共振得分。

    对每只ETF：
      1. 检查短中长三期动量 > 0 的个数
      2. 不满足 min_resonance → 得分 = NaN（不参与排名）
      3. 满足 → 得分 = w1*短动量 + w2*中动量 + w3*长动量

    Returns
    -------
    pd.Series
        index = ETF代码, values = 共振综合得分（NaN=共振不满足）
    """
    scores = {}
    for sym in multi_mom.index:
        row = multi_mom.loc[sym]
        if row.isna().any():
            scores[sym] = np.nan
            continue
        # 统计多少期动量 > 0
        positive_count = sum([
            row["short"] > 0,
            row["medium"] > 0,
            row["long"] > 0,
        ])
        if positive_count < min_resonance:
            scores[sym] = np.nan  # 共振不满足
        else:
            scores[sym] = (weights[0] * row["short"] +
                           weights[1] * row["medium"] +
                           weights[2] * row["long"])
    return pd.Series(scores)


class MultiPeriodEngine(BacktestEngine):
    """多周期动量共振引擎。"""

    def __init__(self,
                 initial_capital: float = cfg.INITIAL_CAPITAL,
                 risk_mode: str = "",
                 momentum_short: int = None,
                 momentum_medium: int = None,
                 momentum_long: int = None,
                 min_resonance: int = None,
                 score_weights: tuple = None,
                 ):
        # 父类用 long 作为动量窗口（用于数据预计算）
        m_long = momentum_long if momentum_long is not None else cfg.MOMENTUM_LONG
        super().__init__(
            initial_capital=initial_capital,
            risk_mode=risk_mode or cfg.RISK_MODE,
            momentum_window=m_long,
            top_n=cfg.TOP_N,
            dynamic_window=False,
        )
        self.mom_short = momentum_short if momentum_short is not None else cfg.MOMENTUM_SHORT
        self.mom_medium = momentum_medium if momentum_medium is not None else cfg.MOMENTUM_MEDIUM
        self.mom_long = m_long
        self.min_resonance = min_resonance if min_resonance is not None else cfg.MIN_RESONANCE
        self.weights = score_weights if score_weights is not None else cfg.SCORE_WEIGHTS

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> "MultiPeriodEngine":
        """执行完整回测（多周期动量共振）。"""
        n = len(self.dates)
        if n == 0:
            raise RuntimeError("无交易日数据")

        etf_syms = cfg.ETF_SYMBOLS

        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(cfg.ETF_POOL.get(s, s) for s in etf_syms)}")
        print(f"  初始资金：{self.initial_capital:,.0f} 元")
        print(f"  动量周期：{self.mom_short}/{self.mom_medium}/{self.mom_long}日")
        print(f"  最小共振：{self.min_resonance}/3期  权重：{self.weights}")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in etf_syms}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # ── Step 2: 风控 ──
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

            # ── Step 4: 多周期动量共振信号 ──
            signal_idx = max(0, idx - 1)  # 避免 look-ahead
            multi_mom = compute_multi_momentum(
                self.etf_data, signal_idx,
                short=self.mom_short, medium=self.mom_medium,
                long=self.mom_long,
            )
            scores = compute_resonance_score(
                multi_mom, self.min_resonance, self.weights,
            )

            # ★ 共振得分替代动量值用于排名
            ranking = rank_etfs_by_momentum(scores)  # 复用排名函数（降序）
            target_etf = ranking.get(1) if len(ranking) > 0 else None
            momentum_str = self._format_resonance(ranking, scores, multi_mom)

            # ── Step 5: 决策（用 scores 替代 momentum）──
            if self.adjustment_days_left <= 0:
                # 将 scores 伪装成 momentum_series 传给父类决策
                self._make_decision(idx, today_data, hold_symbol, target_etf, scores)

            # ── Step 6: 记录 ──
            self._record_day(idx, today_data,
                             target_etf=target_etf or "",
                             momentum_str=momentum_str)

            self._days_since_last_switch += 1
            if self._switch_cooldown > 0:
                self._switch_cooldown -= 1

        self._close_remaining_positions()
        print(f"  回测完成 ✓")
        return self

    def _format_resonance(self, ranking, scores, multi_mom):
        """格式化排名，附带共振信息。"""
        parts = []
        for rk in range(1, min(len(ranking) + 1, 6)):
            sym = ranking.get(rk)
            if sym is None:
                continue
            s = scores.get(sym, np.nan)
            if np.isnan(s):
                continue
            row = multi_mom.loc[sym] if sym in multi_mom.index else None
            if row is not None:
                pos = sum([row["short"] > 0, row["medium"] > 0, row["long"] > 0])
                parts.append(f"#{rk}{cfg.ETF_POOL.get(sym, sym)}({s:.4f},{pos}/3)")
            else:
                parts.append(f"#{rk}{cfg.ETF_POOL.get(sym, sym)}({s:.4f})")
        return " > ".join(parts)
