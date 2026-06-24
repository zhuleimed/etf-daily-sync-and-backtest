"""
组合策略引擎 — 动量轮动 + 配对交易

将资金分配给两个子策略，合并每日净值：
  - 动量轮动（X%）：趋势跟踪，主攻收益
  - 配对交易（1-X%）：市场中性，降低回撤
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    TOTAL_CAPITAL, MOMENTUM_PCT, PAIR_PCT,
    MOMENTUM_CONFIG, PAIR_CONFIG,
)
from strategies.momentum_rotation.engine import BacktestEngine as MomentumEngine
from strategies.pair_trading.engine import PairTradingEngine as PairEngine


@dataclass
class DailyRecord:
    date: str = ""
    mom_value: float = 0.0
    pair_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0


class CombinedEngine:
    """组合策略引擎。"""

    def __init__(self, total_capital: float = TOTAL_CAPITAL):
        self.total_capital = total_capital
        mom_capital = total_capital * MOMENTUM_PCT
        pair_capital = total_capital * PAIR_PCT

        self.mom_engine = MomentumEngine(initial_capital=mom_capital,
                                          risk_mode=MOMENTUM_CONFIG.get("risk_mode", "A"))
        from strategies.pair_trading.config import PAIRS as PT_PAIRS
        pair_cpp = pair_capital / max(len(PT_PAIRS), 1)
        self.pair_engine = PairEngine(initial_capital=pair_capital, capital_per_pair=pair_cpp)

        self.daily_records: List[DailyRecord] = []

    def load_data(self, start_date: str = "2024-01-01",
                  end_date: str = ""):
        """同时加载两个子策略的数据。"""
        self.mom_engine.load_data(start_date=start_date, end_date=end_date)
        self.pair_engine.load_data(start_date=start_date, end_date=end_date)

    def run(self):
        """运行两个子策略后合并净值。"""
        print(f"\n  组合策略: 动量{MOMENTUM_PCT:.0%} + 配对{PAIR_PCT:.0%}")
        print(f"  总资金: {self.total_capital:,.0f} 元")
        print(f"  动量分配: {self.total_capital * MOMENTUM_PCT:,.0f}")
        print(f"  配对分配: {self.total_capital * PAIR_PCT:,.0f}")
        print(f"  {'=' * 40}")

        self.mom_engine.run()
        self.pair_engine.run()

        # 合并净值
        mom_df = self.mom_engine.get_daily_df()[["date", "total_value"]].copy()
        mom_df.columns = ["date", "mom_value"]
        pair_df = self.pair_engine.get_daily_df()[["date", "total_value"]].copy()
        pair_df.columns = ["date", "pair_value"]

        combined = pd.merge(mom_df, pair_df, on="date", how="inner")
        combined["total_value"] = combined["mom_value"] + combined["pair_value"]

        # 计算收益序列
        prev_total = self.total_capital
        for _, row in combined.iterrows():
            daily_ret = (row["total_value"] - prev_total) / prev_total if prev_total > 0 else 0.0
            cum_ret = row["total_value"] / self.total_capital - 1
            self.daily_records.append(DailyRecord(
                date=row["date"],
                mom_value=round(row["mom_value"], 2),
                pair_value=round(row["pair_value"], 2),
                total_value=round(row["total_value"], 2),
                daily_return=round(daily_ret, 6),
                cumulative_return=round(cum_ret, 6),
            ))
            prev_total = row["total_value"]

        print(f"  组合回测完成 ✓")

    def get_daily_df(self) -> pd.DataFrame:
        rows = []
        for r in self.daily_records:
            rows.append({
                "date": r.date, "mom_value": r.mom_value,
                "pair_value": r.pair_value, "total_value": r.total_value,
                "daily_return": r.daily_return,
                "cumulative_return": r.cumulative_return,
            })
        return pd.DataFrame(rows)

    def get_trade_df(self) -> pd.DataFrame:
        return pd.DataFrame()
