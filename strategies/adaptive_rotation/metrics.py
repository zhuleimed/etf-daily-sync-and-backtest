"""绩效指标计算模块"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .engine import DailyRecord, TradeRecord

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return_vs_benchmark: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    switch_count: int = 0
    win_rate: float = 0.0
    initial_capital: float = 10000.0
    final_value: float = 10000.0

    def to_dict(self) -> dict:
        return {"累计收益率": f"{self.total_return:.2%}", "年化收益率": f"{self.annualized_return:.2%}",
                "沪深300收益": f"{self.benchmark_return:.2%}", "超额(沪深300)": f"{self.excess_return_vs_benchmark:.2%}",
                "最大回撤": f"{self.max_drawdown:.2%}", "夏普比率": round(self.sharpe_ratio, 4),
                "调仓切换次数": self.switch_count, "日胜率": f"{self.win_rate:.2%}",
                "初始资金": f"{self.initial_capital:.2f}", "最终资金": f"{self.final_value:.2f}"}


class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def compute(self, daily_records, trade_records, initial_capital, benchmark_return=None):
        metrics = BacktestMetrics(initial_capital=initial_capital)
        if not daily_records:
            return metrics
        metrics.final_value = daily_records[-1].total_value
        total_return = metrics.final_value / initial_capital - 1
        metrics.total_return = total_return
        n = len(daily_records)
        metrics.annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1
        if benchmark_return is not None:
            metrics.benchmark_return = benchmark_return
            metrics.excess_return_vs_benchmark = total_return - benchmark_return
        daily_returns = pd.Series([r.daily_return for r in daily_records])
        metrics.volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        cumulative = pd.Series([1 + r.cumulative_return for r in daily_records])
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = drawdown.min()
        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        avg_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        if std_daily > 0:
            metrics.sharpe_ratio = (avg_daily - rf_daily) / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
        metrics.total_trades = len(trade_records)
        switch_trades = set()
        for t in trade_records:
            if t.trade_type in ("买入",):
                switch_trades.add(t.date)
        metrics.switch_count = len(switch_trades)
        metrics.win_rate = (daily_returns > 0).sum() / len(daily_returns)
        return metrics
