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
    ew_benchmark_return: float = 0.0
    excess_return_vs_benchmark: float = 0.0
    excess_return_vs_ew: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    switch_count: int = 0
    avg_hold_days: float = 0.0
    win_rate: float = 0.0
    trade_win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trade_cost: float = 0.0
    cost_ratio: float = 0.0
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 10000.0
    final_value: float = 10000.0

    def to_dict(self) -> dict:
        return {
            "累计收益率": f"{self.total_return:.2%}",
            "年化收益率": f"{self.annualized_return:.2%}",
            "沪深300收益": f"{self.benchmark_return:.2%}",
            "等权组合收益": f"{self.ew_benchmark_return:.2%}",
            "超额(沪深300)": f"{self.excess_return_vs_benchmark:.2%}",
            "超额(等权)": f"{self.excess_return_vs_ew:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}",
            "回撤持续天数": self.max_drawdown_duration,
            "年化波动率": f"{self.volatility:.2%}",
            "下行波动率": f"{self.downside_volatility:.2%}",
            "夏普比率": round(self.sharpe_ratio, 4),
            "Sortino比率": round(self.sortino_ratio, 4),
            "Calmar比率": round(self.calmar_ratio, 4),
            "总交易笔数": self.total_trades,
            "调仓切换次数": self.switch_count,
            "平均持仓天数": round(self.avg_hold_days, 1),
            "日胜率": f"{self.win_rate:.2%}",
            "交易胜率": f"{self.trade_win_rate:.2%}",
            "盈亏比": round(self.profit_factor, 4),
            "交易总成本": f"{self.total_trade_cost:.2f}",
            "成本占比": f"{self.cost_ratio:.2%}",
            "初始资金": f"{self.initial_capital:.2f}",
            "最终资金": f"{self.final_value:.2f}",
        }


class MetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate

    def compute(self, daily_records: list[DailyRecord], trade_records: list[TradeRecord],
                 initial_capital: float, benchmark_return: Optional[float] = None,
                 ew_benchmark_return: Optional[float] = None) -> BacktestMetrics:
        metrics = BacktestMetrics(initial_capital=initial_capital)
        if not daily_records:
            return metrics
        metrics.start_date = daily_records[0].date
        metrics.end_date = daily_records[-1].date
        metrics.final_value = daily_records[-1].total_value
        total_return = (metrics.final_value / initial_capital) - 1
        metrics.total_return = total_return
        n = len(daily_records)
        metrics.annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n) - 1
        if benchmark_return is not None:
            metrics.benchmark_return = benchmark_return
            metrics.excess_return_vs_benchmark = total_return - benchmark_return
        if ew_benchmark_return is not None:
            metrics.ew_benchmark_return = ew_benchmark_return
            metrics.excess_return_vs_ew = total_return - ew_benchmark_return
        daily_returns = pd.Series([r.daily_return for r in daily_records])
        metrics.volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        cumulative = pd.Series([1 + r.cumulative_return for r in daily_records])
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = drawdown.min()
        if metrics.max_drawdown < 0:
            dd_start = drawdown[drawdown < 0].index[0] if drawdown[drawdown < 0].any() else 0
            dd_end = drawdown.idxmin()
            recovery = cumulative.iloc[dd_end:] >= running_max.iloc[dd_end]
            dd_recovery = recovery[recovery].index.min() if recovery.any() else n
            metrics.max_drawdown_duration = int(dd_recovery - dd_start)
        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        downside = daily_returns[daily_returns < rf_daily]
        metrics.downside_volatility = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 0.0
        avg_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        downside_std = downside.std() if len(downside) > 0 else 1.0
        if std_daily > 0:
            metrics.sharpe_ratio = (avg_daily - rf_daily) / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
        if downside_std > 0:
            metrics.sortino_ratio = (avg_daily - rf_daily) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        metrics.total_trades = len(trade_records)
        metrics.total_trade_cost = sum(t.commission + t.tax for t in trade_records)
        metrics.cost_ratio = metrics.total_trade_cost / initial_capital if initial_capital > 0 else 0.0
        switch_trades = set()
        for t in trade_records:
            if t.trade_type in ("买入",):
                switch_trades.add(t.date)
            elif "调仓买入" in t.trade_type and "第1天" in t.reason:
                switch_trades.add(t.date)
        metrics.switch_count = len(switch_trades)
        all_sells = [t for t in trade_records if "卖出" in t.trade_type]
        if all_sells:
            metrics.avg_hold_days = sum(t.days_held for t in all_sells) / len(all_sells)
        metrics.win_rate = (daily_returns > 0).sum() / len(daily_returns)
        profit_trades = [t for t in trade_records if t.trade_type in
                         ("卖出", "止损卖出", "止盈卖出", "调仓卖出", "风控卖出", "极端回撤清仓", "虚拟卖出")]
        if profit_trades:
            profitable = [t for t in profit_trades if t.profit > 0]
            losing = [t for t in profit_trades if t.profit < 0]
            if profit_trades:
                metrics.trade_win_rate = len(profitable) / len(profit_trades)
            total_profit = sum(t.profit for t in profitable)
            total_loss = abs(sum(t.profit for t in losing))
            if total_loss > 0:
                metrics.profit_factor = total_profit / total_loss
        return metrics
