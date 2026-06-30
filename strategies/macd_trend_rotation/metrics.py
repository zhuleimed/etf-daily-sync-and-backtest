"""绩效指标计算模块"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .engine import DailyRecord, TradeRecord

TRADING_DAYS = 252


@dataclass
class BacktestMetrics:
    total_return = 0.0; annualized_return = 0.0; benchmark_return = 0.0
    ew_benchmark_return = 0.0; excess_return_vs_benchmark = 0.0; excess_return_vs_ew = 0.0
    max_drawdown = 0.0; max_drawdown_duration = 0; volatility = 0.0; downside_volatility = 0.0
    sharpe_ratio = 0.0; sortino_ratio = 0.0; calmar_ratio = 0.0
    total_trades = 0; switch_count = 0; avg_hold_days = 0.0
    win_rate = 0.0; trade_win_rate = 0.0; profit_factor = 0.0
    total_trade_cost = 0.0; cost_ratio = 0.0
    start_date = ""; end_date = ""; initial_capital = 10000.0; final_value = 10000.0

    def to_dict(self):
        return {
            "累计收益率": f"{self.total_return:.2%}", "年化收益率": f"{self.annualized_return:.2%}",
            "沪深300收益": f"{self.benchmark_return:.2%}", "等权组合收益": f"{self.ew_benchmark_return:.2%}",
            "超额(沪深300)": f"{self.excess_return_vs_benchmark:.2%}", "超额(等权)": f"{self.excess_return_vs_ew:.2%}",
            "最大回撤": f"{self.max_drawdown:.2%}", "回撤持续天数": self.max_drawdown_duration,
            "年化波动率": f"{self.volatility:.2%}", "下行波动率": f"{self.downside_volatility:.2%}",
            "夏普比率": round(self.sharpe_ratio, 4), "Sortino比率": round(self.sortino_ratio, 4),
            "Calmar比率": round(self.calmar_ratio, 4), "总交易笔数": self.total_trades,
            "调仓切换次数": self.switch_count, "平均持仓天数": round(self.avg_hold_days, 1),
            "日胜率": f"{self.win_rate:.2%}", "交易胜率": f"{self.trade_win_rate:.2%}",
            "盈亏比": round(self.profit_factor, 4), "交易总成本": f"{self.total_trade_cost:.2f}",
            "成本占比": f"{self.cost_ratio:.2%}", "初始资金": f"{self.initial_capital:.2f}",
            "最终资金": f"{self.final_value:.2f}",
        }


class MetricsCalculator:
    def __init__(self, rf=0.03): self.rf = rf

    def compute(self, drs, trs, ic, br=None, ewr=None):
        m = BacktestMetrics(); m.initial_capital = ic
        if not drs: return m
        m.start_date = drs[0].date; m.end_date = drs[-1].date; m.final_value = drs[-1].total_value
        tr = m.final_value / ic - 1; m.total_return = tr; n = len(drs)
        m.annualized_return = (1 + tr) ** (TRADING_DAYS / n) - 1
        if br is not None: m.benchmark_return = br; m.excess_return_vs_benchmark = tr - br
        if ewr is not None: m.ew_benchmark_return = ewr; m.excess_return_vs_ew = tr - ewr
        dr = pd.Series([r.daily_return for r in drs])
        m.volatility = dr.std() * np.sqrt(TRADING_DAYS)
        cum = pd.Series([1 + r.cumulative_return for r in drs])
        rmax = cum.cummax(); dd = (cum - rmax) / rmax; m.max_drawdown = dd.min()
        if m.max_drawdown < 0:
            ds = dd[dd < 0].index[0] if dd[dd < 0].any() else 0
            de = dd.idxmin(); rec = cum.iloc[de:] >= rmax.iloc[de]
            m.max_drawdown_duration = int(rec[rec].index.min() - ds if rec.any() else n)
        rfd = self.rf / TRADING_DAYS; ds = dr[dr < rfd]
        m.downside_volatility = ds.std() * np.sqrt(TRADING_DAYS) if len(ds) > 0 else 0.0
        sd = dr.std(); dsd = ds.std() if len(ds) > 0 else 1.0
        if sd > 0: m.sharpe_ratio = (dr.mean() - rfd) / sd * np.sqrt(TRADING_DAYS)
        if dsd > 0: m.sortino_ratio = (dr.mean() - rfd) / dsd * np.sqrt(TRADING_DAYS)
        if abs(m.max_drawdown) > 0: m.calmar_ratio = m.annualized_return / abs(m.max_drawdown)
        m.total_trades = len(trs); m.total_trade_cost = sum(t.commission + t.tax for t in trs)
        m.cost_ratio = m.total_trade_cost / ic if ic > 0 else 0.0
        st = set()
        for t in trs:
            if t.trade_type == "买入": st.add(t.date)
            elif "调仓买入" in t.trade_type and "第1天" in t.reason: st.add(t.date)
        m.switch_count = len(st)
        sl = [t for t in trs if "卖出" in t.trade_type]
        if sl: m.avg_hold_days = sum(t.days_held for t in sl) / len(sl)
        m.win_rate = (dr > 0).sum() / len(dr)
        pt = [t for t in trs if t.trade_type in ("卖出","止损卖出","止盈卖出","调仓卖出","风控卖出","极端回撤清仓","虚拟卖出")]
        if pt:
            pr = [t for t in pt if t.profit > 0]; lo = [t for t in pt if t.profit < 0]
            if pt: m.trade_win_rate = len(pr) / len(pt)
            tp = sum(t.profit for t in pr); tl = abs(sum(t.profit for t in lo))
            if tl > 0: m.profit_factor = tp / tl
        return m
