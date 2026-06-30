"""
回测引擎 — MACD 趋势轮动

与 ADX 引擎结构一致，但使用 MACD 评分：
  1. 数据加载 → 2. 风控检查 → 3. MACD评分 → 4. 决策 → 5. 记录
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .config import (
    ETF_POOL, ETF_SYMBOLS, INITIAL_CAPITAL, COMMISSION_RATE, SLIPPAGE,
    TAX_RATE, ADJUSTMENT_DAYS, DB_PATH, RISK_MODE, MIN_HOLD_DAYS,
    SWITCH_CONVICTION_STD, MARKET_INDEX, MARKET_MA_PERIOD,
)
from .data import load_all_etf_data, load_index_data, compute_equal_weight_benchmark
from .momentum_signals import (
    compute_macd_scores, rank_etfs_by_macd, compute_macd_spread, judge_market_regime,
)
from .risk import RiskState, run_all_risk_checks


@dataclass
class DailyRecord:
    date: str = ""
    hold_symbol: str = ""
    hold_shares: int = 0
    hold_close: float = 0.0
    cash: float = 0.0
    stock_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    action: str = "hold"
    regime: str = "neutral"
    top_etf: str = ""
    macd_score: float = 0.0
    reason: str = ""


@dataclass
class BuyLot:
    date: str = ""; symbol: str = ""; shares: int = 0
    price: float = 0.0; total_cost: float = 0.0


@dataclass
class TradeRecord:
    date: str = ""; symbol: str = ""; trade_type: str = ""
    price: float = 0.0; shares: int = 0; amount: float = 0.0
    commission: float = 0.0; tax: float = 0.0; profit: float = 0.0
    days_held: int = 0; return_rate: float = 0.0; reason: str = ""


class BacktestEngine:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL, risk_mode: str = ""):
        self.initial_capital = initial_capital
        self.risk_mode = risk_mode or RISK_MODE
        self._days_since_last_switch = 999
        self.etf_data: dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.index_data: pd.DataFrame = pd.DataFrame()
        self.equal_weight_data: pd.DataFrame = pd.DataFrame()
        self.positions: dict[str, int] = {}
        self.cash: float = initial_capital
        self.open_buys: list[BuyLot] = []
        self.adjustment_from = ""; self.adjustment_to = ""
        self.adjustment_days_left = 0; self.adjustment_total_days = 0
        self.risk_state = RiskState()
        self.daily_records: list[DailyRecord] = []
        self.trade_records: list[TradeRecord] = []
        self.total_trade_cost: float = 0.0

    def load_data(self, start_date="2024-01-01", end_date="", db_path=DB_PATH) -> "BacktestEngine":
        self.etf_data, self.dates = load_all_etf_data(ETF_SYMBOLS, start_date, end_date, db_path)
        self.index_data = load_index_data(MARKET_INDEX, start_date, end_date, db_path)
        self.equal_weight_data = compute_equal_weight_benchmark(self.etf_data)
        return self

    def run(self) -> "BacktestEngine":
        n = len(self.dates)
        if n == 0: raise RuntimeError("无交易日数据")
        print(f"  开始回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  标的：{', '.join(ETF_POOL.get(s, s) for s in ETF_SYMBOLS)}")
        mode_names = {"A": "纯信号", "B": "风控全开", "C": "仅极端回撤"}
        print(f"  风控模式: {self.risk_mode} = {mode_names.get(self.risk_mode, self.risk_mode)}")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            has_position = bool(self.positions)
            hold_symbol = self._get_hold_symbol()

            # 风控
            if self.risk_mode != "A" and has_position and hold_symbol:
                hr = today_data[hold_symbol]
                tv = self._calc_total_value(today_data)
                self.risk_state.update_peak(hr["high"]); self.risk_state.update_peak_total_value(tv)
                ra, rr = run_all_risk_checks(self.risk_state, tv, has_position, hold_symbol,
                    hr["high"], hr["low"], hr["close"], hr["atr"], mode=self.risk_mode)
                if ra != "none":
                    self._execute_risk_exit(idx, today_data, ra, rr)
                    self._record_day(idx, today_data, action_override=ra); continue

            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            sig_idx = max(0, idx - 1)
            idx_aligned = None
            if not self.index_data.empty and sig_idx < len(self.index_data):
                idx_aligned = self.index_data

            scores = compute_macd_scores(self.etf_data, sig_idx)
            ranking = rank_etfs_by_macd(scores)
            spread = compute_macd_spread(scores)
            regime = judge_market_regime(idx_aligned, sig_idx, MARKET_MA_PERIOD)
            target = ranking.get(1) if len(ranking) > 0 else None
            ts = scores.get(target, np.nan) if target else np.nan

            if self.adjustment_days_left <= 0:
                self._make_decision(idx, today_data, hold_symbol, target, scores, spread, regime)

            self._record_day(idx, today_data, regime=regime["regime"],
                             top_etf=target or "", macd_score=round(ts, 4) if not pd.isna(ts) else 0)
            self._days_since_last_switch += 1

        self._close_remaining_positions()
        print(f"  回测完成 ✓"); return self

    def _get_hold_symbol(self) -> Optional[str]:
        if not self.positions: return None
        v = {k: v for k, v in self.positions.items() if v > 0}
        return max(v, key=v.get) if v else None

    def _calc_total_value(self, td: dict) -> float:
        return self.cash + sum(sh * td[s]["close"] for s, sh in self.positions.items() if sh > 0 and s in td)

    def _buy(self, sym: str, amt: float, idx: int, td: dict, tt="买入", reason="") -> int:
        px = td[sym]["open"] * (1 + SLIPPAGE)
        ms = int(amt // px // 100) * 100
        if ms <= 0: return 0
        cost = ms * px; comm = max(cost * COMMISSION_RATE, 0.0); tc = cost + comm
        if tc > self.cash:
            ms = int((self.cash * 0.99) // px // 100) * 100
            if ms <= 0: return 0
            cost = ms * px; comm = max(cost * COMMISSION_RATE, 0.0); tc = cost + comm
        self.cash -= tc
        self.positions[sym] = self.positions.get(sym, 0) + ms
        self.open_buys.append(BuyLot(date=str(td[sym]["date"]), symbol=sym, shares=ms, price=px, total_cost=tc))
        self.total_trade_cost += comm
        self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"), symbol=sym,
            trade_type=tt, price=px, shares=ms, amount=cost, commission=comm, tax=0.0, reason=reason))
        return ms

    def _sell(self, sym: str, idx: int, td: dict, tt="卖出", reason="") -> tuple[float, float]:
        sh = self.positions.get(sym, 0)
        if sh <= 0: return 0.0, 0.0
        px = td[sym]["open"] * (1 - SLIPPAGE)
        amt = sh * px; comm = max(amt * COMMISSION_RATE, 0.0); tax = amt * TAX_RATE
        net = amt - comm - tax
        rem = sh; tcb = 0.0
        for b in self.open_buys[:]:
            if b.symbol != sym or rem <= 0: continue
            used = min(b.shares, rem)
            tcb += b.total_cost * (used / b.shares); b.shares -= used; rem -= used
            if b.shares <= 0: self.open_buys.remove(b)
        profit = net - tcb
        self.cash += net; self.positions[sym] = 0
        if sym in self.positions: del self.positions[sym]
        self.total_trade_cost += comm
        self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"), symbol=sym,
            trade_type=tt, price=px, shares=sh, amount=amt, commission=comm, tax=tax, profit=profit, reason=reason))
        return profit, net

    def _start_adjustment(self, frm: str, to: str, nd=ADJUSTMENT_DAYS):
        self.adjustment_from = frm; self.adjustment_to = to
        self.adjustment_days_left = nd; self.adjustment_total_days = nd

    def _execute_adjustment_step(self, idx: int, td: dict):
        if self.adjustment_days_left <= 0: return
        dn = self.adjustment_total_days - self.adjustment_days_left + 1
        ss = self.adjustment_from; bs = self.adjustment_to
        s_sh = self.positions.get(ss, 0)
        if s_sh > 0:
            sq = max(s_sh // self.adjustment_days_left // 100 * 100, 100)
            if sq > s_sh: sq = s_sh
            p = td[ss]["open"] * (1 - SLIPPAGE); a = sq * p; c = max(a * COMMISSION_RATE, 0.0)
            self.cash += a - c; self.positions[ss] = s_sh - sq
            if self.positions[ss] <= 0: del self.positions[ss]
            self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"), symbol=ss,
                trade_type="调仓卖出", price=p, shares=sq, amount=a, commission=c, tax=0.0, reason=f"渐进调仓第{dn}天"))
        ba = self.cash / self.adjustment_days_left
        if ba > 100:
            p = td[bs]["open"] * (1 + SLIPPAGE); bq = int(ba // p // 100) * 100
            if bq > 0:
                c = bq * p; cm = max(c * COMMISSION_RATE, 0.0); self.cash -= c + cm
                self.positions[bs] = self.positions.get(bs, 0) + bq
                self.trade_records.append(TradeRecord(date=self.dates[idx].strftime("%Y-%m-%d"), symbol=bs,
                    trade_type="调仓买入", price=p, shares=bq, amount=c, commission=cm, tax=0.0, reason=f"渐进调仓第{dn}天"))
        self.adjustment_days_left -= 1

    def _execute_risk_exit(self, idx: int, td: dict, ra: str, rr: str):
        for s in list(self.positions.keys()):
            if self.positions[s] > 0:
                tt = "止损卖出" if "止损" in rr else "止盈卖出" if "止盈" in rr else "极端回撤清仓"
                self._sell(s, idx, td, tt, rr)

    def _make_decision(self, idx: int, td: dict, hs: Optional[str], target: Optional[str],
                        scores: pd.Series, spread: float, regime: dict):
        has_pos = hs is not None
        cur_s = scores.get(hs, np.nan) if hs else np.nan
        tgt_s = scores.get(target, np.nan) if target else np.nan

        # 当前持仓MACD转负 → 平仓
        if has_pos and not pd.isna(cur_s) and cur_s <= 0:
            self._sell(hs, idx, td, reason="MACD转负/信号消失，平仓")
            has_pos = False; hs = None

        if not has_pos:
            if target is not None and not pd.isna(tgt_s) and tgt_s > 0:
                if regime["regime"] == "bear" and tgt_s <= 0.5: return
                self._buy(target, self.cash * 0.98, idx, td, reason=f"开仓{target}（MACD={tgt_s:.4f}）")
                self.risk_state.on_open_position(td[target]["open"]); self._days_since_last_switch = 0
        else:
            if self._days_since_last_switch < MIN_HOLD_DAYS: return
            if target is None or target == hs: return
            if pd.isna(tgt_s) or pd.isna(cur_s) or tgt_s <= 0: return
            diff = tgt_s - cur_s
            if diff > max(spread * SWITCH_CONVICTION_STD, 0.1):
                if ADJUSTMENT_DAYS > 1 and hs:
                    self._start_adjustment(hs, target)
                else:
                    self._sell(hs, idx, td, reason=f"切换{hs}->{target}（MACD分差={diff:.4f}）")
                    self._buy(target, self.cash * 0.98, idx, td, reason=f"买入{target}（MACD={tgt_s:.4f}）")
                    self.risk_state.on_open_position(td[target]["open"]); self._days_since_last_switch = 0

    def _record_day(self, idx: int, td: dict, action_override="", regime="neutral", top_etf="", macd_score=0.0):
        tv = self._calc_total_value(td); sv = sum(sh * td[s]["close"] for s, sh in self.positions.items()
            if sh > 0 and s in td) if self.positions else 0.0
        hs = self._get_hold_symbol() or ""; hc = td[hs]["close"] if hs and hs in td else 0.0
        prev = self.daily_records[-1].total_value if self.daily_records else self.initial_capital
        dr = (tv - prev) / prev if prev > 0 else 0.0
        cr = tv / self.initial_capital - 1
        a = action_override or ("hold" if hs else "hold_cash")
        self.daily_records.append(DailyRecord(date=self.dates[idx].strftime("%Y-%m-%d"),
            hold_symbol=hs, hold_shares=self.positions.get(hs, 0) if hs else 0, hold_close=hc,
            cash=self.cash, stock_value=sv, total_value=tv, daily_return=dr,
            cumulative_return=cr, action=a, regime=regime, top_etf=top_etf, macd_score=macd_score))

    def _close_remaining_positions(self):
        for s in list(self.positions.keys()):
            if self.positions[s] > 0:
                ld = self.dates[-1]; lc = self.etf_data[s].iloc[-1]["close"]
                px = lc * (1 - SLIPPAGE); sh = self.positions[s]; amt = sh * px
                cm = max(amt * COMMISSION_RATE, 0.0); self.cash += amt - cm
                self.trade_records.append(TradeRecord(date=ld.strftime("%Y-%m-%d"), symbol=s,
                    trade_type="虚拟卖出", price=px, shares=sh, amount=amt, commission=cm, tax=0.0, profit=0.0, reason="期末平仓"))
                self.positions[s] = 0; del self.positions[s]

    def get_daily_df(self) -> pd.DataFrame:
        if not self.daily_records: return pd.DataFrame()
        rs = [{"date": r.date, "hold_symbol": r.hold_symbol, "hold_shares": r.hold_shares,
               "hold_close": r.hold_close, "cash": r.cash, "stock_value": r.stock_value,
               "total_value": r.total_value, "daily_return": r.daily_return,
               "cumulative_return": r.cumulative_return, "action": r.action, "regime": r.regime,
               "top_etf": r.top_etf, "macd_score": r.macd_score} for r in self.daily_records]
        df = pd.DataFrame(rs); df["benchmark_return"] = 0.0; df["excess_return"] = 0.0; return df

    def get_trade_df(self) -> pd.DataFrame:
        if not self.trade_records: return pd.DataFrame()
        return pd.DataFrame([{"date": t.date, "symbol": t.symbol, "trade_type": t.trade_type,
            "price": t.price, "shares": t.shares, "amount": t.amount, "commission": t.commission,
            "tax": t.tax, "profit": t.profit, "days_held": t.days_held, "return_rate": t.return_rate,
            "reason": t.reason} for t in self.trade_records])
