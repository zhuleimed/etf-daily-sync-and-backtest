"""
配对交易回测引擎

每对 ETF 独立运行：
  1. 计算比价 price_a / price_b
  2. z-score = (当前比价 - 均值) / 标准差
  3. |z| > 开仓阈值 → 多空同时开 |z| < 平仓阈值 → 两边平仓

信号用 close[T-1]，执行用 open[T]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    PAIRS, INITIAL_CAPITAL, CAPITAL_PER_PAIR,
    ZSCORE_PERIOD, ZSCORE_OPEN, ZSCORE_CLOSE, ZSCORE_STOP,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, OUTPUT_DIR,
)
from strategies.momentum_rotation.data import load_all_etf_data


@dataclass
class PairPosition:
    """单对持仓状态。"""
    pair_id: str = ""
    active: bool = False
    direction: str = ""           # "ab"=多a空b, "ba"=多b空a
    long_symbol: str = ""
    long_shares: int = 0
    long_entry: float = 0.0       # 开仓价（含滑点）
    short_symbol: str = ""
    short_shares: int = 0         # 做空股数（正值）
    short_entry: float = 0.0      # 开仓价（含滑点）
    entry_spread: float = 0.0     # 开仓时的比价
    entry_zscore: float = 0.0
    highest_pnl: float = 0.0      # 最高浮动盈亏
    open_date: str = ""


@dataclass
class DailyRecord:
    date: str = ""
    pairs_active: int = 0
    cash: float = 0.0
    long_value: float = 0.0       # 多头持仓市值
    short_pnl: float = 0.0        # 空头累计已平仓盈亏
    unrealized_short: float = 0.0 # 空头浮动盈亏
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    actions: str = ""


class PairTradingEngine:
    """配对交易回测引擎。"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.etf_data: Dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])

        # 每对持仓
        self.positions: Dict[str, PairPosition] = {}
        for p in PAIRS:
            pid = p["name"]
            self.positions[pid] = PairPosition(pair_id=pid)

        # 结果
        self.daily_records: List[DailyRecord] = []
        self.total_trade_cost: float = 0.0
        self.total_short_pnl: float = 0.0  # 空头累计盈亏

    # ── 数据加载 ──

    def load_data(self, start_date: str = "2024-01-01",
                  end_date: str = ""):
        """加载所有 ETF 数据。"""
        # 收集所有用到的 ETF 代码
        symbols = set()
        for p in PAIRS:
            symbols.add(p["a"])
            symbols.add(p["b"])
        self.etf_data, self.dates = load_all_etf_data(
            symbols=list(symbols), start_date=start_date,
            end_date=end_date, db_path=DB_PATH,
            momentum_window=ZSCORE_PERIOD,
        )

    # ── 主流程 ──

    def run(self):
        """执行完整回测。"""
        n = len(self.dates)
        print(f"  配对交易回测：{self.dates[0].date()} → {self.dates[-1].date()}")
        print(f"  配对数量：{len(PAIRS)} 对，每对资金 {CAPITAL_PER_PAIR} 元")
        print(f"  开仓阈值：|z| > {ZSCORE_OPEN}  平仓：|z| < {ZSCORE_CLOSE}  止损：|z| > {ZSCORE_STOP}")
        print(f"  {'=' * 40}")

        for idx in range(n):
            today_str = str(self.dates[idx].date())
            today_data = {}
            for sym, df in self.etf_data.items():
                if idx < len(df):
                    today_data[sym] = df.iloc[idx]

            signal_idx = max(1, idx - 1)

            # 处理每对
            actions = []
            for pair_cfg in PAIRS:
                pid = pair_cfg["name"]
                sym_a = pair_cfg["a"]
                sym_b = pair_cfg["b"]
                pos = self.positions[pid]

                act = self._process_pair(pair_cfg, pos, idx, signal_idx, today_data, today_str)
                if act:
                    actions.append(act)

            # 记录
            self._record_day(today_str, actions, idx)

        print(f"  回测完成 ✓")

    # ── 单对处理 ──

    def _process_pair(self, pair_cfg: dict, pos: PairPosition,
                      idx: int, signal_idx: int,
                      today_data: dict, today_str: str) -> str:
        """处理一对ETF的信号与执行。"""
        sym_a = pair_cfg["a"]
        sym_b = pair_cfg["b"]

        # 检查数据完整性
        df_a = self.etf_data.get(sym_a)
        df_b = self.etf_data.get(sym_b)
        if df_a is None or df_b is None or signal_idx < ZSCORE_PERIOD:
            return ""
        if signal_idx >= len(df_a) or signal_idx >= len(df_b):
            return ""

        # 计算比价（用 close[T-1] 计算信号）
        close_a = df_a.iloc[signal_idx]["close"]
        close_b = df_b.iloc[signal_idx]["close"]
        if close_b == 0:
            return ""
        spread = close_a / close_b

        # 计算 z-score
        spreads = df_a.iloc[signal_idx - ZSCORE_PERIOD + 1: signal_idx + 1]["close"].values / \
                  df_b.iloc[signal_idx - ZSCORE_PERIOD + 1: signal_idx + 1]["close"].values
        mean_s = np.mean(spreads)
        std_s = np.std(spreads)
        z = (spread - mean_s) / std_s if std_s > 0 else 0

        # ── 已有持仓的处理 ──
        if pos.active:
            open_pnl = self._calc_pair_pnl(pos, today_data)
            pos.highest_pnl = max(pos.highest_pnl, open_pnl)

            # 检查止损 |z| > ZSCORE_STOP
            if abs(z) > ZSCORE_STOP:
                self._close_pair(pos, idx, today_data, today_str, reason="止损")
                return f"{pair_cfg['name']} 止损平仓(z={z:.2f})"

            # 检查获利了结 |z| < ZSCORE_CLOSE
            if abs(z) < ZSCORE_CLOSE:
                self._close_pair(pos, idx, today_data, today_str, reason="获利了结")
                return f"{pair_cfg['name']} 平仓(z={z:.2f})"

            return ""  # 继续持有

        # ── 无持仓：检查开仓信号 ──
        if z > ZSCORE_OPEN:
            # A 比 B 贵 → 空A多B
            self._open_pair(pos, sym_a, sym_b, idx, today_data, today_str, z, spread, "ab")
            return f"{pair_cfg['name']} 开仓空{str(sym_a)[:4]}多{sym_b[:4]}(z={z:.2f})"

        if z < -ZSCORE_OPEN:
            # B 比 A 贵 → 多A空B
            self._open_pair(pos, sym_b, sym_a, idx, today_data, today_str, z, spread, "ba")
            return f"{pair_cfg['name']} 开仓多{sym_a[:4]}空{sym_b[:4]}(z={z:.2f})"

        return ""

    # ── 开仓 ──

    def _open_pair(self, pos: PairPosition, long_sym: str, short_sym: str,
                   idx: int, today_data: dict, today_str: str,
                   zscore: float, spread: float, direction: str):
        """同时开多空两腿，用 open[T] 价格。"""
        long_price = today_data[long_sym]["open"] * (1 + SLIPPAGE)
        short_price = today_data[short_sym]["open"] * (1 - SLIPPAGE)

        # 多头腿：真实买入
        long_amount = CAPITAL_PER_PAIR / 2
        long_shares = int(long_amount // long_price // 100) * 100
        if long_shares <= 0:
            return
        long_cost = long_shares * long_price
        commission_long = max(long_cost * COMMISSION_RATE, 0.0)
        total_long = long_cost + commission_long

        if total_long > self.cash:
            long_shares = int(self.cash // long_price // 100) * 100
            if long_shares <= 0:
                return
            long_cost = long_shares * long_price
            commission_long = max(long_cost * COMMISSION_RATE, 0.0)
            total_long = long_cost + commission_long

        self.cash -= total_long
        self.total_trade_cost += commission_long

        # 空头腿：合成记录（不开仓不占现金）
        short_amount = CAPITAL_PER_PAIR / 2
        short_shares = int(short_amount // short_price // 100) * 100

        pos.active = True
        pos.direction = direction
        pos.long_symbol = long_sym
        pos.long_shares = long_shares
        pos.long_entry = long_price
        pos.short_symbol = short_sym
        pos.short_shares = short_shares
        pos.short_entry = short_price
        pos.entry_spread = spread
        pos.entry_zscore = zscore
        pos.highest_pnl = 0.0
        pos.open_date = today_str

    # ── 平仓 ──

    def _close_pair(self, pos: PairPosition, idx: int,
                    today_data: dict, today_str: str, reason: str = ""):
        """同时平多空两腿，用 open[T] 价格。"""
        if not pos.active:
            return

        # 多头平仓
        if pos.long_shares > 0 and pos.long_symbol in today_data:
            sell_price = today_data[pos.long_symbol]["open"] * (1 - SLIPPAGE)
            revenue = pos.long_shares * sell_price
            commission = max(revenue * COMMISSION_RATE, 0.0)
            self.cash += revenue - commission
            self.total_trade_cost += commission

        # 空头平仓（买入归还）
        short_pnl = 0
        if pos.short_shares > 0 and pos.short_symbol in today_data:
            buy_back_price = today_data[pos.short_symbol]["open"] * (1 + SLIPPAGE)
            cost_to_close = pos.short_shares * buy_back_price
            commission_short = max(cost_to_close * COMMISSION_RATE, 0.0)
            # 空头盈亏 = 开仓收入 - 平仓支出
            short_revenue = pos.short_shares * pos.short_entry
            short_pnl = short_revenue - cost_to_close - commission_short
            self.cash += short_pnl
            self.total_trade_cost += commission_short
            self.total_short_pnl += short_pnl

        # 重置持仓
        self._reset_position(pos)

    def _reset_position(self, pos: PairPosition):
        pos.active = False
        pos.direction = ""
        pos.long_symbol = ""
        pos.long_shares = 0
        pos.long_entry = 0.0
        pos.short_symbol = ""
        pos.short_shares = 0
        pos.short_entry = 0.0
        pos.entry_spread = 0.0
        pos.entry_zscore = 0.0
        pos.highest_pnl = 0.0
        pos.open_date = ""

    # ── 持仓估值 ──

    def _calc_pair_pnl(self, pos: PairPosition, today_data: dict) -> float:
        """计算当前持仓的浮动盈亏。"""
        pnl = 0.0
        if pos.long_symbol in today_data:
            current_long = today_data[pos.long_symbol]["close"]
            pnl += (current_long - pos.long_entry) * pos.long_shares
        if pos.short_symbol in today_data:
            current_short = today_data[pos.short_symbol]["close"]
            pnl += (pos.short_entry - current_short) * pos.short_shares
        return pnl

    # ── 记录 ──

    def _record_day(self, today_str: str, actions: List[str], idx: int = 0):
        """记录当日账户状态（用当日 close 估值）。"""
        long_value = 0.0
        unrealized_short = 0.0
        pairs_active = 0

        for pos in self.positions.values():
            if not pos.active:
                continue
            pairs_active += 1
            if pos.long_symbol and pos.long_shares > 0:
                df = self.etf_data.get(pos.long_symbol)
                if df is not None and idx < len(df):
                    long_value += pos.long_shares * df.iloc[idx]["close"]
            if pos.short_symbol and pos.short_shares > 0:
                df = self.etf_data.get(pos.short_symbol)
                if df is not None and idx < len(df):
                    unrealized_short += (pos.short_entry - df.iloc[idx]["close"]) * pos.short_shares

        total_value = self.cash + long_value + unrealized_short

        if not self.daily_records:
            daily_ret = 0.0
        else:
            prev = self.daily_records[-1].total_value
            daily_ret = (total_value - prev) / prev if prev > 0 else 0.0

        cum_ret = total_value / self.initial_capital - 1
        self.daily_records.append(DailyRecord(
            date=today_str, pairs_active=pairs_active,
            cash=round(self.cash, 2), long_value=round(long_value, 2),
            short_pnl=round(self.total_short_pnl, 2),
            unrealized_short=round(unrealized_short, 2),
            total_value=round(total_value, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            actions="; ".join(actions) if actions else "",
        ))


    def get_daily_df(self) -> pd.DataFrame:
        """生成净值日报表。"""
        rows = []
        for i, rec in enumerate(self.daily_records):
            # 重新估值
            long_val = 0.0
            short_unreal = 0.0
            for pos in self.positions.values():
                if not pos.active:
                    continue
                date_str = rec.date
                # 找对应日期的价格
                for sym, shares, entry, is_long in [
                    (pos.long_symbol, pos.long_shares, pos.long_entry, True),
                    (pos.short_symbol, pos.short_shares, pos.short_entry, False),
                ]:
                    if sym and shares > 0:
                        df = self.etf_data.get(sym)
                        if df is not None:
                            match = df[df["date"] == date_str]
                            if not match.empty:
                                price = match.iloc[0]["close"]
                                if is_long:
                                    long_val += shares * price
                                else:
                                    short_unreal += (entry - price) * shares
            total = rec.cash + long_val + short_unreal + rec.short_pnl
            rows.append({
                "date": rec.date, "pairs_active": rec.pairs_active,
                "cash": rec.cash, "long_value": round(long_val, 2),
                "short_pnl_cumulative": rec.short_pnl,
                "short_unrealized": round(short_unreal, 2),
                "total_value": round(total, 2),
                "daily_return": rec.daily_return,
                "cumulative_return": rec.cumulative_return,
                "actions": rec.actions,
            })
        return pd.DataFrame(rows)

    def get_trade_df(self) -> pd.DataFrame:
        """生成交易明细（从 positions 变更记录）。"""
        # 简化版：返回空 df（详细交易记录待后续扩展）
        return pd.DataFrame()
