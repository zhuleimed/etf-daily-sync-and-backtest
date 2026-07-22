"""
黄金避险轮动策略 - 回测引擎

双模式状态机：
  MODE_NORMAL → 动量轮动（同 momentum_rotation）
  MODE_GOLD   → 持有黄金避险

状态转换：
  NORMAL → GOLD: panic_score > PANIC_THRESHOLD
  GOLD → NORMAL: 恐慌解除(avg_5d >= -1%) 或 黄金止损 或 到期强制退出

时序规则（消除 look-ahead bias）：
  信号用 close[idx-1]，执行用 open[idx]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .momentum_signals import compute_momentum_signals, rank_etfs_by_momentum
from .panic_signals import (
    compute_panic_score, get_broad_avg_5d_return,
    init_panic_history, compute_panic_hard,
    check_bull_market_from_data,
)
from .risk import check_gold_stop_loss, check_extreme_drawdown
from .cost import compute_total_friction_cost


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class DailyRecord:
    """单日快照。"""
    date: str = ""
    hold_symbol: str = ""
    hold_name: str = ""
    hold_shares: int = 0
    hold_close: float = 0.0
    cash: float = 0.0
    stock_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    action: str = ""
    mode: str = "normal"
    panic_score: float = 0.0
    target_etf: str = ""
    reason: str = ""


@dataclass
class TradeRecord:
    """单笔交易记录（兼容 momentum_rotation 的 metrics 计算）。"""
    date: str = ""
    symbol: str = ""
    trade_type: str = ""        # "买入" / "卖出" 等
    price: float = 0.0
    shares: int = 0
    amount: float = 0.0
    commission: float = 0.0
    tax: float = 0.0            # ETF免印花税，始终为0
    pnl: float = 0.0            # 盈亏（也称profit）
    profit: float = 0.0         # 盈亏（别名，兼容metrics）
    days_held: int = 0          # 持仓天数
    reason: str = ""


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """黄金避险轮动回测引擎。"""

    def __init__(
        self,
        initial_capital: float = cfg.INITIAL_CAPITAL,
        momentum_window: int = cfg.MOMENTUM_WINDOW,
    ):
        self.initial_capital = initial_capital
        self.momentum_window = momentum_window

        # 资金与持仓
        self.cash = initial_capital
        self.positions: Dict[str, int] = {}     # {symbol: shares}
        self.avg_cost: Dict[str, float] = {}    # {symbol: avg_cost}

        # 引擎状态
        self._mode = "normal"           # "normal" | "gold"
        self._days_in_gold = 0
        self._days_since_switch = 999
        self._peak_value = initial_capital

        # 恐慌历史（滚动Z-score用，跨日期持久化）
        self._panic_history = init_panic_history()

        # 记录
        self.daily_records: List[DailyRecord] = []
        self.trade_records: List[TradeRecord] = []

        # 数据（load_data 时填充）
        self.etf_data: Dict[str, pd.DataFrame] = {}
        self.dates: pd.DatetimeIndex = pd.DatetimeIndex([])

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def load_data(
        self,
        start_date: str = cfg.START_DATE,
        end_date: str = cfg.END_DATE,
    ):
        """加载ETF数据。"""
        from .data import load_all_etf_data
        self.etf_data, self.dates = load_all_etf_data(
            start_date=start_date,
            end_date=end_date,
        )
        print(f"加载 {len(self.etf_data)} 只ETF，{len(self.dates)} 个交易日")

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self):
        """执行回测主循环。"""
        if not self.etf_data or len(self.dates) == 0:
            raise ValueError("请先调用 load_data() 加载数据")

        for idx in range(len(self.dates)):
            date_str = self.dates[idx].strftime("%Y-%m-%d")

            # --- 每日重置 ---
            self._days_since_switch += 1

            # --- 构建当日行情字典 ---
            today_data = self._build_today_data(idx)

            # --- Step 1: 检查极端回撤 ---
            total_val = self._calc_total_value(today_data)
            if total_val > self._peak_value:
                self._peak_value = total_val

            triggered, reason = check_extreme_drawdown(self._peak_value, total_val)
            if triggered and self._has_position():
                self._sell_all(idx, today_data, reason)
                self._record_day(idx, today_data, "risk_exit", "", 0.0, reason)
                continue

            # --- Step 2: 根据模式执行 ---
            if self._mode in ("gold", "cash"):
                self._handle_gold_mode(idx, today_data, date_str)
            else:
                self._handle_normal_mode(idx, today_data)

            # --- Step 3: 记录当日 ---
            self._record_day(idx, today_data, "", "", 0.0, "")

    # ------------------------------------------------------------------
    # 恐慌/现金模式处理
    # ------------------------------------------------------------------

    def _get_safe_haven_target(self, idx: int, panic_desc: str) -> tuple:
        """
        根据 SAFE_HAVEN 配置确定恐慌时的避险去向。

        Returns
        -------
        (target_symbol, mode_name)
        target_symbol 为 None 表示去现金。
        """
        if cfg.SAFE_HAVEN == "cash":
            return None, "现金"

        if cfg.SAFE_HAVEN == "smart":
            # 智能选择：黄金动量>0则去黄金，否则去现金
            gold_sym = cfg.GOLD_SYMBOL
            if gold_sym in self.etf_data and idx >= 5:
                gold_ret_5d = float(self.etf_data[gold_sym].loc[idx, "ret_5d"])
                if not pd.isna(gold_ret_5d) and gold_ret_5d > 0:
                    return gold_sym, f"黄金(gold_5d={gold_ret_5d*100:.1f}%)"
            return None, f"现金(gold_5d≤0)"

        # "gold" — 默认去黄金
        return cfg.GOLD_SYMBOL, "黄金"

    def _handle_gold_mode(
        self, idx: int, today_data: dict, date_str: str
    ):
        """处理避险持仓模式（黄金或现金）。"""
        gold_sym = cfg.GOLD_SYMBOL
        self._days_in_gold += 1

        # 现金模式：没有持仓，只需等待恐慌解除
        if self._mode == "cash":
            # 最小持有期
            if self._days_in_gold < cfg.MIN_GOLD_HOLD:
                return
            # 检查恐慌解除
            avg_5d = get_broad_avg_5d_return(self.etf_data, idx)
            if avg_5d >= cfg.PANIC_EXIT_THRESHOLD or self._days_in_gold >= cfg.GOLD_MAX_HOLD:
                exit_reason = f"恐慌解除(avg_5d={avg_5d*100:.1f}%)" if avg_5d >= cfg.PANIC_EXIT_THRESHOLD else f"现金避险到期({cfg.GOLD_MAX_HOLD}天)"
                self._mode = "normal"
                self._days_in_gold = 0
                self._days_since_switch = 0
                self._record_day(idx, today_data, "panic_exit", "", 0.0, exit_reason)
            return

        # 黄金模式：检查止损和退出条件
        if gold_sym in self.positions and self.positions.get(gold_sym, 0) > 0:
            # 黄金止损检查
            if gold_sym in self.etf_data and idx >= 5:
                gold_ret_5d = self.etf_data[gold_sym].loc[idx, "ret_5d"]
                if not pd.isna(gold_ret_5d):
                    triggered, reason = check_gold_stop_loss(gold_ret_5d)
                    if triggered:
                        self._sell_all(idx, today_data, reason)
                        self._mode = "normal"
                        self._days_in_gold = 0
                        self._days_since_switch = 0
                        self._record_day(
                            idx, today_data, "gold_stop_loss", "", 0.0, reason
                        )
                        return

        # 最小持仓期
        if self._days_in_gold < cfg.MIN_GOLD_HOLD:
            return

        # 检查恐慌解除
        avg_5d = get_broad_avg_5d_return(self.etf_data, idx)
        panic_exited = avg_5d >= cfg.PANIC_EXIT_THRESHOLD
        force_exit = self._days_in_gold >= cfg.GOLD_MAX_HOLD

        if panic_exited or force_exit:
            exit_reason = (
                f"恐慌解除(avg_5d={avg_5d*100:.1f}%)"
                if panic_exited
                else f"黄金最长持仓到期({cfg.GOLD_MAX_HOLD}天)"
            )
            self._sell_all(idx, today_data, exit_reason)
            self._mode = "normal"
            self._days_in_gold = 0
            self._days_since_switch = 0
            self._record_day(
                idx, today_data, "panic_exit", "", 0.0, exit_reason
            )
            return

    # ------------------------------------------------------------------
    # 正常模式处理（动量轮动）
    # ------------------------------------------------------------------

    def _handle_normal_mode(
        self, idx: int, today_data: dict
    ):
        """处理正常动量轮动模式。"""
        # 1. 计算恐慌信号（信号用 idx-1 避免 look-ahead）
        signal_idx = max(1, idx - 1)

        # 1a. 牛市过滤器：牛市时不触发恐慌
        if cfg.USE_BULL_FILTER:
            is_bull = check_bull_market_from_data(
                self.etf_data, signal_idx, cfg.BULL_FILTER_MA
            )
            if is_bull:
                # 牛市 → 跳过恐慌检测，直接动量轮动
                self._do_momentum_rotation(idx, today_data, signal_idx)
                return

        # 1b. 恐慌检测（硬阈值或Z-score）
        if cfg.PANIC_MODE == "hard":
            panic_result = compute_panic_hard(
                self.etf_data, signal_idx,
                dd_threshold=cfg.PANIC_DD_THRESHOLD,
                vol_threshold=cfg.PANIC_VOL_THRESHOLD,
                breadth_threshold=cfg.PANIC_BREADTH_THRESHOLD,
            )
        else:
            panic_result = compute_panic_score(
                self.etf_data, signal_idx, self._panic_history
            )

        # 2. 恐慌触发 → 切换到避险资产
        if panic_result["is_panic"]:
            hold_sym = self._get_hold_symbol()
            panic_score = panic_result.get("panic_score", 0)
            panic_desc = f"max_dd={panic_result['max_dd']*100:.1f}%"
            if cfg.PANIC_MODE == "hard":
                panic_desc = (f"dd={panic_result['max_dd']*100:.1f}% "
                              f"vol={panic_result['avg_vol_ratio']:.2f} "
                              f"brd={panic_result['breadth']*100:.0f}%")

            # 确定避险去向
            haven_target, haven_mode = self._get_safe_haven_target(idx, panic_desc)

            if hold_sym:
                self._sell_all(idx, today_data, f"恐慌触发({panic_desc})→{haven_mode}")

            if haven_target:
                self._buy(idx, today_data, haven_target, f"恐慌买入{haven_mode}({panic_desc})")
                self._mode = "gold"
                self._days_in_gold = 0
            else:
                # 现金模式：不买入，标记为避险状态
                self._mode = "cash"
                self._days_in_gold = 0

            self._days_since_switch = 0
            self._record_day(
                idx, today_data, "panic_entry",
                haven_target or "CASH",
                panic_score,
                f"恐慌触发→{haven_mode} {panic_desc}"
            )
            return

        # 3. 正常动量轮动
        self._do_momentum_rotation(idx, today_data, signal_idx)

    def _do_momentum_rotation(self, idx: int, today_data: dict, signal_idx: int):
        """执行标准动量轮动逻辑（从 _handle_normal_mode 中提取）。"""
        momentum = compute_momentum_signals(
            self.etf_data, signal_idx, self.momentum_window
        )
        # 只对宽基排名（排除黄金）
        broad_momentum = momentum[momentum.index.isin(cfg.BROAD_SYMBOLS)]
        ranking = rank_etfs_by_momentum(broad_momentum)

        target_etf = ranking.get(1) if len(ranking) > 0 else None
        hold_sym = self._get_hold_symbol()

        # 无持仓 → 买入最强宽基
        if not hold_sym:
            if target_etf and not pd.isna(momentum.get(target_etf, np.nan)):
                target_mom = momentum[target_etf]
                if target_mom > 0:
                    self._buy(
                        idx, today_data, target_etf,
                        f"动量开仓(动量={target_mom*100:.1f}%)"
                    )
                    self._days_since_switch = 0
            return

        # 已有持仓 → 检查是否需要切换
        if target_etf == hold_sym:
            return  # 当前就是最强的，持有

        # 检查 MIN_HOLD_DAYS
        if self._days_since_switch < cfg.MIN_HOLD_DAYS:
            return

        # 检查切换条件
        target_mom = momentum.get(target_etf, np.nan)
        hold_mom = momentum.get(hold_sym, np.nan)
        if pd.isna(target_mom) or pd.isna(hold_mom):
            return

        excess = target_mom - hold_mom

        # 短期动量确认
        if cfg.SHORT_TERM_MOMENTUM_CHECK and idx >= 5:
            check_idx = idx - 1
            try:
                target_5d = self.etf_data[target_etf].loc[check_idx, "ret_5d"]
                if not pd.isna(target_5d) and target_5d <= -0.005:
                    return  # 目标短期动量太弱
            except (KeyError, IndexError):
                pass

        # 摩擦成本
        sell_price = today_data.get(hold_sym, {}).get("open", 0)
        buy_price = today_data.get(target_etf, {}).get("open", 0)
        sell_amount = self.positions.get(hold_sym, 0) * sell_price
        buy_amount = sell_amount

        friction, _ = compute_total_friction_cost(
            hold_sym, target_etf, sell_amount, buy_amount,
            sell_price, buy_price, self.etf_data, idx,
        )
        total_trade = abs(sell_amount) + abs(buy_amount)
        friction_ratio = friction / total_trade if total_trade > 0 else 0.01
        threshold = max(friction_ratio, cfg.MIN_SWITCH_CONVICTION)

        if excess > threshold:
            # 执行切换
            self._sell_all(idx, today_data, f"动量切换卖出→{target_etf}")
            self._buy(idx, today_data, target_etf,
                      f"动量切换买入(excess={excess*100:.2f}%>threshold={threshold*100:.2f}%)")
            self._days_since_switch = 0

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def _buy(self, idx: int, today_data: dict, symbol: str, reason: str = ""):
        """买入ETF（以开盘价+滑点执行）。"""
        if symbol not in today_data:
            return

        open_price = today_data[symbol].get("open", 0)
        if open_price <= 0:
            return

        # 买入价 = open + 滑点
        buy_price = open_price * (1 + cfg.SLIPPAGE)

        # 可用资金（留2%防佣金不足）
        available = self.cash * 0.98

        # 取整100股
        shares = int(available / buy_price / 100) * 100
        if shares < 100:
            return

        cost = shares * buy_price
        commission = cost * cfg.COMMISSION_RATE
        total_cost = cost + commission

        if total_cost > self.cash:
            return

        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + shares
        self.avg_cost[symbol] = total_cost / shares

        # 记录交易
        self.trade_records.append(TradeRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"),
            symbol=symbol,
            trade_type="买入",
            price=buy_price,
            shares=shares,
            amount=cost,
            commission=commission,
            tax=0.0,
            pnl=0.0,
            days_held=0,
            reason=reason,
        ))

    def _sell_all(self, idx: int, today_data: dict, reason: str = ""):
        """卖出全部持仓。"""
        for sym in list(self.positions.keys()):
            if self.positions[sym] <= 0:
                continue
            self._sell(idx, today_data, sym, reason)

    def _sell(self, idx: int, today_data: dict, symbol: str, reason: str = ""):
        """卖出指定ETF全部持仓。"""
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return
        if symbol not in today_data:
            return

        open_price = today_data[symbol].get("open", 0)
        if open_price <= 0:
            return

        # 卖出价 = open - 滑点
        sell_price = open_price * (1 - cfg.SLIPPAGE)

        shares = self.positions[symbol]
        revenue = shares * sell_price
        commission = revenue * cfg.COMMISSION_RATE
        net_revenue = revenue - commission

        # 计算盈亏
        total_cost = self.avg_cost.get(symbol, 0) * shares
        pnl = net_revenue - total_cost

        self.cash += net_revenue

        # 记录交易
        self.trade_records.append(TradeRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"),
            symbol=symbol,
            trade_type="卖出",
            price=sell_price,
            shares=shares,
            amount=revenue,
            commission=commission,
            tax=0.0,
            pnl=pnl,
            profit=pnl,
            days_held=0,
            reason=reason,
        ))

        del self.positions[symbol]
        if symbol in self.avg_cost:
            del self.avg_cost[symbol]

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _build_today_data(self, idx: int) -> dict:
        """构建当日行情字典 {symbol: {open, high, low, close, volume}}。"""
        today = {}
        for sym, df in self.etf_data.items():
            if idx >= len(df):
                continue
            row = df.iloc[idx]
            today[sym] = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        return today

    def _get_hold_symbol(self) -> Optional[str]:
        """获取当前持仓的ETF代码（单ETF全仓模式）。"""
        for sym, shares in self.positions.items():
            if shares > 0:
                return sym
        return None

    def _has_position(self) -> bool:
        return self._get_hold_symbol() is not None

    def _calc_total_value(self, today_data: dict) -> float:
        """计算当前总资产（现金 + 持仓市值）。"""
        stock_val = 0.0
        for sym, shares in self.positions.items():
            if shares > 0 and sym in today_data:
                stock_val += shares * today_data[sym]["close"]
        return self.cash + stock_val

    def _record_day(
        self, idx: int, today_data: dict,
        action: str, target: str, panic_score: float, reason: str,
    ):
        """记录每日快照。"""
        hold_sym = self._get_hold_symbol()
        stock_val = 0.0
        hold_shares = 0
        hold_close = 0.0
        hold_name = ""

        if hold_sym:
            hold_shares = self.positions.get(hold_sym, 0)
            if hold_sym in today_data:
                hold_close = today_data[hold_sym]["close"]
                stock_val = hold_shares * hold_close
            hold_name = cfg.ETF_POOL.get(hold_sym, hold_sym)

        total_val = self.cash + stock_val
        prev_total = self.initial_capital
        if self.daily_records:
            prev_total = self.daily_records[-1].total_value

        daily_ret = (total_val / prev_total - 1) if prev_total > 0 else 0.0
        cum_ret = total_val / self.initial_capital - 1

        mode_str = f"{self._mode}(d{self._days_in_gold})" if self._mode == "gold" else "normal"

        record = DailyRecord(
            date=self.dates[idx].strftime("%Y-%m-%d"),
            hold_symbol=hold_sym or "",
            hold_name=hold_name,
            hold_shares=hold_shares,
            hold_close=hold_close,
            cash=round(self.cash, 2),
            stock_value=round(stock_val, 2),
            total_value=round(total_val, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            action=action or ("holding" if hold_sym else "cash"),
            mode=mode_str,
            panic_score=round(panic_score, 3),
            target_etf=target,
            reason=reason,
        )
        self.daily_records.append(record)

    # ------------------------------------------------------------------
    # 结果导出
    # ------------------------------------------------------------------

    def get_daily_df(self) -> pd.DataFrame:
        """将每日记录导出为 DataFrame。"""
        return pd.DataFrame([r.__dict__ for r in self.daily_records])

    def get_trade_df(self) -> pd.DataFrame:
        """将交易记录导出为 DataFrame。"""
        return pd.DataFrame([t.__dict__ for t in self.trade_records])
