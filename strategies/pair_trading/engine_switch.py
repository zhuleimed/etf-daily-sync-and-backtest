"""
配对交易 → 纯多头轮动切换引擎

将原配对交易的多空对冲改为纯多头风格轮动（不依赖融券），
通过 z-score 判断风格切换时机。

三种切换模式：
  A — 单对最强信号 + 全仓切换
  B — 三对均分资金 + 各自轮动
  C — 多对按信号强度加权分配

信号体系（与原配对交易一致）：
  spread = log(price_a / price_b)
  z-score 60日滚动窗口
  对数比价 + 自适应阈值 + 成交量过滤
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    PAIRS, INITIAL_CAPITAL, CAPITAL_PER_PAIR,
    ZSCORE_PERIOD, ZSCORE_OPEN, ZSCORE_CLOSE, ZSCORE_STOP,
    ZSCORE_OPEN_GROWTH, ZSCORE_OPEN_VALUE,
    COMMISSION_RATE, SLIPPAGE, DB_PATH,
)

logger = logging.getLogger("pair_switch")


# ── 数据结构 ──

@dataclass
class SwitchRecord:
    """每日账户快照"""
    date: str = ""
    mode: str = ""
    hold_symbols: str = ""        # 当前持仓（逗号分隔）
    hold_pct: str = ""            # 各标占比
    cash: float = 0.0
    stock_value: float = 0.0
    total_value: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    action: str = ""
    z_scores: str = ""            # 三对 z-score 汇总
    signals: str = ""             # 信号摘要


# ── 信号计算（共享） ──

def compute_pair_signals(
    etf_data: Dict[str, pd.DataFrame],
    idx: int,
    growth_threshold: Optional[float] = None,
    value_threshold: Optional[float] = None,
) -> List[dict]:
    """计算所有配对的 z-score 及目标信号。

    Args:
        etf_data: {symbol: DataFrame} 行情数据
        idx: 当前日期在 DataFrame 中的行索引
        growth_threshold: z > +this → 买成长侧(默认=ZSCORE_OPEN)
        value_threshold: z < -this → 买价值侧(默认=ZSCORE_OPEN)

    Returns:
        list of dict: [
            {pair, z, target, strength, open_threshold, close_threshold, ...},
            ...
        ]
    """

    # 有效阈值（不对称支持）
    eff_growth_th = growth_threshold if growth_threshold is not None else ZSCORE_OPEN
    eff_value_th = value_threshold if value_threshold is not None else ZSCORE_OPEN

    results = []
    if idx < 1:
        return results

    signal_idx = idx - 1

    for pair in PAIRS:
        sym_a = pair["a"]
        sym_b = pair["b"]
        name = pair["name"]
        df_a = etf_data.get(sym_a)
        df_b = etf_data.get(sym_b)

        if df_a is None or df_b is None:
            continue
        if signal_idx < ZSCORE_PERIOD or signal_idx >= len(df_a) or signal_idx >= len(df_b):
            continue

        close_a = float(df_a.iloc[signal_idx]["close"])
        close_b = float(df_b.iloc[signal_idx]["close"])
        if close_b <= 0 or close_a <= 0:
            continue

        # 对数比价
        spread = np.log(close_a / close_b)

        # 60日滚动窗口的 z-score
        spreads_a = df_a.iloc[signal_idx - ZSCORE_PERIOD + 1: signal_idx + 1]["close"].values
        spreads_b = df_b.iloc[signal_idx - ZSCORE_PERIOD + 1: signal_idx + 1]["close"].values
        spreads_log = np.log(spreads_a / spreads_b)
        mean_s = np.mean(spreads_log)
        std_s = np.std(spreads_log)
        z = (spread - mean_s) / std_s if std_s > 1e-8 else 0.0

        # 自适应阈值
        vol_spread = float(np.std(spreads_log[-20:])) if len(spreads_log) >= 20 else float(std_s)
        adapt_factor = 1.0 + (vol_spread / max(float(std_s), 1e-8) - 1.0) * 0.5
        adapt_ratio = min(max(adapt_factor, 0.8), 1.5)
        growth_open = eff_growth_th * adapt_ratio   # z > +this -> 买成长
        value_open = eff_value_th * adapt_ratio      # z < -this -> 买价值
        open_threshold = ZSCORE_OPEN * adapt_ratio   # 兼容旧字段

        # 成交量过滤
        volume_ok = True
        for sym in [sym_a, sym_b]:
            df = etf_data.get(sym)
            if df is not None and signal_idx >= 20:
                vol_ma = df.iloc[signal_idx - 20: signal_idx + 1]["volume"].mean()
                if vol_ma > 0 and float(df.iloc[signal_idx]["volume"]) < vol_ma * 0.5:
                    volume_ok = False
                    break

        # 方向判定（不对称阈值）
        # z > +growth_open → 成长便宜 → 买成长侧（sym_b）
        # z < -value_open  → 价值便宜 → 买价值侧（sym_a）
        if z > growth_open:
            target = sym_b
            strength = abs(z)
            direction = f"买{sym_b[:4]}(成长)"
        elif z < -value_open:
            target = sym_a
            strength = abs(z)
            direction = f"买{sym_a[:4]}(价值)"
        else:
            target = None
            strength = 0.0
            direction = "无信号"

        results.append({
            "pair": name,
            "pair_cfg": pair,
            "z": z,
            "open_threshold": open_threshold,
            "growth_open": growth_open,
            "value_open": value_open,
            "close_threshold": ZSCORE_CLOSE,
            "stop_threshold": ZSCORE_STOP,
            "target": target,
            "strength": strength,
            "direction": direction,
            "volume_ok": volume_ok,
        })

    return results


# ════════════════════════════════════════════════════════════
# 模式 A：单对最强信号 + 全仓切换
# ════════════════════════════════════════════════════════════

def run_mode_a(
    etf_data: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[List[SwitchRecord], List[dict]]:
    """模式 A：三对中取 |z| 最大的信号，全仓切换。

    逻辑：
      - 没有持仓时，最强信号触发则全仓买入便宜方
      - 有持仓时，若最强信号指向不同 ETF → 切换
      - 持仓 ETF 的 |z| < close 且无其他强信号 → 平仓到现金
      - 止损：|z| > stop 时无条件清仓

    Returns:
        (records, trades)
    """
    cash = initial_capital
    hold_symbol = ""
    hold_shares = 0
    records: List[SwitchRecord] = []
    trades: List[dict] = []
    days_since_trade = 999

    # 不对称阈值
    growth_th = ZSCORE_OPEN_GROWTH if ZSCORE_OPEN_GROWTH is not None else ZSCORE_OPEN
    value_th = ZSCORE_OPEN_VALUE if ZSCORE_OPEN_VALUE is not None else ZSCORE_OPEN

    n = len(dates)
    for idx in range(n):
        today_str = str(dates[idx].date())

        # 构建今日行情
        today_data = {}
        for sym in etf_data:
            if idx < len(etf_data[sym]):
                row = etf_data[sym].iloc[idx]
                today_data[sym] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }

        # 计算信号（支持不对称阈值）
        signals = compute_pair_signals(etf_data, idx,
                                       growth_threshold=growth_th,
                                       value_threshold=value_th)

        # 信号摘要
        z_summary = "; ".join(
            f"{s['pair']} z={s['z']:.2f}"
            for s in signals
        ) if signals else "无数据"

        action = "hold"
        days_since_trade += 1

        # ── 止损检查（已有持仓） ──
        stop_triggered = False
        if hold_symbol and signals:
            for s in signals:
                # 找到包含持仓ETF的对
                if hold_symbol in (s["pair_cfg"]["a"], s["pair_cfg"]["b"]):
                    if abs(s["z"]) > s["stop_threshold"]:
                        stop_triggered = True
                        break

        if stop_triggered and hold_shares > 0 and hold_symbol in today_data:
            # 止损清仓
            sell_price = today_data[hold_symbol]["open"] * (1 - SLIPPAGE)
            revenue = hold_shares * sell_price
            commission = max(revenue * COMMISSION_RATE, 0.0)
            cash += revenue - commission
            trades.append({
                "date": today_str, "action": "止损", "symbol": hold_symbol,
                "shares": hold_shares, "price": round(sell_price, 4),
                "commission": round(commission, 2),
            })
            hold_symbol = ""
            hold_shares = 0
            days_since_trade = 0
            action = "stop_loss"

        # ── 平仓检查（|z| < close） ──
        elif hold_symbol and signals and days_since_trade >= 5:
            # 持仓ETF在所有配对中的z-score都够小 → 价差回归 → 平仓
            weak_signal_count = 0
            relevant_signals = [s for s in signals
                                if hold_symbol in (s["pair_cfg"]["a"], s["pair_cfg"]["b"])]
            for s in relevant_signals:
                if abs(s["z"]) < s["close_threshold"] and abs(s["z"]) > 0:
                    weak_signal_count += 1
            if weak_signal_count >= 1 and hold_shares > 0 and hold_symbol in today_data:
                sell_price = today_data[hold_symbol]["open"] * (1 - SLIPPAGE)
                revenue = hold_shares * sell_price
                commission = max(revenue * COMMISSION_RATE, 0.0)
                cash += revenue - commission
                trades.append({
                    "date": today_str, "action": "平仓", "symbol": hold_symbol,
                    "shares": hold_shares, "price": round(sell_price, 4),
                    "commission": round(commission, 2),
                })
                hold_symbol = ""
                hold_shares = 0
                days_since_trade = 0
                action = "close"

        # ── 开仓/切换检查 ──
        if hold_shares == 0 or action == "hold":
            # 找最强有效信号
            valid = [s for s in signals if s["target"] is not None and s["strength"] > 0 and s["volume_ok"]]
            if valid:
                best = max(valid, key=lambda s: s["strength"])
                target = best["target"]

                # 无持仓 → 开仓
                if hold_shares == 0:
                    if target in today_data:
                        buy_price = today_data[target]["open"] * (1 + SLIPPAGE)
                        max_shares = int(cash // buy_price // 100) * 100
                        if max_shares > 0:
                            cost = max_shares * buy_price
                            commission = max(cost * COMMISSION_RATE, 0.0)
                            cash -= (cost + commission)
                            hold_symbol = target
                            hold_shares = max_shares
                            days_since_trade = 0
                            trades.append({
                                "date": today_str, "action": "买入", "symbol": target,
                                "shares": max_shares, "price": round(buy_price, 4),
                                "commission": round(commission, 2),
                            })
                            action = f"buy_{target[:4]}"

                # 有持仓且信号指向不同标的 → 切换
                elif target != hold_symbol and days_since_trade >= 5:
                    if hold_symbol in today_data and target in today_data:
                        # 卖出旧
                        sell_price = today_data[hold_symbol]["open"] * (1 - SLIPPAGE)
                        revenue = hold_shares * sell_price
                        commission_sell = max(revenue * COMMISSION_RATE, 0.0)
                        cash += revenue - commission_sell
                        trades.append({
                            "date": today_str, "action": "切换卖出", "symbol": hold_symbol,
                            "shares": hold_shares, "price": round(sell_price, 4),
                            "commission": round(commission_sell, 2),
                        })
                        # 买入新
                        buy_price = today_data[target]["open"] * (1 + SLIPPAGE)
                        max_shares = int(cash // buy_price // 100) * 100
                        if max_shares > 0:
                            cost = max_shares * buy_price
                            commission_buy = max(cost * COMMISSION_RATE, 0.0)
                            cash -= (cost + commission_buy)
                            hold_symbol = target
                            hold_shares = max_shares
                            days_since_trade = 0
                            trades.append({
                                "date": today_str, "action": "切换买入", "symbol": target,
                                "shares": max_shares, "price": round(buy_price, 4),
                                "commission": round(commission_buy, 2),
                            })
                            action = f"switch_{hold_symbol[:4]}→{target[:4]}" if hold_symbol else f"buy_{target[:4]}"
                        else:
                            hold_symbol = ""
                            hold_shares = 0
                            action = "sell_only"

        # 估值
        stock_value = 0.0
        if hold_symbol and hold_symbol in today_data:
            stock_value = hold_shares * today_data[hold_symbol]["close"]
        total_value = cash + stock_value

        # 收益率
        if not records:
            daily_ret = 0.0
        else:
            prev = records[-1].total_value
            daily_ret = (total_value - prev) / prev if prev > 0 else 0.0

        cum_ret = (total_value / initial_capital) - 1

        # 持仓描述
        hold_desc = f"{hold_symbol}({hold_shares}股)" if hold_shares > 0 else "现金"
        signal_desc = "; ".join(
            f"{s['pair']}: z={s['z']:.2f}→{s['direction']}" if s['target'] else f"{s['pair']}: z={s['z']:.2f}→持有"
            for s in signals
        ) if signals else ""

        records.append(SwitchRecord(
            date=today_str, mode="A",
            hold_symbols=hold_desc,
            cash=round(cash, 2),
            stock_value=round(stock_value, 2),
            total_value=round(total_value, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            action=action,
            z_scores=z_summary,
            signals=signal_desc,
        ))

    return records, trades


# ════════════════════════════════════════════════════════════
# 模式 B：三对均分资金 + 各自轮动
# ════════════════════════════════════════════════════════════

def run_mode_b(
    etf_data: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[List[SwitchRecord], List[dict]]:
    """模式 B：资金三等分，每对独立运行轮动。

    每对拥有自己的现金池和持仓记录，互不干扰。
    同一 ETF 可能被多对同时持有（如 510050 出现在两对中）。

    每对逻辑：
      - |z| > open → 买入便宜方（用该对的现金）
      - |z| < close → 卖出持仓（回到该对现金）
      - |z| > stop → 止损
      - 否则持有不动
    """
    pair_capitals = initial_capital / len(PAIRS)
    n_pairs = len(PAIRS)

    # 每对的状态
    pair_state = []
    pair_etfs = []
    for p in PAIRS:
        pair_state.append({
            "cash": pair_capitals,
            "symbol": "",
            "shares": 0,
            "days_since_trade": 999,
        })
        pair_etfs.append({p["a"], p["b"]})

    records: List[SwitchRecord] = []
    trades: List[dict] = []

    n = len(dates)
    for idx in range(n):
        today_str = str(dates[idx].date())

        today_data = {}
        for sym in etf_data:
            if idx < len(etf_data[sym]):
                row = etf_data[sym].iloc[idx]
                today_data[sym] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }

        signals = compute_pair_signals(etf_data, idx)

        z_summary = "; ".join(
            f"{s['pair']} z={s['z']:.2f}"
            for s in signals
        ) if signals else "无数据"

        action_parts = []
        stock_value = 0.0
        total_cash = 0.0

        # 每对独立处理
        for pi, (p_cfg, st) in enumerate(zip(PAIRS, pair_state)):
            sig = signals[pi] if pi < len(signals) else None
            st["days_since_trade"] += 1

            if sig is None or not sig["volume_ok"]:
                # 无信号或成交量不足 → 不动
                if st["shares"] > 0:
                    sv = st["shares"] * today_data[st["symbol"]]["close"]
                    stock_value += sv
                else:
                    total_cash += st["cash"]
                continue

            sym_a = p_cfg["a"]
            sym_b = p_cfg["b"]
            z = sig["z"]
            target = sig["target"]

            # 止损
            if st["shares"] > 0 and abs(z) > sig["stop_threshold"] and st["symbol"] in today_data:
                sell_price = today_data[st["symbol"]]["open"] * (1 - SLIPPAGE)
                revenue = st["shares"] * sell_price
                commission = max(revenue * COMMISSION_RATE, 0.0)
                st["cash"] += revenue - commission
                trades.append({
                    "date": today_str, "action": f"P{pi+1}止损", "symbol": st["symbol"],
                    "shares": st["shares"], "price": round(sell_price, 4),
                })
                st["symbol"] = ""
                st["shares"] = 0
                st["days_since_trade"] = 0
                action_parts.append(f"P{pi+1}止损")

            # 有仓位且 |z| < close → 平仓
            elif st["shares"] > 0 and abs(z) < sig["close_threshold"] and abs(z) > 0 and st["symbol"] in today_data:
                sell_price = today_data[st["symbol"]]["open"] * (1 - SLIPPAGE)
                revenue = st["shares"] * sell_price
                commission = max(revenue * COMMISSION_RATE, 0.0)
                st["cash"] += revenue - commission
                trades.append({
                    "date": today_str, "action": f"P{pi+1}平仓", "symbol": st["symbol"],
                    "shares": st["shares"], "price": round(sell_price, 4),
                })
                st["symbol"] = ""
                st["shares"] = 0
                st["days_since_trade"] = 0
                action_parts.append(f"P{pi+1}平")

            # 无仓位且有信号 → 开仓
            elif st["shares"] == 0 and target is not None and target in today_data:
                buy_price = today_data[target]["open"] * (1 + SLIPPAGE)
                max_shares = int(st["cash"] // buy_price // 100) * 100
                if max_shares > 0:
                    cost = max_shares * buy_price
                    commission = max(cost * COMMISSION_RATE, 0.0)
                    st["cash"] -= (cost + commission)
                    st["symbol"] = target
                    st["shares"] = max_shares
                    st["days_since_trade"] = 0
                    trades.append({
                        "date": today_str, "action": f"P{pi+1}买入", "symbol": target,
                        "shares": max_shares, "price": round(buy_price, 4),
                    })
                    action_parts.append(f"P{pi+1}买{target[:4]}")
                else:
                    total_cash += st["cash"]

            # 有仓位且目标切换（不同标的）
            elif st["shares"] > 0 and target is not None and target != st["symbol"] and st["days_since_trade"] >= 5:
                if st["symbol"] in today_data and target in today_data:
                    # 卖出旧
                    sell_price = today_data[st["symbol"]]["open"] * (1 - SLIPPAGE)
                    revenue = st["shares"] * sell_price
                    commission_s = max(revenue * COMMISSION_RATE, 0.0)
                    st["cash"] += revenue - commission_s
                    # 买入新
                    buy_price = today_data[target]["open"] * (1 + SLIPPAGE)
                    max_shares = int(st["cash"] // buy_price // 100) * 100
                    if max_shares > 0:
                        cost = max_shares * buy_price
                        commission_b = max(cost * COMMISSION_RATE, 0.0)
                        st["cash"] -= (cost + commission_b)
                        trades.append({
                            "date": today_str, "action": f"P{pi+1}切", "symbol": f"{st['symbol']}→{target}",
                            "shares": max_shares, "price": round(buy_price, 4),
                        })
                        st["symbol"] = target
                        st["shares"] = max_shares
                        st["days_since_trade"] = 0
                        action_parts.append(f"P{pi+1}切{target[:4]}")
                    else:
                        st["symbol"] = ""
                        st["shares"] = 0
                        action_parts.append(f"P{pi+1}只卖")
                else:
                    # 只有旧标的无行情，跳过
                    action_parts.append(f"P{pi+1}Hold")

            # 估值
            if st["shares"] > 0 and st["symbol"] in today_data:
                stock_value += st["shares"] * today_data[st["symbol"]]["close"]
            elif st["shares"] == 0:
                total_cash += st["cash"]
            else:
                total_cash += st["cash"]

        # 总估值（现金来自有仓位的对的剩余现金）
        # 注意：有仓位的对的现金已经在上面被扣除，剩余的现金通过 stock_value 不参与时计算
        # 重新计算总现金
        total_cash = sum(st["cash"] for st in pair_state)
        total_value = total_cash + stock_value

        # 持仓描述
        hold_parts = [f"{st['symbol']}({st['shares']}股)" for st in pair_state if st["shares"] > 0]
        hold_desc = " | ".join(hold_parts) if hold_parts else "现金"

        if not records:
            daily_ret = 0.0
        else:
            prev = records[-1].total_value
            daily_ret = (total_value - prev) / prev if prev > 0 else 0.0

        cum_ret = (total_value / initial_capital) - 1
        action_str = " | ".join(action_parts) if action_parts else "hold"

        signal_desc = "; ".join(
            f"{s['pair']}: z={s['z']:.2f}→{s['direction']}"
            for s in (signals or [])
        )

        records.append(SwitchRecord(
            date=today_str, mode="B",
            hold_symbols=hold_desc,
            hold_pct="",
            cash=round(total_cash, 2),
            stock_value=round(stock_value, 2),
            total_value=round(total_value, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            action=action_str,
            z_scores=z_summary,
            signals=signal_desc,
        ))

    return records, trades


# ════════════════════════════════════════════════════════════
# 模式 C：多对按信号强度加权分配
# ════════════════════════════════════════════════════════════

def run_mode_c(
    etf_data: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    initial_capital: float = INITIAL_CAPITAL,
) -> Tuple[List[SwitchRecord], List[dict]]:
    """模式 C：按信号强度加权分配资金到各便宜方 ETF。

    算法：
      1. 每对输出目标 ETF 和信号强度 |z|
      2. 汇总所有目标，去重后按强度权重分配资金
      3. 无目标或总强度低 → 持有现金

    份额四舍五入到最小 ETF 份额 = 100 股后实施。
    """
    cash = initial_capital
    # 持仓结构：{symbol: shares}
    positions: Dict[str, int] = {}
    records: List[SwitchRecord] = []
    trades: List[dict] = []
    days_since_trade = 999

    n = len(dates)
    for idx in range(n):
        today_str = str(dates[idx].date())

        today_data = {}
        for sym in etf_data:
            if idx < len(etf_data[sym]):
                row = etf_data[sym].iloc[idx]
                today_data[sym] = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }

        signals = compute_pair_signals(etf_data, idx)

        z_summary = "; ".join(
            f"{s['pair']} z={s['z']:.2f}"
            for s in signals
        ) if signals else "无数据"

        action = "hold"
        days_since_trade += 1

        # ── 止损检查 ──
        stop_triggered = False
        for sym in list(positions.keys()):
            if positions[sym] <= 0:
                continue
            for s in (signals or []):
                if sym in (s["pair_cfg"]["a"], s["pair_cfg"]["b"]):
                    if abs(s["z"]) > s["stop_threshold"]:
                        stop_triggered = True
                        break
            if stop_triggered:
                break

        if stop_triggered and positions:
            # 清空所有持仓
            for sym in list(positions.keys()):
                sh = positions[sym]
                if sh > 0 and sym in today_data:
                    sell_price = today_data[sym]["open"] * (1 - SLIPPAGE)
                    revenue = sh * sell_price
                    commission = max(revenue * COMMISSION_RATE, 0.0)
                    cash += revenue - commission
                    trades.append({
                        "date": today_str, "action": "止损", "symbol": sym,
                        "shares": sh, "price": round(sell_price, 4),
                    })
            positions.clear()
            days_since_trade = 0
            action = "stop_all"

        # ── 正常调仓 ──
        elif signals and days_since_trade >= 3:
            # 汇总目标及其强度
            targets: Dict[str, float] = {}
            for s in signals:
                if s["target"] is not None and s["strength"] > 0 and s["volume_ok"]:
                    t = s["target"]
                    targets[t] = targets.get(t, 0) + s["strength"]

            if targets:
                total_strength = sum(targets.values())
                # 计算每个目标的目标资金
                allocations = {}
                for sym, strength in targets.items():
                    alloc = initial_capital * (strength / total_strength)
                    allocations[sym] = alloc

                # 当前仓位映射到市值
                current_values = {}
                for sym, sh in positions.items():
                    if sh > 0 and sym in today_data:
                        current_values[sym] = sh * today_data[sym]["close"]
                    else:
                        current_values[sym] = 0.0

                # 需要卖出的：当前持有但不在目标中
                to_sell = [sym for sym in list(positions.keys()) if sym not in targets]
                # 需要买入的：目标中但当前未持有或不足
                to_buy = {}

                for sym in targets:
                    target_val = allocations.get(sym, 0)
                    current_val = current_values.get(sym, 0)
                    diff = target_val - current_val
                    if diff > -100:  # 不强制卖，只补不足
                        to_buy[sym] = diff

                # 执行卖出
                for sym in to_sell:
                    sh = positions.pop(sym, 0)
                    if sh > 0 and sym in today_data:
                        sell_price = today_data[sym]["open"] * (1 - SLIPPAGE)
                        revenue = sh * sell_price
                        commission = max(revenue * COMMISSION_RATE, 0.0)
                        cash += revenue - commission
                        trades.append({
                            "date": today_str, "action": "卖出", "symbol": sym,
                            "shares": sh, "price": round(sell_price, 4),
                        })

                # 执行买入
                for sym, needed_val in sorted(to_buy.items(), key=lambda x: -x[1]):
                    if needed_val <= 0 or sym not in today_data:
                        continue
                    buy_price = today_data[sym]["open"] * (1 + SLIPPAGE)
                    available = min(needed_val, cash)
                    max_shares = int(available // buy_price // 100) * 100
                    if max_shares > 0:
                        cost = max_shares * buy_price
                        commission = max(cost * COMMISSION_RATE, 0.0)
                        cash -= (cost + commission)
                        positions[sym] = positions.get(sym, 0) + max_shares
                        trades.append({
                            "date": today_str, "action": "买入", "symbol": sym,
                            "shares": max_shares, "price": round(buy_price, 4),
                        })
                        action = "rebalance"
                        days_since_trade = 0

            else:
                # 无信号 → 如果持仓回归到 close 阈值则平仓
                to_close = []
                for sym in list(positions.keys()):
                    for s in (signals or []):
                        if sym in (s["pair_cfg"]["a"], s["pair_cfg"]["b"]):
                            if abs(s["z"]) < s["close_threshold"] and abs(s["z"]) > 0:
                                to_close.append(sym)
                                break
                for sym in set(to_close):
                    sh = positions.pop(sym, 0)
                    if sh > 0 and sym in today_data:
                        sell_price = today_data[sym]["open"] * (1 - SLIPPAGE)
                        revenue = sh * sell_price
                        commission = max(revenue * COMMISSION_RATE, 0.0)
                        cash += revenue - commission
                        trades.append({
                            "date": today_str, "action": "平仓", "symbol": sym,
                            "shares": sh, "price": round(sell_price, 4),
                        })
                        action = "close"

        # 估值
        stock_value = 0.0
        hold_parts = []
        for sym, sh in sorted(positions.items()):
            if sh > 0 and sym in today_data:
                sv = sh * today_data[sym]["close"]
                stock_value += sv
                hold_parts.append(f"{sym}({sh}股¥{sv:.0f})")
        total_value = cash + stock_value

        hold_desc = " | ".join(hold_parts) if hold_parts else "现金"
        alloc_desc = "; ".join(
            f"{sym} {targets.get(sym, 0)/max(sum(targets.values()),1)*100:.0f}%"
            for sym in sorted(targets.keys())
        ) if signals and any(s["target"] for s in signals) else ""

        if not records:
            daily_ret = 0.0
        else:
            prev = records[-1].total_value
            daily_ret = (total_value - prev) / prev if prev > 0 else 0.0

        cum_ret = (total_value / initial_capital) - 1

        signal_desc = "; ".join(
            f"{s['pair']}: z={s['z']:.2f}→{s['direction']}"
            for s in (signals or [])
        )

        records.append(SwitchRecord(
            date=today_str, mode="C",
            hold_symbols=hold_desc,
            hold_pct=alloc_desc,
            cash=round(cash, 2),
            stock_value=round(stock_value, 2),
            total_value=round(total_value, 2),
            daily_return=round(daily_ret, 6),
            cumulative_return=round(cum_ret, 6),
            action=action,
            z_scores=z_summary,
            signals=signal_desc,
        ))

    return records, trades
