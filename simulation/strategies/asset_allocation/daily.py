"""资产配置策略 — 模拟盘每日运行入口（独立实现，多资产权重模式）

机制（与回测完全一致）：
  - 月末最后交易日收盘：用过去 60 日波动率计算风险平价权重（信号）
  - 下月首个交易日开盘：按新权重调仓（T+1 执行，检查涨跌停）
  - 每日收盘估值 + 微信日报 + CSV 日志 + SQLite 快照

⚠️ 独立实现（不走 DailySimEngine——那是单仓模型）：
  必须手动维护状态字段（last_update/total_value）——见 memory 独立策略教训。

状态文件：simulation/output/state_asset_allocation.json
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from simulation.framework.data import load_latest_data, is_trading_day
from simulation.framework.notify import push_daily_report, push_error_alert
from simulation.framework.log_writer import append_simulation_log
from simulation.framework import sim_db

from strategies.asset_allocation.config import (
    ASSETS, INITIAL_CAPITAL, VOL_WINDOW, ANNUAL_FACTOR, COMMISSION, SLIPPAGE,
)
from strategies.asset_allocation.backtest import risk_parity_weights

logger = logging.getLogger("asset_alloc_sim")

STRATEGY_ID = "asset_allocation"
STRATEGY_NAME = "资产配置(风险平价)"
STATE_FILE = PROJECT_ROOT / "simulation" / "output" / f"state_{STRATEGY_ID}.json"
DB_PATH = PROJECT_ROOT / "data" / "etf_daily.db"


# ════════════════════════════════════════════════════════
# 状态管理（独立 JSON，多资产持仓）
# ════════════════════════════════════════════════════════

def load_state() -> dict | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state: dict) -> None:
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def init_state() -> dict:
    return {
        "version": 1,
        "initial_capital": INITIAL_CAPITAL,
        "last_update": "",
        "cash": INITIAL_CAPITAL,
        "holdings": {},          # {symbol: {"shares": int, "avg_cost": float}}
        "target_weights": {},    # 当前目标权重（月末计算）
        "pending_weights": None, # 月末算好、下月初执行的权重
        "total_value": 0.0,
        "peak_value": INITIAL_CAPITAL,
        "trade_log": [],
        "strategy_name": STRATEGY_NAME,
    }


# ════════════════════════════════════════════════════════
# 权重与调仓
# ════════════════════════════════════════════════════════

def compute_weights(etf_data: dict[str, pd.DataFrame], date: str) -> dict[str, float]:
    """月末风险平价权重（用截至 date 的 60 日波动率）。"""
    rets = {}
    for sym, df in etf_data.items():
        sub = df[df["date"] <= date]
        if len(sub) >= 30:
            rets[sym] = sub["close"].pct_change().dropna()
    vol_df = pd.DataFrame(rets).std() * np.sqrt(ANNUAL_FACTOR)
    if vol_df.isna().all():
        return {}
    w = risk_parity_weights(vol_df)
    return {k: v for k, v in w.items() if v > 0.005}


def execute_rebalance(
    state: dict, etf_data: dict[str, pd.DataFrame], today_str: str,
) -> dict:
    """按目标权重调仓（开盘价执行）。返回调仓明细。"""
    target = state.get("pending_weights") or {}
    if not target:
        return {}

    trades = {}
    total_value = state["total_value"]
    positions = state["holdings"]

    # 先算各资产目标市值
    target_value = {sym: total_value * w for sym, w in target.items()}

    for sym, tv in target_value.items():
        px_row = etf_data.get(sym)
        if px_row is None:
            continue
        row = px_row[px_row["date"] == today_str]
        if row.empty:
            continue
        open_px = float(row.iloc[0]["open"])
        cur_shares = positions.get(sym, {}).get("shares", 0)
        cur_value = cur_shares * open_px
        diff = tv - cur_value
        if abs(diff) < tv * 0.005:  # 0.5% 内不动
            continue
        # 检查涨跌停（简化：open 相对昨收 ±10%/20%）
        prev_row = px_row[px_row["date"] < today_str]
        if not prev_row.empty:
            prev_close = float(prev_row.iloc[-1]["close"])
            limit = 0.20 if sym.startswith(("159", "588")) else 0.10
            if open_px >= prev_close * (1 + limit):
                trades[sym] = f"涨停跳过"
                continue
            if open_px <= prev_close * (1 - limit):
                trades[sym] = f"跌停跳过"
                continue
        px = open_px * (1 + SLIPPAGE)
        shares_delta = int(diff // px // 100) * 100
        if shares_delta == 0:
            continue
        cost = shares_delta * px
        fee = abs(cost) * COMMISSION
        state["cash"] -= cost + fee
        new_shares = cur_shares + shares_delta
        if new_shares <= 0:
            positions.pop(sym, None)
        else:
            positions[sym] = {
                "shares": new_shares,
                "avg_cost": (cur_value + cost) / new_shares,
            }
        trades[sym] = f"{'买' if shares_delta > 0 else '卖'}{abs(shares_delta)}股@{px:.3f}"

    state["pending_weights"] = None
    state["target_weights"] = target
    return trades


# ════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════

def main() -> None:
    today_str = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        msg = f"{today_str} 非交易日，跳过"
        logger.info(msg)
        push_daily_report(STRATEGY_NAME, [msg])
        return

    # ── 数据加载（波动率窗口 + 60 天缓冲） ──
    lookback = VOL_WINDOW + 30
    etf_data = load_latest_data(
        list(ASSETS.keys()), str(DB_PATH), lookback_days=lookback, momentum_window=20
    )
    if not etf_data or any(d.empty for d in etf_data.values()):
        msg = "行情数据加载失败"
        logger.error(msg)
        push_error_alert(STRATEGY_NAME, msg)
        return

    # ── 状态 ──
    state = load_state()
    if state is None:
        state = init_state()
    state["last_update"] = today_str

    # ── 执行昨日待执行调仓（月初首日） ──
    trades = {}
    if state.get("pending_weights"):
        trades = execute_rebalance(state, etf_data, today_str)

    # ── 估值（收盘价） ──
    total_value = state["cash"]
    for sym, pos in state["holdings"].items():
        df = etf_data.get(sym)
        if df is None:
            continue
        row = df[df["date"] == today_str]
        if not row.empty:
            total_value += pos["shares"] * float(row.iloc[0]["close"])
    state["total_value"] = round(total_value, 2)
    if total_value > state.get("peak_value", 0):
        state["peak_value"] = total_value

    # ── 月末：计算下月权重（信号） ──
    # 日历判断（不能用数据窗口——数据总是截至今天，恒会误判"今天=月末"）：
    # 今天之后到月末若还有工作日（chinese_calendar），则今天不是月末最后交易日
    def _is_month_last_trading_day(ts: pd.Timestamp) -> bool:
        from chinese_calendar import is_workday
        cur = ts + pd.Timedelta(days=1)
        month_end = ts + pd.offsets.MonthEnd(0)
        while cur <= month_end:
            if is_workday(cur.date()):
                return False
            cur += pd.Timedelta(days=1)
        return True

    is_month_end = _is_month_last_trading_day(pd.Timestamp(today_str))
    if is_month_end:
        new_w = compute_weights(etf_data, today_str)
        if new_w:
            state["pending_weights"] = new_w

    save_state(state)

    # ── CSV 日志 ──
    total_return = (total_value / state["initial_capital"] - 1) * 100
    hold_desc = ";".join(
        f"{ASSETS.get(s, s)}{p['shares']}股" for s, p in state["holdings"].items()
    ) or "空仓"
    action = "调仓: " + ";".join(trades.values()) if trades else (
        "月末计算下月权重" if is_month_end else "持有"
    )
    report = {
        "date": today_str,
        "action": action,
        "total_value": total_value,
        "holdings": hold_desc,
        "target_weights": state.get("target_weights", {}),
        "pending_weights": state.get("pending_weights"),
    }
    append_simulation_log(STRATEGY_ID, STRATEGY_NAME, report, ASSETS)
    try:
        sim_db.record_account_daily({
            "date": today_str,
            "strategy": STRATEGY_ID,
            "strategy_name": STRATEGY_NAME,
            "cash": round(state["cash"], 2),
            "stock_value": round(total_value - state["cash"], 2),
            "total_value": round(total_value, 2),
            "total_return": round(total_value / state["initial_capital"] - 1, 6),
            "position_symbol": hold_desc[:50],
            "position_shares": len(state["holdings"]),
        })
    except Exception as e:
        logger.warning(f"SQLite日结记录失败: {e}")

    # ── 微信日报 ──
    lines = [f"  {STRATEGY_NAME}  |  {today_str}"]
    lines.append("")
    lines.append("【本月执行】")
    if trades:
        lines.append("  >> " + "; ".join(f"{ASSETS.get(k,k)}: {v}" for k, v in trades.items()))
    else:
        lines.append("  >> 无调仓（权重未变）")
    lines.append("")
    lines.append("【当前配置】")
    for sym, w in sorted(state.get("target_weights", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {ASSETS.get(sym, sym)}: {w*100:.0f}%")
    if state.get("pending_weights"):
        lines.append("")
        lines.append("【下月配置（明日开盘执行）】")
        for sym, w in sorted(state["pending_weights"].items(), key=lambda x: -x[1]):
            lines.append(f"  {ASSETS.get(sym, sym)}: {w*100:.0f}%")
    lines.append("")
    lines.append("【账户日结】")
    lines.append(f"  总资产: {total_value:,.2f}  总收益率: {total_return:+.2f}%")
    push_daily_report(STRATEGY_NAME, lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
