"""
模拟盘运行日志 — CSV 记录器

记录每个策略每日运行的状态快照，存储在 simulation/output/simulation_log.csv
便于跨策略追踪和复盘。
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# CSV 路径
_LOG_DIR = Path(__file__).resolve().parent.parent / "output"
_LOG_PATH = _LOG_DIR / "simulation_log.csv"

# CSV 列定义
_FIELDS = [
    "日期",           # YYYY-MM-DD
    "策略",           # 策略名称
    "操作",           # 今日操作描述
    "持仓标的",       # ETF代码
    "持仓名称",       # ETF中文名
    "持仓数量",       # 股数
    "持仓均价",       # 成本价
    "现金",           # 可用资金
    "市值",           # 持仓市值
    "总资产",         # 现金+市值
    "累计收益率",      # 总资产/初始资金-1
    "订单执行",       # 今日是否有订单成交
    "明日待执行",     # 待执行订单描述
]


def _make_action_description(report: dict, etf_pool: dict[str, str]) -> str:
    """将 report action 转为可读的中文操作描述。"""
    action = report.get("action", "unknown")

    if action == "order_blocked":
        blocked = report.get("order_blocked", {})
        return f"订单取消: {blocked.get('reason', '未知')}"

    if action == "open_pending":
        symbol = report.get("hold_symbol", "")
        return f"空仓，发出买入信号({symbol})←明日执行"

    if action == "hold_cash":
        return "空仓，无买入信号"

    if action == "hold":
        hold_sym = report.get("hold_symbol", "")
        hold_name = etf_pool.get(hold_sym, hold_sym)
        return f"持有 {hold_name}({hold_sym})"

    if action == "open":
        execd = report.get("order_executed", {})
        s = execd.get("symbol", "")
        return f"执行昨买入({s})，开仓"

    if action == "switch":
        execd = report.get("order_executed", {})
        sell = execd.get("sell", {}).get("symbol", "")
        buy = execd.get("buy", {}).get("symbol", "")
        return f"执行切换({sell}→{buy})"

    if action == "risk_sell":
        execd = report.get("order_executed", {})
        s = execd.get("symbol", "")
        return f"风控卖出({s})"

    if action == "switch_pending":
        state = report.get("state")
        po = state.pending_order if state else None
        if po:
            return f"发出切换信号({po.get('sell_symbol','')}→{po.get('buy_symbol','')})←明日执行"
        return "切换信号待执行"

    if action == "risk_pending":
        state = report.get("state")
        po = state.pending_order if state else None
        if po:
            return f"风控触发，发出卖出信号({po.get('symbol','')})←明日执行"
        return "风控待执行"

    return action


def _make_pending_description(report: dict, etf_pool: dict[str, str]) -> str:
    """描述明日待执行的订单。"""
    state = report.get("state")
    if not state or not state.pending_order:
        return ""
    po = state.pending_order
    pa = po.get("action", "")
    if pa == "buy":
        sym = po.get("symbol", "")
        return f"买入({sym})"
    elif pa == "sell":
        sym = po.get("symbol", "")
        return f"卖出({sym})"
    elif pa == "switch":
        return f"切换({po.get('sell_symbol','')}→{po.get('buy_symbol','')})"
    return str(po)


def _make_order_executed_description(report: dict) -> str:
    """描述今日已执行的订单。"""
    execd = report.get("order_executed")
    if not execd:
        blocked = report.get("order_blocked")
        if blocked:
            return f"❌{blocked.get('reason', '被封锁')}"
        return ""
    t = execd.get("type", "")
    if t == "buy":
        return f"✅买入{execd.get('symbol','')}{execd.get('shares',0)}股@{execd.get('price','')}"
    elif t == "sell":
        return f"✅卖出{execd.get('symbol','')}{execd.get('shares',0)}股@{execd.get('price','')} PnL{execd.get('pnl',0):+.2f}"
    elif t == "switch":
        s = execd.get("sell", {}).get("symbol", "")
        b = execd.get("buy", {}).get("symbol", "")
        return f"✅切换{s}→{b}"
    return ""


def append_simulation_log(
    strategy_name: str,
    report: dict[str, Any],
    etf_pool: dict[str, str],
) -> None:
    """追加一条模拟盘日志到 CSV。

    每次 strategy 完成 run_daily() 后调用一次。

    Parameters
    ----------
    strategy_name : str
        策略中文名，如 "动量轮动模拟盘"
    report : dict
        DailySimEngine.run_daily() 返回的报表
    etf_pool : dict
        {symbol: name} 映射表，用于中文名显示
    """
    state = report.get("state")
    if not state:
        return  # 无状态不可记录

    # 累计收益率
    cum_ret = ""
    total_value = report.get("total_value", state.cash)
    if state.initial_capital > 0:
        cum_ret = f"{(total_value / state.initial_capital - 1) * 100:.2f}%"

    # 持仓信息
    hold_sym = report.get("hold_symbol", "")
    hold_shares = report.get("hold_shares", 0)
    hold_name = etf_pool.get(hold_sym, hold_sym) if hold_sym else ""
    hold_cost = ""
    if state.position and state.position.shares > 0:
        hold_cost = f"{state.position.avg_cost:.4f}"

    row = {
        "日期": report.get("date", ""),
        "策略": strategy_name,
        "操作": _make_action_description(report, etf_pool),
        "持仓标的": hold_sym,
        "持仓名称": hold_name,
        "持仓数量": hold_shares,
        "持仓均价": hold_cost,
        "现金": round(state.cash, 2),
        "市值": round(report.get("stock_value", 0), 2),
        "总资产": round(total_value, 2),
        "累计收益率": cum_ret,
        "订单执行": _make_order_executed_description(report),
        "明日待执行": _make_pending_description(report, etf_pool),
    }

    # 原子追加写入（目录自动创建）
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = _LOG_PATH.exists()

    with open(_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
