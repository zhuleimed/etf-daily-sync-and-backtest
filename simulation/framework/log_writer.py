"""
模拟盘运行日志 — CSV 记录器

每个策略独立一个 CSV 文件，路径：simulation/output/sim_log_{策略id}.csv
新策略首次运行时自动创建并从第一天开始记录。
已有策略（有 state_xxx.json）调用 init_from_state() 记录当前状态作为起点。

CSV 列说明：
  操作 — 今日操作描述，如"空仓，发出买入信号←明日执行"、"持有"、"执行切换：开盘卖→买"等
  订单执行 — 今日是否有订单成交及明细
  明日待执行 — 今日产生的信号将在明日开盘执行

切换（switch）说明：
  T日信号产生：发出切换信号(A→B)，此时不交易
  T+1日执行：以开盘价卖出A，同时用全部所得买入B
  任一方向被涨跌停封锁则整个切换取消
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

_LOG_DIR = Path(__file__).resolve().parent.parent / "output"

_FIELDS = [
    "日期",
    "策略",
    "操作",
    "持仓标的",
    "持仓名称",
    "持仓数量",
    "持仓均价",
    "现金",
    "市值",
    "总资产",
    "累计收益率",
    "订单执行",
    "明日待执行",
]


def _csv_path(strategy_id: str) -> Path:
    """返回某个策略的 CSV 文件路径。"""
    return _LOG_DIR / f"sim_log_{strategy_id}.csv"


def _db_path() -> str:
    """返回数据库路径。"""
    return str(Path(__file__).resolve().parent.parent.parent / "data" / "etf_daily.db")


def _lookup_close(symbol: str, date_str: str) -> float:
    """从数据库查询某日收盘价。"""
    try:
        db = _db_path()
        with sqlite3.connect(db) as conn:
            cur = conn.execute(
                "SELECT close FROM etf_daily WHERE symbol = ? AND date = ?",
                (symbol, date_str),
            )
            row = cur.fetchone()
            return row[0] if row else 0.0
    except Exception:
        return 0.0


def _make_action_description(report: dict, etf_pool: dict[str, str]) -> str:
    """将 report action 转为清晰的中文操作描述，切换一定说清楚方向。"""
    action = report.get("action", "unknown")
    state = report.get("state")

    if action == "order_blocked":
        blocked = report.get("order_blocked", {})
        return f"订单取消: {blocked.get('reason', '未知')}"

    if action == "open_pending":
        # 空仓 → 发出买入信号
        po = state.pending_order if state else None
        sym = po.get("symbol", "") if po else ""
        name = etf_pool.get(sym, sym)
        return f"空仓，发出买入信号({name}[{sym}])←明日开盘执行"

    if action == "hold_cash":
        return "空仓，无买入信号"

    if action == "hold":
        sym = report.get("hold_symbol", "")
        name = etf_pool.get(sym, sym)
        return f"持有 {name}[{sym}]"

    if action == "open":
        # 昨日买入信号 → 今日开仓执行
        execd = report.get("order_executed", {})
        s = execd.get("symbol", "")
        sh = execd.get("shares", 0)
        px = execd.get("price", 0)
        name = etf_pool.get(s, s)
        return f"执行昨买入：开盘买入{name}[{s}] {sh}股@{px:.4f}"

    if action == "switch":
        # 昨日切换信号 → 今日执行：开盘卖旧买新
        execd = report.get("order_executed", {})
        sell = execd.get("sell", {})
        buy = execd.get("buy", {})
        s_sym = sell.get("symbol", "")
        b_sym = buy.get("symbol", "")
        s_sh = sell.get("shares", 0)
        b_sh = buy.get("shares", 0)
        s_px = sell.get("price", 0)
        b_px = buy.get("price", 0)
        s_name = etf_pool.get(s_sym, s_sym)
        b_name = etf_pool.get(b_sym, b_sym)
        return f"执行切换：开盘卖出{s_name}[{s_sym}] {s_sh}股@{s_px:.4f} → 买入{b_name}[{b_sym}] {b_sh}股@{b_px:.4f}"

    if action == "switch_pending":
        # 今日产生切换信号（有持仓 → 切换）
        po = state.pending_order if state else None
        if po:
            s_sym = po.get("sell_symbol", "")
            b_sym = po.get("buy_symbol", "")
            s_name = etf_pool.get(s_sym, s_sym)
            b_name = etf_pool.get(b_sym, b_sym)
            return f"发出切换信号({s_name}[{s_sym}]→{b_name}[{b_sym}])←明日开盘卖出{s_sym}并买入{b_sym}"
        return "切换信号待执行"

    if action == "risk_sell":
        execd = report.get("order_executed", {})
        s = execd.get("symbol", "")
        sh = execd.get("shares", 0)
        px = execd.get("price", 0)
        name = etf_pool.get(s, s)
        return f"风控卖出：开盘卖出{name}[{s}] {sh}股@{px:.4f}"

    if action == "risk_pending":
        po = state.pending_order if state else None
        if po:
            sym = po.get("symbol", "")
            name = etf_pool.get(sym, sym)
            return f"风控触发，发出卖出信号({name}[{sym}])←明日开盘执行"
        return "风控待执行"

    return action


def _make_pending_description(report: dict, etf_pool: dict[str, str]) -> str:
    """明日待执行订单描述。"""
    state = report.get("state")
    if not state or not state.pending_order:
        return ""
    po = state.pending_order
    pa = po.get("action", "")
    if pa == "buy":
        sym = po.get("symbol", "")
        name = etf_pool.get(sym, sym)
        return f"明日开盘买入{name}[{sym}]"
    elif pa == "sell":
        sym = po.get("symbol", "")
        name = etf_pool.get(sym, sym)
        return f"明日开盘卖出{name}[{sym}]"
    elif pa == "switch":
        s_sym = po.get("sell_symbol", "")
        b_sym = po.get("buy_symbol", "")
        s_name = etf_pool.get(s_sym, s_sym)
        b_name = etf_pool.get(b_sym, b_sym)
        return f"明日开盘卖出{s_name}[{s_sym}]→买入{b_name}[{b_sym}]"
    return str(po)


def _make_order_executed_description(report: dict) -> str:
    """今日订单成交描述。"""
    execd = report.get("order_executed")
    if not execd:
        blocked = report.get("order_blocked")
        if blocked:
            return f"❌{blocked.get('reason', '被封锁')}"
        return ""
    t = execd.get("type", "")
    if t == "buy":
        return f"✅买入{execd.get('symbol','')} {execd.get('shares',0)}股 @{execd.get('price','')}"
    elif t == "sell":
        return f"✅卖出{execd.get('symbol','')} {execd.get('shares',0)}股 @{execd.get('price','')} 盈亏{execd.get('pnl',0):+.2f}"
    elif t == "switch":
        s = execd.get("sell", {}).get("symbol", "")
        b = execd.get("buy", {}).get("symbol", "")
        return f"✅完成切换{s}→{b}"
    return ""


# ═══════════════════════════════════════════════════════════════
#  公开 API
# ═══════════════════════════════════════════════════════════════


def append_simulation_log(
    strategy_id: str,
    strategy_name: str,
    report: dict[str, Any],
    etf_pool: dict[str, str],
) -> None:
    """追加一条模拟盘日志到策略独立的 CSV。

    CSV 不存在时自动处理：
      - 如果 state_{strategy_id}.json 存在 → 先写入起始行，再写当日行
      - 如果 state 不存在 → 直接新建 CSV 并写当日行（新策略首次运行）

    Parameters
    ----------
    strategy_id : str
        策略ID，用于 CSV 文件名，如 "momentum_rotation"
    strategy_name : str
        策略中文名，如 "动量轮动模拟盘"
    report : dict
        DailySimEngine.run_daily() 返回的报表
    etf_pool : dict
        {symbol: name} 映射表
    """
    state = report.get("state")
    if not state:
        return

    csv_path = _csv_path(strategy_id)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── CSV 不存在时：自动初始化（追记已有状态） ──
    if not csv_path.exists():
        state_file = _LOG_DIR / f"state_{strategy_id}.json"
        if state_file.exists():
            try:
                raw = json.loads(state_file.read_text(encoding="utf-8"))
                pos = raw.get("position", {})
                sym = pos.get("symbol", "")
                pos_shares = pos.get("shares", 0)
                avg_cost = pos.get("avg_cost", 0.0)
                last_up = raw.get("last_update", "")
                ic = raw.get("initial_capital", 10000)
                cash = raw.get("cash", 0.0)

                sv = 0.0
                if sym and pos_shares > 0:
                    cp = _lookup_close(sym, last_up)
                    sv = pos_shares * cp if cp > 0 else 0.0

                tv = cash + sv
                cr = f"{(tv / ic - 1) * 100:.2f}%" if ic > 0 else ""
                hn = etf_pool.get(sym, sym) if sym else ""
                hc = f"{avg_cost:.4f}" if sym and pos_shares > 0 else ""

                init_row = {
                    "日期": f"{last_up}←历史起点",
                    "策略": strategy_name,
                    "操作": "以下为追记的历史状态起始行"
                            + ("（有持仓）" if pos_shares > 0 else "（空仓）"),
                    "持仓标的": sym, "持仓名称": hn,
                    "持仓数量": pos_shares, "持仓均价": hc,
                    "现金": round(cash, 2), "市值": round(sv, 2),
                    "总资产": round(tv, 2), "累计收益率": cr,
                    "订单执行": "", "明日待执行": "",
                }
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=_FIELDS)
                    w.writeheader()
                    w.writerow(init_row)
            except Exception:
                pass

    # ── 写入当日行 ──
    total_value = report.get("total_value", state.cash)
    cum_ret = ""
    if state.initial_capital > 0:
        cum_ret = f"{(total_value / state.initial_capital - 1) * 100:.2f}%"

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

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not csv_path.exists() or (csv_path.stat().st_size == 0):
            writer.writeheader()
        writer.writerow(row)
