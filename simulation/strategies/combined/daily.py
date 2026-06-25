"""
组合策略 — 每日模拟盘入口

聚合 momentum_rotation（80%）和 pair_trading（20%）的每日净值。
读取子策略的状态文件，提取信号和持仓信息，按权重合并显示。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.state import StateManager
from simulation.framework.data import is_trading_day
from simulation.framework.notify import push_daily_report, push_error_alert

from simulation.strategies.combined.config import (
    TOTAL_CAPITAL, MOMENTUM_PCT, PAIR_PCT,
    STRATEGY_NAME, STATE_FILE_DIR,
)

logger = logging.getLogger("combined_sim")

ETF_NAMES = {
    "510050": "上证50", "510300": "沪深300", "510500": "中证500",
    "512100": "中证1000", "563000": "中证2000", "159915": "创业板",
    "588000": "科创50",
}


def _read_state(state_path: Path) -> dict | None:
    """读取JSON状态文件，返回原始dict。"""
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception:
        return None


def _format_signal(raw: dict | None) -> str:
    """从状态文件的 pending_order 提取信号描述。"""
    if not raw:
        return "无信号"
    po = raw.get("pending_order")
    if not po:
        return "无新信号"
    action = po.get("action", "?")
    if action == "buy":
        sym = po.get("symbol", "")
        sym_code = sym
        return f"买入{ETF_NAMES.get(sym, sym[:4])}({sym_code})（明日执行）"
    elif action == "sell":
        sym = po.get("symbol", "")
        sym_code = sym
        return f"卖出{ETF_NAMES.get(sym, sym[:4])}({sym_code})（明日执行）"
    elif action == "switch":
        ss = po.get("sell_symbol", "")
        bs = po.get("buy_symbol", "")
        return f"切换{ETF_NAMES.get(ss, ss[:4])}({ss})->{ETF_NAMES.get(bs, bs[:4])}({bs})（明日执行）"
    return f"其他({action})"


def _format_holding(raw: dict | None, initial_capital: float) -> str:
    """从状态文件提取持仓描述和资金状况。"""
    if not raw:
        return "空仓", 0.0
    cash = raw.get("cash", 0)
    pos = raw.get("position", {})
    shares = pos.get("shares", 0)
    symbol = pos.get("symbol", "")
    total_cost = pos.get("total_cost", 0)
    cum_pnl = raw.get("cumulative_pnl", 0)

    if shares > 0 and symbol:
        sym_name = ETF_NAMES.get(symbol, symbol[:4])
        # 使用状态文件中保存的 total_value（含当日浮动盈亏）
        tv = raw.get("total_value", cash + total_cost)
        return f"{sym_name}({symbol}) {shares}股", tv
    else:
        return "空仓", cash


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
        return

    state_dir = Path(STATE_FILE_DIR)
    mom_raw = _read_state(state_dir / "state_momentum_rotation.json")
    pair_raw = _read_state(state_dir / "state_pair_trading.json")

    # ── 提取子策略信号 ──
    mom_signal = _format_signal(mom_raw)
    pair_signal = _format_signal(pair_raw)

    # ── 提取子策略持仓/资金 ──
    SUB_CAPITAL = 10000  # 各子策略独立使用全资 10000
    mom_hold, mom_total = _format_holding(mom_raw, SUB_CAPITAL)
    pair_hold, pair_total = _format_holding(pair_raw, SUB_CAPITAL)

    # 如果状态文件不存在，用初始值
    if mom_raw is None:
        mom_hold, mom_total = "空仓（未开始）", SUB_CAPITAL
        mom_signal = "未运行"
    if pair_raw is None:
        pair_hold, pair_total = "空仓（未开始）", SUB_CAPITAL
        pair_signal = "未运行"

    # ── 计算组合净值 ──
    # 子策略独立运行全资，用收益率加权计算组合
    mom_return = (mom_total / SUB_CAPITAL - 1) if SUB_CAPITAL > 0 else 0
    pair_return = (pair_total / SUB_CAPITAL - 1) if SUB_CAPITAL > 0 else 0
    mom_alloc = TOTAL_CAPITAL * MOMENTUM_PCT  # 8000
    pair_alloc = TOTAL_CAPITAL * PAIR_PCT       # 2000
    combined_total = mom_alloc * (1 + mom_return) + pair_alloc * (1 + pair_return)
    combined_return = (combined_total / TOTAL_CAPITAL - 1) if TOTAL_CAPITAL > 0 else 0

    # ── 持久化组合状态 ──
    state_mgr = StateManager(str(STATE_FILE_DIR), "combined")
    state = state_mgr.load() or state_mgr.init_new(TOTAL_CAPITAL)
    state.last_update = today_str
    state.cash = combined_total
    if combined_total > state.peak_value:
        state.peak_value = combined_total
    state_mgr.save(state)

    # ── 统一格式报告 ──
    lines = []
    lines.append("")
    lines.append("  ===========================================")
    lines.append(f"  {STRATEGY_NAME} | {today_str}")
    lines.append(f"  ===========================================")

    lines.append(f"  >> 今日子策略信号")
    lines.append(f"      动量轮动({MOMENTUM_PCT:.0%}): {mom_signal}")
    lines.append(f"      持仓: {mom_hold}")
    lines.append(f"      配对交易({PAIR_PCT:.0%}): {pair_signal}")
    lines.append(f"      持仓: {pair_hold}")

    lines.append(f"  -------------------------------------------")
    lines.append(f"  组合日结")
    mom_display = mom_alloc * (1 + mom_return)
    pair_display = pair_alloc * (1 + pair_return)
    lines.append(f"    动量(80%): {mom_display:>8.2f}  ({mom_return*100:+7.2f}%)")
    lines.append(f"    配对(20%): {pair_display:>8.2f}  ({pair_return*100:+7.2f}%)")
    lines.append(f"    ---------------------------------------")
    lines.append(f"    组合总资产: {combined_total:>8.2f}  收益率: {combined_return*100:+7.2f}%")

    lines.append(f"  ===========================================")

    for line in lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
