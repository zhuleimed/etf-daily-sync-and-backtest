"""
组合策略 — 每日模拟盘入口

聚合 momentum_rotation（80%）和 pair_trading（20%）的每日净值。

不直接执行交易，而是读取两个子策略的状态文件，按权重合并净值。
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

from simulation.framework.state import StateManager, SimState
from simulation.framework.data import is_trading_day
from simulation.framework.notify import push_daily_report, push_error_alert

from simulation.strategies.combined.config import (
    TOTAL_CAPITAL, MOMENTUM_PCT, PAIR_PCT,
    STRATEGY_NAME, STATE_FILE_DIR,
)

logger = logging.getLogger("combined_sim")


def _read_state_value(state_path: Path, initial: float) -> float:
    """从状态文件读取当前总资产。"""
    if not state_path.exists():
        return initial
    try:
        with open(state_path) as f:
            raw = json.load(f)
        cash = raw.get("cash", initial)
        pos = raw.get("position", {})
        shares = pos.get("shares", 0)
        # 没有最新 close，使用 total_cost + pnl 近似
        total_cost = pos.get("total_cost", 0)
        cum_pnl = raw.get("cumulative_pnl", 0)
        if shares > 0:
            return cash + total_cost + cum_pnl
        return cash
    except Exception:
        return initial


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    today_str = date.today().isoformat()
    logger.info(f"{STRATEGY_NAME} | {today_str}")

    if not is_trading_day(today_str):
        push_daily_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
        return

    state_dir = Path(STATE_FILE_DIR)
    mom_state_path = state_dir / "state_momentum_rotation.json"
    pair_state_path = state_dir / "state_pair_trading.json"

    mom_value = _read_state_value(mom_state_path, TOTAL_CAPITAL * MOMENTUM_PCT)
    pair_value = _read_state_value(pair_state_path, TOTAL_CAPITAL * PAIR_PCT)

    # 组合净值 = 80% 动量 + 20% 配对
    mom_initial = TOTAL_CAPITAL * MOMENTUM_PCT  # 8000
    pair_initial = TOTAL_CAPITAL * PAIR_PCT       # 2000

    mom_return = mom_value / mom_initial - 1 if mom_initial > 0 else 0
    pair_return = pair_value / pair_initial - 1 if pair_initial > 0 else 0

    weighted_mom = mom_value * MOMENTUM_PCT
    weighted_pair = pair_value * PAIR_PCT
    # 实际上组合总资产 = mom_position的实际市值 + pair_position的实际市值
    # 但这里更准确的组合估值是用比例加权
    combined_value = mom_initial * (1 + mom_return) + pair_initial * (1 + pair_return)
    combined_return = combined_value / TOTAL_CAPITAL - 1

    # 读取两个子策略的 state 对象（用于保存组合策略的状态）
    state_mgr = StateManager(str(STATE_FILE_DIR), "combined")
    state = state_mgr.load() or state_mgr.init_new(TOTAL_CAPITAL)
    state.last_update = today_str

    # 更新组合状态中的现金和总资产
    state.cash = combined_value  # 将总资产存入 cash（组合无实际持仓）
    if combined_value > state.peak_value:
        state.peak_value = combined_value

    state_mgr.save(state)

    lines = [
        f"📊 组合策略汇总 ({today_str})",
        f"  动量({MOMENTUM_PCT:.0%}): ¥{mom_value:.2f} ({mom_return*100:+.2f}%)",
        f"  配对({PAIR_PCT:.0%}): ¥{pair_value:.2f} ({pair_return*100:+.2f}%)",
        f"  ─────────────────────────────",
        f"  组合总资产: ¥{combined_value:.2f}",
        f"  组合收益率: {combined_return*100:+.2f}%",
    ]

    for line in lines:
        logger.info(line)
    push_daily_report(STRATEGY_NAME, lines)
    logger.info(f"{STRATEGY_NAME} 完成 ✓")


if __name__ == "__main__":
    main()
