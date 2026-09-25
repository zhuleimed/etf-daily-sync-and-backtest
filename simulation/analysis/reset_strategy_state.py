"""策略状态清零工具 —— 把指定策略重置为空仓、收益归零重新起算。

用途：当策略的**逻辑本身**发生变更（如接线修复、参数重构）时，之前积累的
      持仓与盈亏不再代表新逻辑，需要清零重来。

做两件事（缺一不可）：
  1. 状态文件 state_{sid}.json → 空仓 / 现金=初始资金 / 累计盈亏与交易记录清空
  2. 在 sim_log_{sid}.csv 追加一条"注释行"，作为收益序列的断点

  注释行的措辞必须包含「清零重启」或「重构」关键字——下游有两处依赖它：
    · summary._compute_metrics   —— 从最后一个【注释行之后重新计算绩效指标
    · backtest_align             —— 识别收益断点，避免与回测对齐时产生假偏差
  漏掉关键字会导致"清零后仍显示历史亏损回撤"。

用法：
    python -m simulation.analysis.reset_strategy_state --sid bollinger_reversion --reason "接线修复"
    python -m simulation.analysis.reset_strategy_state --all-candidates --reason "接线修复" --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.log_writer import _FIELDS  # noqa: E402
from simulation.framework.state import PositionState, StateManager  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "simulation" / "output"

# 2026-09 接线修复涉及的全部策略（原共用动量轮动信号函数）
CANDIDATE_SIDS = [
    "bollinger_reversion", "median_momentum", "sharpe_ranking", "sortino_ranking",
    "spread_reversion", "tail_risk", "volume_price", "dual_momentum",
]


def reset_one(sid: str, reason: str, dry_run: bool = False) -> None:
    sm = StateManager(OUTPUT_DIR, sid)
    state = sm.load()
    if state is None:
        print(f"  ⚠ {sid}: 无状态文件，跳过")
        return

    csv_path = OUTPUT_DIR / f"sim_log_{sid}.csv"
    name = sid
    if csv_path.exists():
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            name = rows[-1].get("策略") or sid

    old_pos = state.position
    print(f"  {sid}: 现金 {state.cash:.2f} → {state.initial_capital:.2f} | "
          f"持仓 {old_pos.symbol or '空'}×{old_pos.shares} → 空 | "
          f"累计盈亏 {state.cumulative_pnl:+.2f} → 0")

    if dry_run:
        return

    # ── 1. 状态清零 ──
    state.position = PositionState()
    state.cash = state.initial_capital
    state.cumulative_pnl = 0.0
    state.cumulative_cost = 0.0
    state.trade_log = []
    state.peak_value = state.initial_capital
    state.total_value = state.initial_capital
    state.days_since_switch = 0
    state.pending_order = None
    state.last_update = date.today().isoformat()
    state.strategy_name = sid
    sm.save(state)

    # ── 2. CSV 断点注释行（含"清零重启"关键字，下游依赖） ──
    if csv_path.exists():
        row = {k: "" for k in _FIELDS}
        row["日期"] = date.today().isoformat()
        row["策略"] = name
        row["操作"] = f"【接线修复·清零重启】{reason}——持仓与盈亏清零，累计收益率自本日重新起算（历史见上方行）"
        row["累计收益率"] = "0.00%"
        with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=_FIELDS).writerow(row)
    else:
        print(f"    ⚠ {sid}: 无 CSV，未写断点行")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", action="append", default=[])
    ap.add_argument("--all-candidates", action="store_true")
    ap.add_argument("--reason", default="策略逻辑变更")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sids = list(args.sid)
    if args.all_candidates:
        sids += CANDIDATE_SIDS
    if not sids:
        ap.error("请用 --sid 或 --all-candidates 指定策略")

    print(f"{'[试运行] ' if args.dry_run else ''}清零 {len(sids)} 个策略，理由：{args.reason}")
    for sid in sids:
        reset_one(sid, args.reason, args.dry_run)
    print("完成" if not args.dry_run else "（未写入任何文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
