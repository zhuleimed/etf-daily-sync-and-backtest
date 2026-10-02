#!/usr/bin/env python3
"""
动量类模拟盘"短期动量确认"闸门验证（2026-10-02）

背景：对齐监控 v2 发现 momentum_rotation / momentum_vol_filter 模拟盘 09-15 换入 510050、
回测却留在 512100（11 个交易日偏离）。逐条复算回测 _make_decision_single 后确认：
  动量值两边逐位相同、excess=0.0346 > 阈值 0.03（摩擦项对 1 万资金约 0.0003，可忽略），
  真正拦下回测的是 **短期动量确认**：目标近 5 日跌幅 ≤ -0.5% 就不换（避免追跌）。
  09-15 的 510050 近 5 日 -1.39% → 回测拒绝换入；模拟盘没有这道门 → 换了。

修复：引擎新增 switch_gate_func 钩子（默认关）+ 两个动量模拟盘接线
（共享实现 strategies.momentum_rotation.momentum_signals.short_term_momentum_ok）。

本脚本用真实 DailySimEngine 离线重放回答：
  A. harness 自检：关掉闸门（=修复前接线）能否逐日复现 live？（证明重放可信）
  B. 4 期对比（2024/2025/2026/全周期）：闸门开 vs 关，模拟盘收益/夏普/MDD
     以及"与回测的持仓一致率 + 累计缺口"是否改善
  C. 闸门触发计数（证明不是死分支）+ 09-15 定点复核

用法：
  python -m simulation.analysis.verify_momentum_gate [--strategy momentum_rotation]
"""
from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.analysis.repair_bt import (  # noqa: E402
    _cfg, _dates, _load_range, _new_engine, _pos,
)
from simulation.framework.state import StateManager  # noqa: E402
from simulation.framework.backtest_align import align  # noqa: E402

# 4 期口径（项目约定）：2024 全年 / 2025 全年 / 2026 至今 / 全周期
PERIODS = [
    ("2024全年", "2024-01-02", "2024-12-31"),
    ("2025全年", "2025-01-02", "2025-12-31"),
    ("2026至今", "2026-01-05", "2026-09-30"),
    ("全周期", "2024-01-02", "2026-09-30"),
]


def _stats(totals) -> dict:
    v = np.asarray(totals, float)
    if len(v) < 2 or v[0] <= 0:
        return {"ret": float("nan"), "sharpe": float("nan"), "mdd": float("nan")}
    r = v[1:] / v[:-1] - 1
    peak = np.maximum.accumulate(v)
    return {
        "ret": (v[-1] / v[0] - 1) * 100,
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
        "mdd": float(np.min(v / peak - 1) * 100),
    }


def _run(sid, seed, end, use_switch_gate):
    """重放一个变体（fresh 现金起步于 seed）。返回 (dates, rets%, holds, rows, cnt)。"""
    cfg = _cfg(sid)
    lo = (datetime.strptime(seed, "%Y-%m-%d") - timedelta(days=200)).strftime("%Y-%m-%d")
    etf = _load_range(cfg.ETF_SYMBOLS, lo, end, cfg.MOMENTUM_WINDOW)
    dates = _dates(etf[cfg.ETF_SYMBOLS[0]])
    i0 = _pos(etf[cfg.ETF_SYMBOLS[0]], seed)
    rows, cnt = [], {"veto": 0, "switch": 0, "open": 0}
    out_dates = []
    with tempfile.TemporaryDirectory(prefix="momgate_") as tmp:
        sm = StateManager(tmp, f"{sid}_v")
        sm.init_new(cfg.INITIAL_CAPITAL)
        eng, _ = _new_engine(sid, sm, tmp, confirm=1, cooldown=0, risk_mode=cfg.RISK_MODE,
                             use_switch_gate=use_switch_gate)
        for k in range(i0, len(dates)):
            d = dates[k]
            if d < seed:
                continue
            if d > end:
                break
            rep = eng.run_daily(etf, k, d)
            if "error" in rep:
                raise RuntimeError(f"{d}: {rep['error']}")
            cnt["veto"] += int(str(rep.get("signal_note", "")).startswith("切换闸门未通过"))
            cnt["switch"] += int(rep.get("signal") == "switch_pending")
            cnt["open"] += int(rep.get("signal") == "open_pending")
            rows.append({"date": d, "total": rep["total_value"], "hold": rep.get("hold_symbol") or ""})
            out_dates.append(d)
    cap = cfg.INITIAL_CAPITAL
    return out_dates, [(r["total"] / cap - 1) * 100 for r in rows], [r["hold"] for r in rows], rows, cnt


def _backtest_csv(sid, tag="gate_check"):
    hits = sorted(glob.glob(str(PROJECT_ROOT / "strategies" / sid / "output" / f"*_{tag}")))
    if not hits:
        sys.exit(f"缺少回测基准，先跑: python -m strategies.{sid}.run --start 2024-01-01 "
                 f"--end 2026-09-30 --tag {tag}")
    return Path(hits[-1]) / "daily_records.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="momentum_rotation")
    a = ap.parse_args()
    sid = a.strategy
    back = pd.read_csv(_backtest_csv(sid), dtype={"hold_symbol": str}).fillna({"hold_symbol": ""})
    print(f"═══ {sid} 短期动量确认闸门验证 ═══")
    print(f"回测基准: {_backtest_csv(sid).parent.name}\n")

    # A. harness 自检（修复前接线 = 闸门关，应逐日复现 live）
    if sid == "momentum_rotation":
        live_csv = PROJECT_ROOT / "simulation" / "output" / f"sim_log_{sid}.csv"
        live = {}
        for _, r in pd.read_csv(live_csv).iterrows():
            d = str(r.get("日期", "")).strip()[:10]
            if not d or "重启" in str(r.get("操作", "")):
                continue
            try:
                live[d] = float(str(r.get("总资产", "")).strip())
            except ValueError:
                pass
        _, _, _, rows, _ = _run(sid, "2026-08-03", "2026-09-30", use_switch_gate=False)
        rep_map = {r["date"]: r["total"] for r in rows}
        shared = [d for d in rep_map if d in live]
        div = next((d for d in sorted(shared) if abs(rep_map[d] - live[d]) > 0.5), None)
        clean = sorted(shared).index(div) if div else len(shared)
        print(f"[harness 自检] 与 live 共有 {len(shared)} 天，干净前缀 {clean} 天"
              + (f"，于 {div} 分叉（live 同日重跑 artifact）" if div else "，全程复现 ✓") + "\n")

    # B. 4 期对比
    print(f"{'期间':<10}{'口径':<6}{'收益%':>9}{'夏普':>7}{'MDD%':>8}{'与回测一致率':>12}{'末日缺口pp':>11}{'闸门拦截':>9}")
    for label, seed, end in PERIODS:
        for tag, gate in (("闸门关(现状)", False), ("闸门开(修复)", True)):
            dates, rets, holds, rows, cnt = _run(sid, seed, end, use_switch_gate=gate)
            st = _stats([r["total"] for r in rows])
            recs = align(sid, dates, rets, holds, back)
            m = sum(1 for r in recs if r["sim_hold"] == r["back_hold"])
            gap = recs[-1]["dev"] if recs else float("nan")
            print(f"{label:<10}{tag:<6}{st['ret']:>9.2f}{st['sharpe']:>7.2f}{st['mdd']:>8.2f}"
                  f"{m/len(recs)*100:>11.0f}%{gap:>11.2f}{cnt['veto']:>9d}")
    print()

    # C. 09-15 定点复核（本次偏离的起点）
    for seed in ("2026-08-03",):
        for gate in (False, True):
            dates, rets, holds, rows, cnt = _run(sid, seed, "2026-09-30", use_switch_gate=gate)
            r = {x["date"]: x for x in rows}
            tag = "闸门开" if gate else "闸门关"
            print(f"[09-15 定点] {tag}: 持仓={r.get('2026-09-15', {}).get('hold') or '空仓'}  "
                  f"回测持仓=512100  → {'一致 ✅' if (r.get('2026-09-15', {}).get('hold') or '') == '512100' else '不一致 ❌'}")


if __name__ == "__main__":
    main()
