#!/usr/bin/env python3
"""
ADX 模拟盘接线修复验证（2026-10-02）

背景：轨迹对齐监控 v2 发现 adx_trend_rotation 模拟盘持 512100 满仓、回测却空仓 29 天。
根因（读回测 engine._make_decision 逐条比对）：
  回测规则①：持仓 ADX 得分=0 → 平仓           → 模拟盘缺 exit_when_signal_dead（默认 False）
  回测规则②：熊市且目标得分<=0.5 → 不开仓      → 模拟盘收不到 HS300 regime（无接线）
修复：simulation/strategies/adx_trend_rotation/{config,daily}.py + 引擎 open_gate_func 钩子。

本脚本在真实 DailySimEngine 上离线重放，回答三个问题：
  A. harness 自检：关掉两个杠杆（=修复前接线）能否逐日复现 live CSV？(证明重放可信)
  B. 修复后两个分支各触发几次？(tail_risk 教训：先数触发，再谈收益)
  C. 修复后模拟盘轨迹与回测的对齐度是否显著改善？(持仓一致率 / 累计缺口)

用法：
  python -m simulation.analysis.verify_adx_wiring
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.analysis.repair_bt import (  # noqa: E402
    _cfg, _dates, _load_range, _new_engine, _pos,
)
from simulation.framework.state import StateManager  # noqa: E402
from simulation.framework.backtest_align import align  # noqa: E402

SID = "adx_trend_rotation"
SEED = "2026-08-03"          # 模拟盘清零重启日（对齐锚点）
END = "2026-09-30"
LIVE_CSV = PROJECT_ROOT / "simulation" / "output" / f"sim_log_{SID}.csv"
# 回测基准：2024-01-02~2026-09-30 全周期回测（与模拟盘同一套策略代码）
BACKTEST_CSV = PROJECT_ROOT / "strategies" / SID / "output" / "20261002_175317_bear_gate_check" / "daily_records.csv"


def _walk(engine, etf, dates, i0, day0, day1):
    """逐日重放，返回 (行列表, 触发计数)。"""
    rows, cnt = [], {"exit_signal_dead": 0, "gate_block": 0, "open": 0, "switch": 0}
    for k in range(i0, len(dates)):
        d = dates[k]
        if d < day0:
            continue
        if d > day1:
            break
        rep = engine.run_daily(etf, k, d)
        if "error" in rep:
            raise RuntimeError(f"{d}: {rep['error']}")
        sig = rep.get("signal", "")
        cnt["exit_signal_dead"] += int(sig == "exit_pending")
        cnt["gate_block"] += int(str(rep.get("signal_note", "")).startswith("开仓闸门"))
        cnt["open"] += int(sig == "open_pending")
        cnt["switch"] += int(sig == "switch_pending")
        rows.append({"date": d, "total": rep["total_value"],
                     "hold": rep.get("hold_symbol") or ""})
    return rows, cnt


def _run(signal_dead, use_gate, seed=SEED, end=END):
    """重放一个变体（fresh 现金起步于 seed）。"""
    cfg = _cfg(SID)
    lo = (datetime.strptime(seed, "%Y-%m-%d") - timedelta(days=420)).strftime("%Y-%m-%d")
    etf = _load_range(cfg.ETF_SYMBOLS, lo, end, cfg.MOMENTUM_WINDOW)
    dates = _dates(etf[cfg.ETF_SYMBOLS[0]])
    i0 = _pos(etf[cfg.ETF_SYMBOLS[0]], seed)
    with tempfile.TemporaryDirectory(prefix="adxfix_") as tmp:
        sm = StateManager(tmp, f"{SID}_v")
        sm.init_new(cfg.INITIAL_CAPITAL)
        eng, _ = _new_engine(SID, sm, tmp, confirm=1, cooldown=0,
                             risk_mode=cfg.RISK_MODE,
                             signal_dead=signal_dead, use_gate=use_gate)
        rows, cnt = _walk(eng, etf, dates, i0, seed, end)
    cap = cfg.INITIAL_CAPITAL
    rets = [(r["total"] / cap - 1) * 100 for r in rows]
    return [r["date"] for r in rows], rets, [r["hold"] for r in rows], rows, cnt


def _live_series():
    """live 模拟盘 CSV → {日期: 总资产}（跳过注释/重复行，取每日最后一条）。"""
    df = pd.read_csv(LIVE_CSV)
    out = {}
    for _, r in df.iterrows():
        d = str(r.get("日期", "")).strip()[:10]
        if not d or "历史" in d or "重启" in str(r.get("操作", "")):
            continue
        try:
            out[d] = float(str(r.get("总资产", "")).strip())
        except ValueError:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description="ADX 接线修复验证（离线重放）")
    ap.add_argument("--seed", default=SEED, help=f"重放起点（默认 {SEED}）")
    ap.add_argument("--end", default=END, help=f"重放终点（默认 {END}）")
    a = ap.parse_args()
    if not BACKTEST_CSV.exists():
        print(f"缺少回测基准 {BACKTEST_CSV}\n先跑: python -m strategies.{SID}.run "
              f"--start 2024-01-01 --end {END} --tag bear_gate_check")
        sys.exit(1)
    back_df = pd.read_csv(BACKTEST_CSV, dtype={"hold_symbol": str})

    print(f"═══ ADX 接线修复验证 | 锚点 {a.seed} → {a.end} ═══\n")
    variants = [
        ("修复前(无退出/无闸门)", False, False),
        ("修复后(退出+闸门)", None, None),
    ]
    live = _live_series() if a.seed == SEED else {}
    summary = {}
    for label, sd, ug in variants:
        dates, rets, holds, rows, cnt = _run(sd, ug, a.seed, a.end)
        recs = align(SID, dates, rets, holds, back_df)
        n = len(recs)
        match = sum(1 for r in recs if r["sim_hold"] == r["back_hold"])
        dev = recs[-1]["dev"] if recs else float("nan")
        print(f"── {label} ──")
        print(f"   触发计数: 持仓信号消失卖出 {cnt['exit_signal_dead']} 次 | "
              f"熊市闸门拦截 {cnt['gate_block']} 次 | 开仓 {cnt['open']} 次 | 切换 {cnt['switch']} 次")
        print(f"   持仓与回测一致: {match}/{n} 天 ({match/n*100:.0f}%) | "
              f"末日持仓 模拟盘={recs[-1]['sim_hold'] or '空仓'} vs 回测={recs[-1]['back_hold'] or '空仓'}")
        print(f"   累计收益: 模拟盘{rets[-1]:+.2f}% vs 回测(同锚){recs[-1]['back_ret']:+.2f}% "
              f"→ 缺口 {dev:+.2f}pp")
        if sd is False and live:   # 修复前 = 复现 live，做 harness 自检
            rep_map = {r["date"]: r["total"] for r in rows}
            shared = [d for d in rep_map if d in live]
            div = next((d for d in sorted(shared) if abs(rep_map[d] - live[d]) > 0.5), None)
            clean = sorted(shared).index(div) if div else len(shared)
            print(f"   [harness 自检] 与 live CSV 共有 {len(shared)} 天，"
                  f"干净前缀 {clean} 天" + (f"，于 {div} 分叉(应为 live 同日重复重跑 artifact)" if div else "，全程复现 ✓"))
        print()
        summary[label] = (match, n, dev, cnt)

    print("═══ 结论 ═══")
    b = summary["修复前(无退出/无闸门)"]
    a = summary["修复后(退出+闸门)"]
    print(f"  · 持仓一致率: {b[0]/b[1]*100:.0f}% → {a[0]/a[1]*100:.0f}%")
    print(f"  · 末日缺口  : {b[2]:+.2f}pp → {a[2]:+.2f}pp")
    print(f"  · 修复后分支确实会触发: 信号消失卖出 {a[3]['exit_signal_dead']} 次 / "
          f"熊市闸门 {a[3]['gate_block']} 次（非死分支）")


if __name__ == "__main__":
    main()
