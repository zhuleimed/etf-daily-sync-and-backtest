#!/usr/bin/env python
"""
方案A — 联合参数网格搜索

基于单参数扫描的结果，对 TOP 候选组合做联合验证。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .config import PAIRS, INITIAL_CAPITAL, DB_PATH, OUTPUT_DIR
from strategies.momentum_rotation.data import load_all_etf_data
import strategies.pair_trading.engine_switch as eng


def compute_metrics(records, initial_capital):
    values = np.array([r.total_value for r in records])
    if len(values) < 10:
        return {"total_return": 0, "sharpe": 0, "max_dd": 0}

    total_ret = values[-1] / initial_capital - 1
    n = len(values)
    annual_ret = (values[-1] / initial_capital) ** (245 / max(n, 1)) - 1 if n > 0 else 0
    dly = np.diff(values) / values[:-1]
    dly = np.append(0, dly)
    excess = dly.mean() - 0.03 / 245
    std = dly.std()
    sharpe = (excess / std) * np.sqrt(245) if std > 1e-8 else 0
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    max_dd = dd.max()
    calmar = annual_ret / max_dd if max_dd > 0 else 0
    trades = sum(1 for r in records if r.action not in ("hold", ""))

    df = pd.DataFrame([{"date": r.date, "v": r.total_value} for r in records])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    yearly = {}
    for year in sorted(df["year"].unique()):
        yd = df[df["year"] == year]
        if len(yd) >= 5:
            yearly[year] = yd["v"].iloc[-1] / yd["v"].iloc[0] - 1

    return {
        "total_return": total_ret, "annual_ret": annual_ret,
        "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar,
        "trades": trades, "yearly": yearly,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    symbols = list(set(p["a"] for p in PAIRS) | set(p["b"] for p in PAIRS))
    print(f"\n{'=' * 75}")
    print(f"  方案A 联合网格搜索")
    print(f"  {'=' * 75}")
    print(f"  数据加载…")
    etf_data, dates = load_all_etf_data(
        symbols=symbols, start_date=args.start, end_date=args.end,
        db_path=DB_PATH, momentum_window=90,
    )
    print(f"  {len(dates)} 个交易日")

    # 候选参数组合（基于单参数扫描的 TOP 结果）
    candidates = [
        # (period, open, close) — 说明
        # 单参数最佳
        (20, 2.0, 0.3),   # 基准周期20
        (60, 2.0, 0.3),   # 基准（原始）
        (60, 3.0, 0.3),   # 高开仓阈值（单参数最佳）
        (60, 2.5, 0.5),   # 中等提升
        (60, 2.0, 0.7),   # 高平仓阈值
        (60, 2.5, 0.3),   # 高开仓+小平仓
        # 周期20探索
        (20, 2.5, 0.3),   # 周期20+高开仓
        (20, 3.0, 0.3),   # 周期20+最高开仓
        (20, 2.0, 0.5),   # 周期20+中平仓
        (20, 2.0, 0.7),   # 周期20+高平仓
        (20, 2.5, 0.5),   # 周期20+双提升
        (20, 3.0, 0.5),   # 周期20+最高开仓+中平仓
        (20, 2.5, 0.7),   # 周期20+高开仓+高平仓
        (20, 1.5, 0.5),   # 周期20+低开仓+中平仓
        # 中等周期
        (40, 2.0, 0.3),   # 周期40
        (40, 2.5, 0.5),   # 周期40+双提升
        (40, 3.0, 0.3),   # 周期40+最高开仓
        (45, 2.0, 0.3),   # 周期45
        # 高开仓组合
        (60, 2.5, 0.7),   # 双高
        (60, 3.0, 0.5),   # 最高开+中平
        (60, 3.0, 0.7),   # 双最高（最后验证）
    ]

    results = []
    print(f"\n  {'参数组合':<20} {'总收益':<10} {'年化':<10} {'夏普':<8} {'回撤':<8} {'Calmar':<8} {'交易':<6} {'2024':<8} {'2025':<8} {'2026':<8} {'评分':<8}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    best_score = -999
    best_params = None

    for period, open_th, close_th in candidates:
        eng.ZSCORE_PERIOD = period
        eng.ZSCORE_OPEN = open_th
        eng.ZSCORE_CLOSE = close_th

        records, trades = eng.run_mode_a(etf_data, dates, initial_capital=args.money)
        m = compute_metrics(records, args.money)

        # 综合评分
        score = (m["sharpe"] * 15 +
                 m["total_return"] * 30 +
                 m["calmar"] * 5 -
                 m["max_dd"] * 40)
        # 2026年加分
        yr_2026 = m.get("yearly", {}).get(2026, 0)
        score += yr_2026 * 30

        yr = m.get("yearly", {})
        label = f"P{period}O{open_th}C{close_th}"
        print(f"  {label:<20} "
              f"{m['total_return']*100:+7.2f}% "
              f"{m['annual_ret']*100:+7.2f}% "
              f"{m['sharpe']:<8.2f} "
              f"{m['max_dd']*100:>6.2f}% "
              f"{m['calmar']:<8.2f} "
              f"{m['trades']:<6} "
              f"{yr.get(2024,0)*100:+7.2f}% "
              f"{yr.get(2025,0)*100:+7.2f}% "
              f"{yr.get(2026,0)*100:+7.2f}% "
              f"{score:<8.1f}")

        if score > best_score:
            best_score = score
            best_params = (period, open_th, close_th)

        results.append({**m, "params": (period, open_th, close_th), "score": score})

    # 输出 TOP5
    results.sort(key=lambda x: -x["score"])
    print(f"\n{'=' * 75}")
    print(f"  TOP 5 参数组合")
    print(f"{'=' * 75}")
    print(f"  {'排名':<6} {'参数':<20} {'总收益':<10} {'夏普':<8} {'回撤':<8} {'Calmar':<8} {'2026':<8} {'评分':<8}")
    print(f"  {'-'*6} {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for i, r in enumerate(results[:5]):
        p = r["params"]
        yr = r.get("yearly", {})
        score_str = f"{r['score']:.1f}"
        flag = " ⭐" if i == 0 else ""
        print(f"  #{i+1:<3} {flag} P{p[0]}O{p[1]}C{p[2]}  "
              f"{r['total_return']*100:+7.2f}% "
              f"{r['sharpe']:<8.2f} "
              f"{r['max_dd']*100:>6.2f}% "
              f"{r['calmar']:<8.2f} "
              f"{yr.get(2026,0)*100:+7.2f}% "
              f"{score_str:<8}")
    print()

    # 最佳组合详细验证
    p, o, c = best_params
    print(f"{'=' * 75}")
    print(f"  最优组合: PERIOD={p}, OPEN={o}, CLOSE={c}")
    print(f"{'=' * 75}")
    eng.ZSCORE_PERIOD = p
    eng.ZSCORE_OPEN = o
    eng.ZSCORE_CLOSE = c
    eng.ZSCORE_STOP = 3.0
    records, trades = eng.run_mode_a(etf_data, dates, initial_capital=args.money)
    m = compute_metrics(records, args.money)

    # 同时跑原始参数做对比
    eng.ZSCORE_PERIOD = 60
    eng.ZSCORE_OPEN = 2.0
    eng.ZSCORE_CLOSE = 0.3
    base_records, _ = eng.run_mode_a(etf_data, dates, initial_capital=args.money)
    base_m = compute_metrics(base_records, args.money)

    print(f"\n  {'指标':<12} {'原始(60/2.0/0.3)':<20} {'优化后':<20} {'变化':<10}")
    print(f"  {'---':<12} {'---':<20} {'---':<20} {'---':<10}")
    for key, label, is_pct in [
        ("total_return", "总收益", True), ("annual_ret", "年化", True),
        ("sharpe", "夏普", False), ("max_dd", "回撤", True),
        ("calmar", "Calmar比", False), ("trades", "交易数", False)
    ]:
        orig = base_m.get(key, 0)
        opt = m.get(key, 0)
        if is_pct:
            print(f"  {label:<12} {orig*100:>+10.2f}%      {opt*100:>+10.2f}%      {opt-orig:+.2f}%")
        else:
            print(f"  {label:<12} {orig:>10.2f}      {opt:>10.2f}      {opt-orig:+.2f}")

    yr_b = base_m.get("yearly", {})
    yr_o = m.get("yearly", {})
    for y in [2024, 2025, 2026]:
        print(f"  {y:<12} {yr_b.get(y,0)*100:>+10.2f}%      {yr_o.get(y,0)*100:>+10.2f}%      {yr_o.get(y,0)-yr_b.get(y,0):+.2f}%")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, f"grid_search_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # 保存全量对比
    summary_rows = []
    for r in results:
        p = r["params"]
        yr = r.get("yearly", {})
        summary_rows.append({
            "period": p[0], "open": p[1], "close": p[2],
            "total_return": f"{r['total_return']*100:.2f}%",
            "sharpe": f"{r['sharpe']:.2f}",
            "max_dd": f"{r['max_dd']*100:.2f}%",
            "calmar": f"{r['calmar']:.2f}",
            "trades": r["trades"],
            "ret_2024": f"{yr.get(2024,0)*100:.2f}%",
            "ret_2025": f"{yr.get(2025,0)*100:.2f}%",
            "ret_2026": f"{yr.get(2026,0)*100:.2f}%",
            "score": f"{r['score']:.1f}",
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "grid_results.csv"), index=False)
    print(f"\n  输出 → {out_dir}")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()
