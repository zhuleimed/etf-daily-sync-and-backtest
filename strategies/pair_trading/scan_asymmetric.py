#!/usr/bin/env python
"""
不对称阈值扫描 — 方向感知开仓阈值优化

测试不同的 growth_threshold 和 value_threshold 组合。
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from .config import PAIRS, INITIAL_CAPITAL, DB_PATH, OUTPUT_DIR, ZSCORE_OPEN
from strategies.momentum_rotation.data import load_all_etf_data
import strategies.pair_trading.engine_switch as eng


def compute_metrics(records, initial_capital):
    values = np.array([r.total_value for r in records])
    if len(values) < 10:
        return {"total_return": 0, "sharpe": 0, "max_dd": 0, "score": -999}

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

    # 综合评分（偏向收益和稳定性）
    score = (sharpe * 15 + total_ret * 30 + calmar * 5 - max_dd * 40
             + yearly.get(2026, 0) * 30)

    return {
        "total_return": total_ret, "annual_ret": annual_ret,
        "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar,
        "trades": trades, "yearly": yearly, "score": score,
    }


fmt_pct = lambda v: f"{v*100:+6.2f}%"

# 加载数据
symbols = list(set(p["a"] for p in PAIRS) | set(p["b"] for p in PAIRS))
print(f"\n{'=' * 75}")
print(f"  不对称阈值扫描 — 方向感知开仓阈值")
print(f"{'=' * 75}")

etf_data, dates = load_all_etf_data(
    symbols=symbols, start_date="2024-01-01", end_date="",
    db_path=DB_PATH, momentum_window=90,
)
print(f"  {len(dates)} 个交易日 | 初始资金 {INITIAL_CAPITAL:.0f}\n")

# 网格：growth_th × value_th
# growth=买成长 z>+X, value=买价值 z<-Y
# 组合列表（基于对称 3.0 最优，探索更低 growth 阈值）
combos = []

# 基础：对称 3.0（当前最优）
for g in [3.0]:
    for v in [3.0]:
        combos.append((g, v, "对称3.0基准"))

# growth更低（更容易买成长）
for g in [2.5, 2.0, 1.8, 1.5, 1.2]:
    combos.append((g, 3.0, f"growth={g}/value=3.0"))

# value更低（更容易买价值）
for v in [2.5, 2.0]:
    combos.append((3.0, v, f"growth=3.0/value={v}"))

# 双向都低
for g, v in [(2.5, 2.5), (2.0, 2.0), (1.5, 1.5)]:
    combos.append((g, v, f"对称{g}"))

# 极端测试
for g, v in [(2.0, 4.0), (1.5, 4.0), (4.0, 2.0)]:
    combos.append((g, v, f"growth={g}/value={v}"))

# 去重
seen = set()
unique = []
for g, v, label in combos:
    key = (g, v)
    if key not in seen:
        seen.add(key)
        unique.append((g, v, label))

print(f"  {'参数组合':<24} {'总收益':<10} {'年化':<10} {'夏普':<8} {'回撤':<8} {'Calmar':<8} {'交易':<6} {'2024':<8} {'2025':<8} {'2026':<8} {'评分':<8}")
print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

results = []
for growth_th, value_th, label in unique:
    eng.ZSCORE_OPEN_GROWTH = growth_th
    eng.ZSCORE_OPEN_VALUE = value_th

    records, trades = eng.run_mode_a(etf_data, dates, initial_capital=INITIAL_CAPITAL)
    m = compute_metrics(records, INITIAL_CAPITAL)

    yr = m.get("yearly", {})
    flag = " ◀-基准" if growth_th == 3.0 and value_th == 3.0 else ""
    print(f"  G{growth_th:.1f}V{value_th:.1f} {label:<14} "
          f"{m['total_return']*100:+7.2f}% "
          f"{m['annual_ret']*100:+7.2f}% "
          f"{m['sharpe']:<8.2f} "
          f"{m['max_dd']*100:>6.2f}% "
          f"{m['calmar']:<8.2f} "
          f"{m['trades']:<6} "
          f"{yr.get(2024,0)*100:+7.2f}% "
          f"{yr.get(2025,0)*100:+7.2f}% "
          f"{yr.get(2026,0)*100:+7.2f}% "
          f"{m['score']:<8.1f}"
          f"{flag}")

    results.append({**m, "growth": growth_th, "value": value_th, "label": label})

# TOP 5
results.sort(key=lambda x: -x["score"])
print(f"\n{'=' * 75}")
print(f"  TOP 5 不对称组合")
print(f"{'=' * 75}")
print(f"  {'排名':<6} {'Growth':<8} {'Value':<8} {'总收益':<10} {'夏普':<8} {'回撤':<8} {'2026':<8} {'评分':<8}")
print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for i, r in enumerate(results[:5]):
    yr = r.get("yearly", {})
    flag = " ⭐" if i == 0 else ""
    print(f"  #{i+1:<3}{flag} {r['growth']:<8.1f} {r['value']:<8.1f} "
          f"{r['total_return']*100:+7.2f}% "
          f"{r['sharpe']:<8.2f} "
          f"{r['max_dd']*100:>6.2f}% "
          f"{yr.get(2026,0)*100:+7.2f}% "
          f"{r['score']:<8.1f}")

# 最佳组合详细对比
best = results[0]
print(f"\n{'=' * 75}")
print(f"  最优不对称组合: Growth={best['growth']:.1f}, Value={best['value']:.1f}")
print(f"  (vs 基准对称: ZSCORE_OPEN={ZSCORE_OPEN})")
print(f"{'=' * 75}")

eng.ZSCORE_OPEN_GROWTH = best["growth"]
eng.ZSCORE_OPEN_VALUE = best["value"]
opt_records, opt_trades = eng.run_mode_a(etf_data, dates, initial_capital=INITIAL_CAPITAL)

eng.ZSCORE_OPEN_GROWTH = ZSCORE_OPEN
eng.ZSCORE_OPEN_VALUE = ZSCORE_OPEN
base_records, _ = eng.run_mode_a(etf_data, dates, initial_capital=INITIAL_CAPITAL)

def fmt(r):
    vals = np.array([v.total_value for v in r])
    ret = vals[-1] / INITIAL_CAPITAL - 1
    dd = (np.maximum.accumulate(vals) - vals) / np.maximum.accumulate(vals)
    return ret, dd.max()

opt_r, opt_d = fmt(opt_records)
base_r, base_d = fmt(base_records)
print(f"  总收益: {opt_r*100:+.2f}% (基准{base_r*100:+.2f}%)")
print(f"  回撤:  {opt_d*100:.2f}% (基准{base_d*100:.2f}%)")
print(f"  变化:  收益{opt_r-base_r:+.2%} 回撤{opt_d-base_d:+.2%}")

# 保存
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(OUTPUT_DIR, f"asym_scan_{ts}")
os.makedirs(out_dir, exist_ok=True)

rows = []
for r in results:
    yr = r.get("yearly", {})
    rows.append({
        "growth": r["growth"], "value": r["value"],
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
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "asym_results.csv"), index=False)
print(f"\n  输出 → {out_dir}")
print(f"{'=' * 75}\n")
