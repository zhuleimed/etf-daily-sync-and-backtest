#!/usr/bin/env python3
"""方向 5 稳健性验证：宽度闸不是窗口巧合吗？（2026-10-02）

背景：方向 5 实验里"宽度闸(<30%)·半仓"双判据通过（全周期夏普 0.56→0.63、熊市 MDD
-22.6%→-12.9%）。但阈值 0.30 与"半仓"是**从 market_breadth 既有配置继承**的，
且**半仓闸一多半时间在风险态**（平均暴露 ~77%）——必须回答两件事：
  A. 是不是阈值/暴露刚好合适（窗口巧合）？
  B. 改善是因为"择时"，还是只是因为"平均少投一点"（暴露更低）？

## 三关（事前登记判据）
① 参数邻域：thr ∈ {.20,.25,.30,.35,.40} × expo ∈ {0,0.3,0.5,0.7}（20 组）
   每组要求：全周期夏普 > 基线 且 熊市 MDD 改善 ≥5pp
   → **通过**需 ≥50%（≥10/20 组）满足
② 跨期/滚动：
   (a) 2022-23 选最优 → 2024-26 验证；(b) 2024-26 选最优 → 2022-23 验证；
   (c) 滚动：训练 2022-23→测 2024；2022-24→测 2025；2022-25→测 2026
   → **通过**需 (a)(b) 均通过，且 (c) 中 ≥2/3 折的测试段夏普 > 该段基线
③ 安慰剂（关键）：
   - 常数暴露对照：暴露 = 真实闸门的平均暴露（同样的"少投"程度，但不择时）
   - 随机闸门：同样触发比例、随机选日，200 次（seed 固定）→ 看真实闸门夏普的分位
   → **通过**需 真实闸门 > 常数暴露对照，且 真实夏普 ≥ 随机分布的 P90

用法：python -m simulation.analysis.portfolio_gate_validate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.analysis.portfolio_gate import (     # noqa: E402
    build_experiment, gated_nav, breadth_series, _stats, BEAR, BULL, FULL,
)

THRS = [0.20, 0.25, 0.30, 0.35, 0.40]
EXPOS = [0.0, 0.3, 0.5, 0.7]
N_RANDOM = 200
SEED = 20261002


def _gated_params(r_base, breadth, thr, expo):
    return gated_nav(r_base, breadth < thr, expo)


def main():
    R, r_base, signals = build_experiment()
    nav_base = (1 + r_base).cumprod()
    breadth = breadth_series(R.index)
    base_full = _stats(nav_base, *FULL)
    base_bear = _stats(nav_base, *BEAR)

    # ── ① 参数邻域 ──
    print(f"\n═══ ① 参数邻域（thr × expo = {len(THRS)}×{len(EXPOS)}）═══")
    n_pass, total, best = 0, 0, []
    for thr in THRS:
        row = []
        for expo in EXPOS:
            nav = _gated_params(r_base, breadth, thr, expo)
            f = _stats(nav, *FULL)
            b = _stats(nav, *BEAR)
            ok = (f["sharpe"] > base_full["sharpe"]) and (b["mdd"] - base_bear["mdd"] >= 5)
            n_pass += ok
            total += 1
            row.append(f"{f['sharpe']:.2f}{'✅' if ok else '  '}")
            best.append((f["sharpe"], thr, expo))
        print(f"  thr={thr:.2f}: " + "  ".join(f"expo{e}:{v}" for e, v in zip(EXPOS, row)))
    print(f"  → 通过 {n_pass}/{total} 组（要求 ≥50%）: {'✅ 通过' if n_pass >= total/2 else '❌ 证伪'}")

    # ── ② 跨期 / 滚动 ──
    print(f"\n═══ ② 跨期 / 滚动 ═══")
    def _pick(train_lo, train_hi):
        best_cfg, best_sh = None, -9
        for thr in THRS:
            for expo in EXPOS:
                st = _stats(_gated_params(r_base, breadth, thr, expo), train_lo, train_hi)
                if st and st["sharpe"] > best_sh:
                    best_cfg, best_sh = (thr, expo), st["sharpe"]
        return best_cfg, best_sh

    results = {}
    for label, (tr, te) in {
        "(a) 22-23定参→24-26验证": (BEAR, BULL),
        "(b) 24-26定参→22-23验证": (BULL, BEAR),
    }.items():
        cfg, sh_tr = _pick(*tr)
        nav = _gated_params(r_base, breadth, *cfg)
        s_te = _stats(nav, *te)
        s_base = _stats(nav_base, *te)
        ok = s_te["sharpe"] > s_base["sharpe"]
        results[label] = ok
        print(f"  {label}: 定参 {cfg}（训练段夏普 {sh_tr:.2f}）→ 验证段夏普 {s_te['sharpe']:.2f} "
              f"vs 基线 {s_base['sharpe']:.2f} → {'✅' if ok else '❌'}")

    folds = [("2022-23", BEAR, ("2024-01-02", "2024-12-31")),
             ("2022-24", ("2022-01-04", "2024-12-31"), ("2025-01-02", "2025-12-31")),
             ("2022-25", ("2022-01-04", "2025-12-31"), ("2026-01-05", "2026-09-30"))]
    ok_folds = 0
    for train_label, tr, te in folds:
        cfg, _ = _pick(*tr)
        nav = _gated_params(r_base, breadth, *cfg)
        s_te = _stats(nav, *te)
        s_base = _stats(nav_base, *te)
        ok = s_te["sharpe"] > s_base["sharpe"]
        ok_folds += ok
        print(f"  滚动 训练{train_label} → 测试{te[0][:4]}: 定参 {cfg} → 夏普 {s_te['sharpe']:.2f} "
              f"vs 基线 {s_base['sharpe']:.2f} → {'✅' if ok else '❌'}")
    roll_ok = ok_folds >= 2
    cross_ok = all(results.values())
    print(f"  → 双向跨期 {'✅' if cross_ok else '❌'} | 滚动 {ok_folds}/3 {'✅' if roll_ok else '❌'}")

    # ── ③ 安慰剂 ──
    print(f"\n═══ ③ 安慰剂对照 ═══")
    thr, expo = 0.30, 0.5
    sig = (breadth < thr)
    frac = float(sig.reindex(R.index).ffill().fillna(False).mean())
    avg_expo = 1 - frac * (1 - expo)
    print(f"  真实闸门: 触发比例 {frac:.0%}，平均暴露 {avg_expo:.0%}")
    nav_real = gated_nav(r_base, sig, expo)
    real_sh = _stats(nav_real, *FULL)["sharpe"]
    real_mdd = _stats(nav_real, *FULL)["mdd"]

    nav_const = (1 + r_base * avg_expo).cumprod()
    const_sh = _stats(nav_const, *FULL)["sharpe"]
    const_mdd = _stats(nav_const, *FULL)["mdd"]
    print(f"  常数暴露 {avg_expo:.0%}（不择时）: 夏普 {const_sh:.2f} MDD {const_mdd:.2f}%")
    print(f"  真实闸门              : 夏普 {real_sh:.2f} MDD {real_mdd:.2f}%")

    rng = np.random.default_rng(SEED)
    n = len(R.index)
    k = int(round(frac * n))
    shs = []
    for _ in range(N_RANDOM):
        idx = rng.choice(n, size=k, replace=False)
        mask = pd.Series(False, index=R.index)
        mask.iloc[idx] = True
        shs.append(_stats(gated_nav(r_base, mask, expo), *FULL)["sharpe"])
    shs = np.array(shs)
    p90 = float(np.percentile(shs, 90))
    pct = float((shs < real_sh).mean() * 100)
    print(f"  随机闸门 {N_RANDOM} 次（同样触发比例）: 夏普 中位 {np.median(shs):.2f} / P90 {p90:.2f}")
    print(f"  → 真实闸门夏普 {real_sh:.2f} 位于随机分布 P{pct:.0f}")
    placebo_ok = (real_sh > const_sh) and (real_sh >= p90)
    print(f"  → 择时是否带来超额（> 常数暴露 且 ≥P90）: {'✅ 通过' if placebo_ok else '❌ 证伪'}")

    # ── 总结 ──
    print(f"\n═══ 总结 ═══")
    print(f"  ① 参数邻域 {n_pass}/{total} {'✅' if n_pass >= total/2 else '❌'}"
          f" | ② 跨期 {'✅' if cross_ok else '❌'} / 滚动 {ok_folds}/3"
          f" | ③ 安慰剂 {'✅' if placebo_ok else '❌'}")
    if n_pass >= total/2 and cross_ok and roll_ok and placebo_ok:
        print("  → 三关全过：宽度闸不是窗口巧合，且择时本身贡献超额")
    else:
        print("  → 有未过关项，需按未过项重新评估（不要只看全窗口那一行）")


if __name__ == "__main__":
    main()
