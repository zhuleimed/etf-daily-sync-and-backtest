#!/usr/bin/env python3
"""方向 5：组合层"总闸"（熊市开关）实验（2026-10-02）

## 假设
H1：在**组合层**给风险平价组合加总闸（识别到"风险态"就降仓），能改善风险调整收益，
    并显著压低熊市回撤。
    与"策略内部加过滤器永远是负优化"（铁律3）不同——**组合层形态从未验证过**，
    这是方向 5 的全部意义。

## 事前判据（任一成立即证伪 → 方向 5 终止）
F1：全周期(2022-2026)夏普 ≤ 无闸门基线
F2：熊市段(2022-2023) 最大回撤改善 < 5pp

## 三个闸门候选（全部用项目既有参数，不调参）
① 宽度闸   ：7 宽基中 close>MA20 的占比 < 0.30（market_breadth 的 WEAK 阈值）→ 风险态
② HS300 闸 ：000300 close < MA10（hs300_ma_timing 的网格最优周期）→ 风险态
③ 波动率闸 ：组合 20 日已实现波动 > 过去 250 日的 80 分位 → 风险态
每种闸门 × {半仓 0.5 / 空仓 0.0}

## 方法要点
- 组合 = 19 条 sleeve 的风险平价（月度再平衡，见 portfolio_layer.simulate_portfolio）
- 闸门信号只用 **t-1 及之前**的数据（当日信号次日生效）
- 风险态期间收益按 exposure × 组合收益（其余现金，0 收益，保守）
- 窗口 2022-01-04→2026-09-30（含 2022-2023 熊市；sleeve 曲线现场跑到该窗口）

用法：python -m simulation.analysis.portfolio_gate [--refresh]
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.framework.portfolio_weights import collect_sleeves, align_window, VOL_WINDOW  # noqa: E402
from simulation.analysis.portfolio_layer import simulate_portfolio                              # noqa: E402
from strategies.momentum_rotation import config as C                                            # noqa: E402
from strategies.momentum_rotation.data import load_all_etf_data                                 # noqa: E402
from strategies.market_breadth.engine import compute_breadth                                    # noqa: E402
import sqlite3                                                                                  # noqa: E402

BEAR = ("2022-01-04", "2023-12-29")
BULL = ("2024-01-02", "2026-09-30")
FULL = ("2022-01-04", "2026-09-30")
EXPOSURES = [("半仓", 0.5), ("空仓", 0.0)]
BREATH_WEAK = 0.30        # market_breadth 的弱市阈值
HS300_MA = 10             # hs300_ma_timing 的网格最优


def _stats(nav: pd.Series, lo: str, hi: str) -> dict | None:
    s = nav[(nav.index >= lo) & (nav.index <= hi)]
    if len(s) < 5:
        return None
    v = s.values
    r = v[1:] / v[:-1] - 1
    peak = np.maximum.accumulate(v)
    return {"ret": (v[-1] / v[0] - 1) * 100,
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "mdd": float(np.min(v / peak - 1) * 100), "n": len(v)}


def gate_signals(dates: pd.DatetimeIndex, port_ret: pd.Series) -> dict[str, pd.Series]:
    """三个闸门信号（True=风险态）。全部只用 t 及之前的数据，使用方再滞后一天。"""
    lo = (dates[0] - timedelta(days=400)).strftime("%Y-%m-%d")
    signals = {}

    # ① 宽度闸：7 宽基 close>MA20 占比
    etf, _ = load_all_etf_data(symbols=C.ETF_SYMBOLS, start_date=lo,
                               end_date=dates[-1].strftime("%Y-%m-%d"), db_path=C.DB_PATH)
    d0 = etf[C.ETF_SYMBOLS[0]]["date"]
    dmap = {str(d)[:10]: i for i, d in enumerate(d0)}
    br = {}
    for d in dates:
        i = dmap.get(str(d)[:10])
        if i is not None:
            br[d] = compute_breadth(etf, i, 20)
    breadth = pd.Series(br)
    signals["宽度闸(<30%)"] = breadth < BREATH_WEAK

    # ② HS300 闸：指数 close < MA10
    with sqlite3.connect(f"file:{C.DB_PATH}?mode=ro", uri=True) as c:
        idx = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol='000300' AND date>=? ORDER BY date",
            c, params=[lo])
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"]
    signals["HS300闸(<MA10)"] = (idx < idx.rolling(HS300_MA).mean()).reindex(dates).ffill().fillna(False)

    # ③ 波动率闸：组合 20 日波动 > 过去 250 日的 80 分位
    rv = port_ret.rolling(20).std() * np.sqrt(252)
    thr = rv.rolling(250, min_periods=60).quantile(0.80)
    signals["波动率闸(>80分位)"] = (rv > thr).fillna(False)
    return signals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    print(f"═══ 方向 5：组合层总闸实验 | {FULL[0]} → {FULL[1]} ═══")
    R_raw, src, note = collect_sleeves(FULL[0], FULL[1], run_missing=True)
    R = R_raw[(R_raw.index >= FULL[0]) & (R_raw.index <= FULL[1])]
    for sid, why in note.items():
        print(f"  ⚠ 跳过 {sid}: {why}")
    if R.shape[1] < 5:
        print(f"⚠ 可用 sleeve 不足（{R.shape[1]} 条），终止")
        return
    print(f"  sleeve {R.shape[1]} 条 × {R.shape[0]} 天"
          f"（{R.index[0].date()} → {R.index[-1].date()}）")

    nav_base = simulate_portfolio(R, "rp")
    r_base = nav_base.pct_change().fillna(0.0)
    signals = gate_signals(R.index, r_base)

    rows = [("基线·无闸门", None, 1.0)]
    for name, sig in signals.items():
        for expo_label, expo in EXPOSURES:
            rows.append((f"{name}·{expo_label}", sig, expo))

    print(f"\n{'方案':<24}{'期间':<12}{'收益%':>9}{'夏普':>7}{'MDD%':>8}{'风险态天数':>10}")
    table = {}
    for label, sig, expo in rows:
        mult = pd.Series(1.0, index=R.index)
        if sig is not None:
            s = sig.reindex(R.index).ffill().fillna(False).astype(float)
            mult = (1 + s * (expo - 1)).shift(1).fillna(1.0)   # 次日生效
        nav = (1 + r_base * mult).cumprod()
        table[label] = {}
        risk_days = int((mult < 1).sum())
        for plabel, (lo, hi) in [("熊市2022-23", BEAR), ("牛市2024-26", BULL), ("全周期", FULL)]:
            st = _stats(nav, lo, hi)
            table[label][plabel] = st
            if st:
                print(f"{label:<24}{plabel:<12}{st['ret']:>9.2f}{st['sharpe']:>7.2f}"
                      f"{st['mdd']:>8.2f}{risk_days if plabel=='全周期' else '':>10}")

    # ── 判据 ──
    print(f"\n═══ 判据（事前登记）═══")
    b = table["基线·无闸门"]
    for label, sig, expo in rows:
        if sig is None:
            continue
        w = table[label]
        f1 = "通过" if (w["全周期"] and b["全周期"] and w["全周期"]["sharpe"] > b["全周期"]["sharpe"]) else "证伪"
        d_mdd = (w["熊市2022-23"]["mdd"] - b["熊市2022-23"]["mdd"]) if w["熊市2022-23"] and b["熊市2022-23"] else 0
        f2 = "通过" if d_mdd >= 5 else "证伪"
        print(f"  {label:<24} 全周期夏普 {w['全周期']['sharpe']:.2f} vs {b['全周期']['sharpe']:.2f} → F1 {f1}"
              f" | 熊市MDD改善 {d_mdd:+.1f}pp → F2 {f2}")


if __name__ == "__main__":
    main()
