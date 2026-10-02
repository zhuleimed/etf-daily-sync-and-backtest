#!/usr/bin/env python3
"""方向 3 第一步：策略层组合（2026-10-02）

## 问题
30+ 条策略各自单打独斗，组合层只有资产配置（ETF 风险平价）和 combined（固定 80/20）。
本实验回答：**把已有策略当资产来配，能不能同时改善收益与回撤？该怎么配？**

## 假设与事前判据
H1：策略层组合（等权 / 风险平价）能同时改善"收益-回撤"——夏普高于单策略中位数、回撤小于单策略中位数。
- F1：组合全周期夏普 ≤ 单策略夏普中位数 → 证伪
- F2：组合全周期 MDD ≥ 单策略 MDD 中位数 → 证伪
（中位数是可比基准：单策略"事后最好"是上界、不可投资；中位数代表"随便挑一条的期望"）

## 方法
- 曲线来源：每条策略**自己的回测**（2024-01-02→2026-09-30，同一窗口）。缺 run.py 的直接
  驱动其 engine（同接口）；`combined` 排除（它本身是 momentum+pair 的聚合器，计入会重复）。
- 组合构建：**只用过去数据**。每月末按"过去 60 交易日波动率"算权重（等权=1/N；风险平价=1/σ 归一），
  下个月持有，期间各 sleeve 独立涨跌（漂移、不日内再平衡）。
- 输出：4 期对比（2024/2025/2026/全周期）× {等权, 风险平价, 波动率目标化, 单策略中位数/最好/最差}
- 相关性：21 条曲线的两两相关均值（分散化空间的直接度量）

## 用法
    python -m simulation.analysis.portfolio_layer [--start 2024-01-01] [--end 2026-09-30] [--refresh]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "simulation" / "output" / "portfolio_layer_curves.csv"
# run.py 有 bug、改为直接驱动 engine 的策略（2026-10-02 实测）
#   macd_trend_rotation: run.py 回测跑完后在 MetricsCalculator.compute(initial_capital=...)
#   处 TypeError（签名不匹配）→ daily_records.csv 不落盘
FORCE_ENGINE = {"macd_trend_rotation"}
EXCLUDE = {
    "combined",         # 聚合器（momentum+pair），计入会重复
    "neural_momentum",  # 2026-10-02 实测：其 run.py 回测与 momentum_rotation 曲线**逐位相同**
                        # （其 strategies/neural_momentum/momentum_signals.py 只有动量函数，
                        #  而模拟盘用的是 sim 侧的 signals → 又是一处两侧不一致，待单独排查）
}
VOL_WINDOW = 60                 # 权重估计窗口（交易日）
TARGET_VOL = 0.15               # 波动率目标化：目标年化波动
PERIODS = [("2024全年", "2024-01-02", "2024-12-31"),
           ("2025全年", "2025-01-02", "2025-12-31"),
           ("2026至今", "2026-01-05", "2026-09-30"),
           ("全周期",   "2024-01-02", "2026-09-30")]


# ─────────────────────── 曲线采集 ───────────────────────

def _pipeline_strategies() -> list[str]:
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    return [m for m in re.findall(r'"id":\s*"(\w+)"', src)
            if m not in ("sync", "backtest_align") and m not in EXCLUDE]


def _from_output(sid: str, start: str, end: str):
    """找一段**起点与请求一致**的回测（起点不同=不可比：动量类路径依赖，
    2021 起与 2024 起的同窗口结果可以差几十个百分点）。

    起点容差 10 自然日（请求 2024-01-01，实际数据首日是 2024-01-02）。
    """
    import glob
    lo_tol = (pd.Timestamp(start) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi_tol = (pd.Timestamp(end) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    for d in sorted(glob.glob(str(ROOT / f"strategies/{sid}/output/*/")), reverse=True):
        f = Path(d) / "daily_records.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty or "date" not in df.columns or "total_value" not in df.columns:
            continue
        first, last = str(df["date"].iloc[0])[:10], str(df["date"].iloc[-1])[:10]
        if first <= lo_tol and last >= hi_tol:
            return df[["date", "total_value"]].copy()
    return None


def _run_backtest(sid: str, start: str, end: str):
    if sid == "asset_allocation":       # 无 engine.py，走自己的 backtest 模块
        try:
            mod = __import__("strategies.asset_allocation.backtest", fromlist=["*"])
            with contextlib.redirect_stdout(io.StringIO()):
                close_df, _ = mod.load_data()
                nav = mod.backtest_equity_series(close_df, "risk_parity", start=start)
            if nav is None:
                return None
            df = nav.reset_index()
            df.columns = ["date", "total_value"][: len(df.columns)]
            return df
        except Exception as e:
            print(f"    ⚠ asset_allocation 失败: {type(e).__name__}: {e}")
            return None
    rp = ROOT / f"strategies/{sid}/run.py"
    if rp.exists() and sid not in FORCE_ENGINE:
        try:
            proc = subprocess.run([sys.executable, "-m", f"strategies.{sid}.run",
                                   "--start", start, "--end", end, "--tag", "alloc2026"],
                                  cwd=str(ROOT), capture_output=True, text=True, timeout=900)
        except Exception as e:
            print(f"    ⚠ {sid} run.py 失败: {e}")
            return None
        # 从 stdout 解析真实输出目录（neural_momentum 的 OUTPUT_DIR 指向 momentum 目录，
        # 不能按 strategies/<sid>/output 猜——与 backtest_align 同一手法）
        out_dir = None
        for line in (proc.stdout or "").splitlines():
            if "输出目录" in line:
                out_dir = line.split(":", 1)[1].strip()
        if out_dir:
            f = Path(out_dir) / "daily_records.csv"
            if f.exists():
                df = pd.read_csv(f)
                if not df.empty:
                    return df[["date", "total_value"]].copy()
        return _from_output(sid, start, end)
    # 无 run.py：直接驱动 engine（同接口）
    try:
        mod = __import__(f"strategies.{sid}.engine", fromlist=["*"])
        # 必须选**本模块定义**的引擎类（dir() 里还有 import 进来的基类 BacktestEngine，
        # 选错就会跑成 momentum，产出 7 条完全相同的曲线——2026-10-02 踩过）
        own = [n for n in dir(mod) if n.endswith("Engine")
               and isinstance(getattr(mod, n), type)
               and getattr(mod, n).__module__ == mod.__name__]
        if not own:
            print(f"    ⚠ {sid}: engine 模块内无自定义引擎类，跳过")
            return None
        eng_cls = getattr(mod, own[0])
        with contextlib.redirect_stdout(io.StringIO()):
            eng = eng_cls()
            eng.load_data(start_date=start, end_date=end)
            eng.run()
            df = eng.get_daily_df()
        return df[["date", "total_value"]].copy()
    except Exception as e:
        print(f"    ⚠ {sid} 直接驱动 engine 失败: {type(e).__name__}: {e}")
        return None


def collect(refresh: bool, start: str, end: str) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        print(f"  用缓存 {CACHE}（--refresh 可重跑）")
        return pd.read_csv(CACHE, dtype={"sid": str})
    sids = _pipeline_strategies()
    print(f"  采集 {len(sids)} 条策略回测曲线（排除 {sorted(EXCLUDE)}）…")
    frames = []
    for sid in sids:
        df = _from_output(sid, start, end)
        if df is None:
            df = _run_backtest(sid, start, end)
        if df is None or df.empty:
            print(f"    ❌ {sid}: 无曲线")
            continue
        df = df.rename(columns={"total_value": "tv"})
        df["sid"] = sid
        frames.append(df[["date", "sid", "tv"]])
        print(f"    ✓ {sid:<24}{len(df)} 天  {str(df['date'].iloc[0])[:10]} → {str(df['date'].iloc[-1])[:10]}")
    out = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


# ─────────────────────── 组合构建 ───────────────────────

def build_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """date × sid 的日收益矩阵（total_value 变化率）。"""
    rets = {}
    for sid, g in panel.groupby("sid"):
        s = g.sort_values("date").set_index("date")["tv"]
        s = pd.to_numeric(s, errors="coerce")
        s.index = pd.to_datetime(s.index)            # 统一索引类型（asset_allocation 为字符串）
        s = s[~s.index.duplicated(keep="last")]      # 个别曲线有重复日期行（如 gold 670 天）
        rets[sid] = s.pct_change()
    R = pd.DataFrame(rets)
    return R


def simulate_portfolio(R: pd.DataFrame, mode: str, vol_scale: bool = False) -> pd.Series:
    """按月再平衡的组合净值（各 sleeve 独立涨跌，漂移不日内再平衡）。

    mode: "ew" 等权 | "rp" 风险平价(1/σ) ；vol_scale: 再叠加波动率目标化。
    权重只用再平衡日**之前**的数据（无未来信息）。
    """
    dates = R.index
    vol = R.rolling(VOL_WINDOW, min_periods=20).std() * np.sqrt(252)
    value = 1.0
    sleeve_val: dict[str, float] = {}
    out = []
    month = None
    for i, d in enumerate(dates):
        r = R.iloc[i]
        avail = r.dropna()
        ym = str(d)[:7]
        if month is None or ym != month:
            month = ym
            if sleeve_val:                       # 收口上一期
                value = sum(sleeve_val.values())
            # 交易日首日/月末：用**上一行**的波动率（当日不可知）
            v = vol.iloc[i - 1] if i > 0 else vol.iloc[i]
            cand = [s for s in avail.index if not pd.isna(v.get(s, np.nan)) and v[s] > 0]
            if not cand:
                cand = list(avail.index)
            if mode == "rp":
                w = {s: 1.0 / v[s] for s in cand}
            else:
                w = {s: 1.0 for s in cand}
            tot = sum(w.values())
            # 首日收益全为 NaN → cand 为空 → 保持现金（否则净值会被算成 0 → inf）
            sleeve_val = ({s: value * w[s] / tot for s in cand} if cand and tot > 0 else {})
        for s in list(sleeve_val.keys()):
            if s in avail.index:
                sleeve_val[s] *= (1 + avail[s])
        out.append(sum(sleeve_val.values()) if sleeve_val else value)
    nav = pd.Series(out, index=dates)
    if vol_scale:
        r = nav.pct_change().fillna(0)
        realized = r.rolling(20, min_periods=10).std() * np.sqrt(252)
        scale = (TARGET_VOL / realized).clip(0.2, 1.5).shift(1).fillna(1.0)
        nav = (1 + r * scale).cumprod()
    return nav


def stats(nav: pd.Series, lo: str, hi: str):
    s = nav[(nav.index >= lo) & (nav.index <= hi)]
    if len(s) < 3:
        return None
    v = s.values
    r = v[1:] / v[:-1] - 1
    peak = np.maximum.accumulate(v)
    return {"ret": (v[-1] / v[0] - 1) * 100,
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "mdd": float(np.min(v / peak - 1) * 100), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()

    print(f"═══ 方向 3 第一步：策略层组合 | {a.start} → {a.end} ═══")
    panel = collect(a.refresh, a.start, a.end)
    R = build_returns(panel)
    R = R[(R.index >= a.start) & (R.index <= a.end)]
    print(f"  曲线 {R.shape[1]} 条 × {R.shape[0]} 天")

    # ── 相关性（分散化空间）──
    C = R.corr()
    iu = np.triu_indices_from(C.values, k=1)
    pair = C.values[iu]
    print(f"\n  两两相关：均值 {np.nanmean(pair):.2f} | 中位 {np.nanmedian(pair):.2f} "
          f"| 最低 {np.nanmin(pair):.2f} | 最高 {np.nanmax(pair):.2f}")

    # ── 组合与参照 ──
    navs = {
        "等权组合": simulate_portfolio(R, "ew"),
        "风险平价组合": simulate_portfolio(R, "rp"),
        "风险平价+波动率目标": simulate_portfolio(R, "rp", vol_scale=True),
    }
    single = {sid: (1 + R[sid].fillna(0)).cumprod() for sid in R.columns}

    def _single_by(stat_key: str, lo: str, hi: str) -> pd.Series:
        vals = {sid: stats(nav, lo, hi) for sid, nav in single.items()}
        vals = {k: v[stat_key] for k, v in vals.items() if v}
        return pd.Series(vals)

    print(f"\n{'口径':<20}{'期间':<10}{'收益%':>9}{'夏普':>7}{'MDD%':>8}")
    table = {}
    for label, nav in navs.items():
        table[label] = {}
        for plabel, lo, hi in PERIODS:
            st = stats(nav, lo, hi)
            table[label][plabel] = st
            if st:
                print(f"{label:<20}{plabel:<10}{st['ret']:>9.2f}{st['sharpe']:>7.2f}{st['mdd']:>8.2f}")
    print(f"\n{'单策略参照':<20}{'期间':<10}{'中位夏普':>10}{'最好夏普':>10}{'最差夏普':>10}{'中位MDD':>10}")
    refs = {}
    for plabel, lo, hi in PERIODS:
        sh = _single_by("sharpe", lo, hi)
        md = _single_by("mdd", lo, hi)
        refs[plabel] = {"sh_med": sh.median(), "sh_best": sh.max(), "sh_worst": sh.min(),
                        "mdd_med": md.median()}
        print(f"{'':<20}{plabel:<10}{sh.median():>10.2f}{sh.max():>10.2f}{sh.min():>10.2f}{md.median():>10.2f}")

    # ── 判据 ──
    print(f"\n═══ 判据（事前登记，对'全周期'）═══")
    full = refs["全周期"]
    for label in navs:
        st = table[label]["全周期"]
        f1 = "证伪" if not (st["sharpe"] > full["sh_med"]) else "通过"
        f2 = "证伪" if not (st["mdd"] > full["mdd_med"]) else "通过"
        print(f"  {label:<20} 夏普 {st['sharpe']:.2f} vs 单策略中位 {full['sh_med']:.2f} → F1 {f1}"
              f" | MDD {st['mdd']:.2f}% vs 中位 {full['mdd_med']:.2f}% → F2 {f2}")


if __name__ == "__main__":
    main()
