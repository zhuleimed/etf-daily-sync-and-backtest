#!/usr/bin/env python3
"""方向 1 第一步：宽池动量实验（2026-10-02）

## 假设（事前登记）
H1：把 momentum_rotation 的**同一套逻辑**从 7 只宽基扩到"按当日流动性动态筛选的宽池"，
    风险调整后表现（夏普）显著改善 → 值得继续做"扩池 + 多因子/ML"。

## 证伪判据（任一成立 = 证伪，按用户决定停止方向 1）
F1：宽池的**全周期夏普** ≤ 7 只宽基基线；
F2：4 个对比期间（2024 / 2025 / 2026 / 全周期）中，宽池在 **≥3 期**夏普 ≤ 基线。

## 方法（无未来信息）
- **日期轴** = 7 只宽基的共同交易日历（A 股标准日历）。不直接用
  `load_all_etf_data(宽池)`——那个函数取"所有标的共同交易日"，会把回测截断到
  最新上市那只的上市日。
- **单只数据**：用 `momentum_rotation.data._load_single_etf` 在标的**自身行序列**上
  算好动量/ATR/成交额（前 20 行为 NaN，天然无预上市污染），再 reindex 到统一日历。
- **逐日资格**（只用当日及之前数据）：`listed`(当日 close>0) & `hist_ok`(此前有效
  交易日 ≥ MIN_HISTORY) & `liq_ok`(过去 20 日日均成交额 ≥ 阈值)。
- **信号注入**：包装 engine 模块的 `compute_momentum_signals`，把不合格标的动量置 NaN
  （`rank_etfs_by_momentum` 会 dropna）。引擎其余全部复用：min_hold、切换置信度、
  短期动量确认、渐进调仓、风控模式、摩擦成本、涨跌停外的执行规则——**不改一行已验证代码**。
- **持仓豁免**：若持有标的当日不合格，仍保留其动量值，否则 `_make_decision_single`
  会因 current_mom=NaN 直接 return、卡住不换。

## 用法
    python -m simulation.analysis.wide_pool_momentum [--start 2024-01-01] [--end 2026-09-30]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strategies.momentum_rotation.engine as MOM_ENGINE              # noqa: E402
from strategies.momentum_rotation import config as C                  # noqa: E402
from strategies.momentum_rotation.data import (                       # noqa: E402
    _load_single_etf, load_all_etf_data, load_benchmark_data,
    compute_equal_weight_benchmark,
)
from strategies.momentum_rotation.engine import BacktestEngine        # noqa: E402

DB = str(ROOT / "data" / "etf_daily.db")
MIN_HISTORY = 60        # 要求：信号日前至少 60 个有效交易日
WARM_DAYS = 400         # 宽池数据预热（自然日）：必须覆盖 MIN_HISTORY 个交易日
LIQ_WINDOW = 20         # 流动性窗口（交易日）
TIERS = [("≥5亿", 5e8), ("≥1亿", 1e8), ("≥5000万", 5e7)]
PERIODS = [("2024全年", "2024-01-02", "2024-12-31"),
           ("2025全年", "2025-01-02", "2025-12-31"),
           ("2026至今", "2026-01-05", "2026-09-30"),
           ("全周期",   "2024-01-02", "2026-09-30")]


# ────────────────────────── 数据 ──────────────────────────

def curated_pool() -> list[str]:
    """定向池 = 本仓库既有策略配置池的并集（宽基/跨境/行业/避险/配置/低波）。

    关键：这些标的的选取**早于本次实验**（是为别的策略挑的），所以不存在"事后挑选"
    偏差；覆盖的资产类别 = 宽基+QDII+债券+黄金+豆粕+行业主题 → 正好检验
    "资产类别可及性"这个从宽池实验里发现的真机制。
    """
    import re
    confs = ["momentum_rotation", "cross_border", "sector_rotation", "industry_momentum",
             "gold_safe_haven", "asset_allocation", "low_vol_rotation", "asset_allocation"]
    codes: set[str] = set()
    for sid in confs:
        p = ROOT / "strategies" / sid / "config.py"
        if p.exists():
            # 只取 ETF 代码（1xxxxx/5xxxxx），排除指数代码（000xxx/399xxx）
            codes |= {c for c in re.findall(r'"(\d{6})"\s*:', p.read_text(encoding="utf-8"))
                      if c[0] in "15"}
    return sorted(codes)


def _candidate_symbols() -> list[str]:
    """宽池候选：历史上有效交易日 ≥ MIN_HISTORY+5 的所有标的（流动性由逐日掩码判定）。"""
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as c:
        rows = c.execute(
            "SELECT symbol, COUNT(*) n FROM etf_daily GROUP BY symbol HAVING n >= ?",
            (MIN_HISTORY + 5,),
        ).fetchall()
    return sorted(r[0] for r in rows)


SPLIT_THRESHOLD = 0.22   # |单日涨跌| 超此值 → 判为份额折算/拆分（ETF 有 ±10%/±20% 涨跌停）


def adjust_splits(df: pd.DataFrame, momentum_window: int) -> pd.DataFrame:
    """抹平未复权价格里的"份额折算/拆分"跳变，并重算全部派生列。

    背景：本项目 ETF 价格为**未复权**。宽池里 114 只标的出现过单日 |涨跌| > 22% 的跳变
    （最大 -67%、+256%），物理上不可能（ETF 有涨跌停），是份额折算/拆分。
    动量策略专挑涨幅榜第一，会被"假涨"吸引买入，再吃下折算后的"假暴跌"——2026-07-06
    持仓 588170 单日 -66.69% 即此。**这不是策略表现，是数据缺陷**（宽基 7 只这些年没有
    折算，所以项目一直没踩到）。

    做法（后复权）：逐日 ratio = close/prev_close，对 |ratio-1| > SPLIT_THRESHOLD 的日子，
    把该日**之前**的价格 ×ratio、成交量 ÷ratio（成交额不变）；随后重算
    pct_chg/累计/momentum/amount_ma20/ATR。不做分红复权（幅度小，且对各池影响一致）。
    """
    df = df.copy()
    close = df["close"].to_numpy(dtype=float)
    prev = np.r_[np.nan, close[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((prev > 0) & (close > 0), close / prev, 1.0)
    jump = np.abs(ratio - 1) > SPLIT_THRESHOLD
    if not jump.any():
        return df
    factor = np.ones(len(df))
    cum = 1.0
    for i in range(len(df) - 1, -1, -1):      # 从后往前：跳变日之前的历史乘上 ratio
        factor[i] = cum
        if jump[i]:
            cum *= ratio[i]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].to_numpy(dtype=float) * factor
    df["volume"] = df["volume"].to_numpy(dtype=float) / factor
    # ── 重算派生列（与 momentum_rotation.data._load_single_etf 同口径）──
    df["pct_chg"] = df["close"].pct_change().fillna(0.0)
    df["cumulative_returns"] = (1 + df["pct_chg"]).cumprod()
    df.loc[df.index[0], "cumulative_returns"] = 1.0
    df["amount"] = df["close"] * df["volume"]
    df["amount_ma20"] = df["amount"].rolling(20).mean().bfill().fillna(df["amount"])
    prev_close = df["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(20).mean().bfill().fillna(df["tr"])
    df["momentum"] = df["close"] / df["close"].shift(momentum_window) - 1
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    return df


def build_wide_pool(ext_dates: pd.DatetimeIndex, engine_dates: pd.DatetimeIndex,
                    start: str, momentum_window: int, symbols: list[str] | None = None):
    """把宽池每只标的对齐到统一日历。

    关键：**资格矩阵在"扩展日历"上算**（含 start 前的预热期），否则回测前 60 个交易日
    会因"上市未满 60 日"而无人可选（早期被迫空仓，污染 2024 期间的对比）。
    引擎用的帧则裁剪回 engine_dates，保证与引擎的位置索引一一对应。

    返回 (etf_data, 资格用矩阵包)；矩阵在 ext_dates 上，调用方 reindex 到 engine_dates。
    """
    symbols = symbols if symbols is not None else _candidate_symbols()
    # 预热必须 ≥ MIN_HISTORY 个**交易日**（60 交易日 ≈ 90 自然日），取 400 自然日留足余量：
    # 否则回测首日所有标的都"上市未满 60 日"，早期被迫空仓。
    lo = (pd.to_datetime(start) - timedelta(days=WARM_DAYS)).strftime("%Y-%m-%d")
    frames, close_m, amt_m = {}, {}, {}
    t0 = time.time()
    for i, sym in enumerate(symbols):
        df = _load_single_etf(sym, lo, "", DB, momentum_window)
        if df is None or len(df) <= momentum_window:
            continue
        df = adjust_splits(df, momentum_window)      # 抹平份额折算/拆分假跳变
        df = df.set_index("date")
        f = pd.DataFrame(index=ext_dates)
        for col in ("open", "high", "low", "close", "volume", "amount", "amount_ma20",
                    "atr", "momentum", "pct_chg", "cumulative_returns"):
            f[col] = df[col].reindex(ext_dates)
        # 价格类前向填充（停牌日沿用最近价）；动量不填（NaN = 不可选）
        for col in ("open", "high", "low", "close", "volume", "amount", "amount_ma20", "atr"):
            f[col] = f[col].ffill().fillna(0.0)
        f["pct_chg"] = f["pct_chg"].fillna(0.0)          # 未上市日收益记 0（供等权基准）
        f["cumulative_returns"] = f["cumulative_returns"].ffill().fillna(1.0)
        close_m[sym] = f["close"]
        amt_m[sym] = f["amount"]
        # 引擎帧：加 date 列（等权基准函数需要）+ 裁剪到引擎日历 + RangeIndex
        # （引擎按**位置**索引取数，`df.loc[idx]` 需要 0..n-1 的整数标签）
        f["date"] = ext_dates
        g = f.reindex(engine_dates).reset_index(drop=True)
        g["date"] = engine_dates
        frames[sym] = g
        if (i + 1) % 400 == 0:
            print(f"    已加载 {i+1}/{len(symbols)} 只（{time.time()-t0:.0f}s）")
    close_df = pd.DataFrame(close_m, index=ext_dates)
    amt_df = pd.DataFrame(amt_m, index=ext_dates)
    valid = (close_df > 0).cumsum()                       # 累计有效交易日（扩展日历）
    liq20 = amt_df.rolling(LIQ_WINDOW, min_periods=LIQ_WINDOW).mean()
    return frames, close_df, valid, liq20


# ────────────────────────── 回测 ──────────────────────────

def run_engine(etf_data, dates, mask: pd.DataFrame | None, label: str,
               start: str, end: str) -> dict:
    eng = BacktestEngine(initial_capital=C.INITIAL_CAPITAL)
    scope = [s for s, d in etf_data.items() if s in C.ETF_POOL or not d.empty]
    eng.etf_data = etf_data
    eng.dates = dates
    eng.etf_benchmark_data = {}
    try:
        eng.benchmark_data = load_benchmark_data(
            start_date=start, end_date=end, db_path=DB, momentum_window=C.MOMENTUM_WINDOW)
    except ValueError:
        eng.benchmark_data = pd.DataFrame()
    eng.equal_weight_data = compute_equal_weight_benchmark(etf_data)

    orig = MOM_ENGINE.compute_momentum_signals
    if mask is not None:

        def masked(etf_data_, signal_idx, window, _orig=orig, _mask=mask, _eng=eng):
            mom = _orig(etf_data_, signal_idx, window)
            if signal_idx >= len(_mask):
                return mom
            elig = _mask.iloc[signal_idx]
            hold = _eng._get_hold_symbol()
            bad = [s for s in mom.index if not bool(elig.get(s, False)) and s != hold]
            mom[bad] = np.nan
            return mom

        MOM_ENGINE.compute_momentum_signals = masked
    # 引擎日循环用模块级 ETF_SYMBOLS 组装 today_data（第 227/693 行硬编码），
    # 换池（宽池/定向池）必须临时替换为池内标的；运行完还原（不改已验证代码）。
    orig_symbols = MOM_ENGINE.ETF_SYMBOLS
    MOM_ENGINE.ETF_SYMBOLS = list(etf_data.keys())
    try:
        t0 = time.time()
        eng.run()
        print(f"    （{label} 耗时 {time.time()-t0:.0f}s，{len(scope)} 只标的）")
    finally:
        MOM_ENGINE.compute_momentum_signals = orig
        MOM_ENGINE.ETF_SYMBOLS = orig_symbols

    df = eng.get_daily_df()
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    n_trades = len(eng.trade_records)
    turnover = float(pd.to_numeric(eng.get_trade_df()["amount"], errors="coerce").sum()) if n_trades else 0.0
    return {"label": label, "daily": df, "n_trades": n_trades, "turnover": turnover,
            "avg_pool": float(mask.iloc[:, :].sum(axis=1).mean()) if mask is not None else np.nan}


def period_stats(daily: pd.DataFrame, lo: str, hi: str) -> dict | None:
    d = daily[(daily["date"] >= lo) & (daily["date"] <= hi)]
    v = pd.to_numeric(d["total_value"], errors="coerce").dropna().values
    if len(v) < 2:
        return None
    r = v[1:] / v[:-1] - 1
    peak = np.maximum.accumulate(v)
    return {"ret": (v[-1] / v[0] - 1) * 100,
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "mdd": float(np.min(v / peak - 1) * 100), "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--pool", choices=["wide", "curated"], default="wide",
                    help="wide=全市场宽池（按流动性分档）；curated=定向池（既有策略配置并集）")
    a = ap.parse_args()

    print(f"═══ {'定向池' if a.pool == 'curated' else '宽池'}动量实验 | {a.start} → {a.end} ═══")
    # 扩展窗口加载：资格判定需要 start 之前的预热（否则前 60 日无人可选）
    lo = (pd.to_datetime(a.start) - timedelta(days=WARM_DAYS)).strftime("%Y-%m-%d")
    base_ext, ext_dates = load_all_etf_data(
        symbols=C.ETF_SYMBOLS, start_date=lo, end_date=a.end, db_path=DB)
    keep = ext_dates >= pd.Timestamp(a.start)
    dates = ext_dates[keep]
    base_data = {s: d[d["date"] >= pd.Timestamp(a.start)].reset_index(drop=True)
                 for s, d in base_ext.items()}
    print(f"  日期轴（7 只宽基共同日历）: {dates[0].date()} → {dates[-1].date()} 共 {len(dates)} 天")

    if a.pool == "curated":
        syms = curated_pool()
        print(f"\n[1/3] 定向池 {len(syms)} 只（既有策略配置并集，选取早于本实验）:")
        print(f"      {' '.join(syms)}")
        frames, close_df, valid, liq20 = build_wide_pool(
            ext_dates, dates, a.start, C.MOMENTUM_WINDOW, symbols=syms)
        print(f"\n[2/3] 跑基线（7 只宽基，同引擎同窗口）")
        base = run_engine(base_data, dates, None, "基线·7只宽基", a.start, a.end)
        print(f"\n[3/3] 跑定向池（标的均为既定策略在用，不加流动性掩码）")
        results = [base, run_engine(frames, dates, None, "定向池(37只)", a.start, a.end)]
    else:
        print(f"\n[1/3] 加载宽池数据…")
        frames, close_df, valid, liq20 = build_wide_pool(ext_dates, dates, a.start, C.MOMENTUM_WINDOW)
        print(f"  候选 {len(frames)} 只（有效交易日 ≥ {MIN_HISTORY+5}）")
        print(f"\n[2/3] 跑基线（7 只宽基，同引擎同窗口）")
        base = run_engine(base_data, dates, None, "基线·7只宽基", a.start, a.end)
        print(f"\n[3/3] 跑宽池各流动性档")
        results = [base]
        for name, thr in TIERS:
            mask_ext = (close_df > 0) & (valid >= MIN_HISTORY) & (liq20 >= thr)
            mask = mask_ext.reindex(dates).fillna(False)
            print(f"  ── 档位 {name}: 平均可选 {mask.sum(axis=1).mean():.0f} 只"
                  f"（首日 {mask.iloc[0].sum()} / 末日 {mask.iloc[-1].sum()}）")
            results.append(run_engine(frames, dates, mask, f"宽池{name}", a.start, a.end))

    # ── 报告 ──
    print(f"\n{'口径':<14}{'期间':<10}{'收益%':>9}{'夏普':>7}{'MDD%':>8}{'交易笔数':>9}{'平均可选':>9}")
    table = {}
    for res in results:
        table[res["label"]] = {}
        for plabel, lo, hi in PERIODS:
            st = period_stats(res["daily"], lo, hi)
            table[res["label"]][plabel] = st
            if st:
                print(f"{res['label']:<14}{plabel:<10}{st['ret']:>9.2f}{st['sharpe']:>7.2f}"
                      f"{st['mdd']:>8.2f}{res['n_trades']:>9d}{res['avg_pool']:>9.0f}")

    # ── 判据 ──
    print(f"\n═══ 判据（事前登记）═══")
    b = table["基线·7只宽基"]
    for label in [r["label"] for r in results if r["label"] != "基线·7只宽基"]:
        w = table[label]
        worse = [p for p, _, _ in PERIODS
                 if w[p] and b[p] and not (w[p]["sharpe"] > b[p]["sharpe"])]
        full_ok = w["全周期"] and b["全周期"] and w["全周期"]["sharpe"] > b["全周期"]["sharpe"]
        f1 = "证伪" if not full_ok else "通过"
        f2 = "证伪" if len(worse) >= 3 else "通过"
        print(f"  {label}: 全周期夏普 {w['全周期']['sharpe']:.2f} vs 基线 {b['全周期']['sharpe']:.2f}"
              f" → F1 {f1} | 不占优期间 {len(worse)}/4（{','.join(worse) or '无'}）→ F2 {f2}")


if __name__ == "__main__":
    main()
