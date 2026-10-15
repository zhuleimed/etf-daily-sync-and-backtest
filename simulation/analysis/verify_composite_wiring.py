#!/usr/bin/env python3
"""
composite_momentum 两处模拟盘接线缺口的离线重放验证（2026-10-16）

背景（对齐监控 v2 + 代码复核发现，尚未修）：
  W1 **regime 按"数组位置"取** —— daily.py 把 `today_idx`（ETF 数组位置）直接传给
     `judge_market_regime(index_data, today_idx)`，而它内部用 `iloc[date_idx]`；
     ETF 窗口取 D-120 天、指数窗口取 D-320 天（loader 自己又往前多取 180 天）
     ⇒ **错位约 130 行（≈6.5 个月）**，判牛熊用的是半年前的行情。
     对照：adx_trend_rotation 用 `refresh_regime()` 按日期定位（2026-10-02 修对）。
  W2 **切换阈值口径差** —— 回测 `_make_decision` 用 `max(截面标准差×0.5, 0.1)`
     （近日 0.15~0.25）；模拟盘 config 里 `MIN_SWITCH_CONVICTION=0.03`（照抄动量模板）
     ⇒ **松 5~8 倍**，模拟盘会明显更爱换手。

本脚本用真实 DailySimEngine 离线重放回答（同一路径、单变量对照）：
  A. harness 自检：现状口径（W1 错位 + W2 0.03）能否逐日复现 live？（证明重放可信）
  B. 4 期对比：现状 vs 只修 W1 vs 只修 W2 vs 全修 —— 收益/夏普/MDD
     以及"与回测的持仓一致率 + 末日缺口"
  C. 两口径的 regime 判定差异天数（证明 W1 不是死分支）

用法：
  python -m simulation.analysis.verify_composite_wiring            # 自检 + 4 期对比
  python -m simulation.analysis.verify_composite_wiring --selfcheck-only
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.analysis.repair_bt import _dates, _isolation, _load_range, _pos  # noqa: E402
from simulation.framework.broker import SimBroker  # noqa: E402
from simulation.framework.engine import DailySimEngine  # noqa: E402
from simulation.framework.state import StateManager  # noqa: E402
from simulation.framework.backtest_align import align  # noqa: E402
from simulation.strategies.composite_momentum.config import (  # noqa: E402
    ETF_POOL, ETF_SYMBOLS, MOMENTUM_WINDOW, MIN_SWITCH_CONVICTION, MIN_HOLD_DAYS,
    COMMISSION_RATE, SLIPPAGE, INITIAL_CAPITAL, RISK_MODE, STOP_LOSS_PCT,
    PROFIT_THRESHOLD, DRAWBACK_PCT, DRAWDOWN_THRESHOLD, MARKET_INDEX, MARKET_MA_PERIOD,
    DB_PATH,
)
from strategies.composite_momentum.config import SWITCH_CONVICTION_STD  # noqa: E402
from strategies.composite_momentum.momentum_signals import (  # noqa: E402
    compute_composite_score, compute_composite_spread, judge_market_regime,
    rank_etfs_by_composite,
)

# 模拟盘 live 的数据窗口（daily.py 的 load_etf_and_index_data / load_latest_data 配方）
ETF_LOOKBACK_DAYS = 90 + MOMENTUM_WINDOW + 10          # ETF 窗口 = D-120 天
IDX_LOOKBACK_DAYS = 90 + MOMENTUM_WINDOW + 30          # 指数窗口 = D-140 天
# 注：daily.py 是直接 SQL 查 index_daily（date >= today-140d），不走策略的
#     load_index_data（那个会在内部再往前多取 180 天）。实测错位 = 14 个交易日
#     （ETF 79 行、指数 93 行），即判牛熊用的是 4 周前的行情。

PERIODS = [
    ("2024全年", "2024-01-02", "2024-12-31"),
    ("2025全年", "2025-01-02", "2025-12-31"),
    ("2026至今", "2026-01-05", "2026-09-30"),
    ("全周期", "2024-01-02", "2026-09-30"),
]


def _load_index(lo: str, hi: str) -> pd.DataFrame:
    """指数(沪深300)日线，只读、按日期升序。"""
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
            conn, params=[MARKET_INDEX, lo, hi],
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


def _mk_regime_pos(etf_dates: list[str], idx: pd.DataFrame):
    """返回两个"第 k 天的指数行号"函数：live 现状（错位）/ 按日期（正确）。

    live 现状的错位量 = 指数窗口行数 - ETF 窗口行数（两个窗口都从 D 往回按自然日取）：
      ETF  窗口 [D-120d, D] → L_etf 行；指数窗口 [D-320d, D] → L_idx 行；
      `iloc[today_idx=L_etf-1]` 落在指数窗口第 L_etf 行 = 今日往前 (L_idx-L_etf) 行。
    """
    idx_dates = pd.to_datetime(idx["date"]).dt.strftime("%Y-%m-%d").tolist()
    pos_by_date = {d: i for i, d in enumerate(idx_dates)}

    def _last_idx_on_or_before(day: str) -> int:
        return int(np.searchsorted(np.array(idx_dates), day, side="right")) - 1

    def _rows_between(day_from: str, day_to: str, dates_sorted: list[str]) -> int:
        a = int(np.searchsorted(np.array(dates_sorted), day_from, side="left"))
        b = int(np.searchsorted(np.array(dates_sorted), day_to, side="right"))
        return b - a

    def pos_stale(k: int) -> int:
        d = etf_dates[k]
        etf_from = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=ETF_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        idx_from = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=IDX_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        l_etf = _rows_between(etf_from, d, etf_dates)
        l_idx = _rows_between(idx_from, d, idx_dates)
        return max(0, _last_idx_on_or_before(d) - (l_idx - l_etf))

    def pos_fixed(k: int) -> int:
        d = etf_dates[k]
        return pos_by_date.get(d, _last_idx_on_or_before(d))

    return pos_stale, pos_fixed, idx_dates


def _signal_func(idx: pd.DataFrame, pos_fn):
    """信号函数：得分按指定 regime 缩放（与 compute_composite_score 内部口径一致）。"""
    def _sig(etf_data, today_idx, *a, **k):
        scores = compute_composite_score(etf_data, today_idx, index_data=None,
                                         market_ma_period=MARKET_MA_PERIOD)
        regime = judge_market_regime(idx, pos_fn(today_idx), MARKET_MA_PERIOD)["regime"]
        if regime == "bear":
            scores = scores * 0.8
        elif regime == "bull":
            scores = scores * 1.1
        return scores
    return _sig


def _thr_live(momentum, target_etf, hold_sym):
    """模拟盘现状：绝对阈值 0.03（照抄动量模板）。"""
    return MIN_SWITCH_CONVICTION


def _thr_backtest(momentum, target_etf, hold_sym):
    """回测口径：max(截面标准差 × SWITCH_CONVICTION_STD, 0.1)。"""
    return max(compute_composite_spread(momentum) * SWITCH_CONVICTION_STD, 0.1)


def _new_engine(state_mgr, thr_func, signal_func, risk_mode: str = ""):
    """与 live daily.py 同款接线，仅替换 switch_threshold_func / signal_func / risk_mode。"""
    broker = SimBroker(state_mgr, commission_rate=COMMISSION_RATE, slippage=SLIPPAGE)
    return DailySimEngine(
        state_mgr=state_mgr, broker=broker, config={"initial_capital": INITIAL_CAPITAL},
        signal_func=signal_func, rank_func=rank_etfs_by_composite, etf_pool=ETF_POOL,
        momentum_window=MOMENTUM_WINDOW,
        min_switch_conviction=MIN_SWITCH_CONVICTION, min_hold_days=MIN_HOLD_DAYS,
        risk_mode=risk_mode or RISK_MODE,
        stop_loss_pct=STOP_LOSS_PCT, profit_threshold=PROFIT_THRESHOLD,
        drawback_pct=DRAWBACK_PCT, drawdown_threshold=DRAWDOWN_THRESHOLD,
        switch_threshold_func=thr_func,
    )


def _run(seed: str, end: str, fix_regime: bool, fix_threshold: bool, risk_mode: str = ""):
    """重放一个变体（seed 起 fresh 现金）。返回 (dates, rets%, holds, rows)。"""
    lo = (datetime.strptime(seed, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d")
    etf = _load_range(ETF_SYMBOLS, lo, end, MOMENTUM_WINDOW)
    dates = _dates(etf[ETF_SYMBOLS[0]])
    i0 = _pos(etf[ETF_SYMBOLS[0]], seed)
    # 指数：多取一段以保证"错位读取"也有足够的行（错位最多 ~130 行）
    idx = _load_index((datetime.strptime(lo, "%Y-%m-%d") - timedelta(days=250)).strftime("%Y-%m-%d"), end)
    pos_stale, pos_fixed, _ = _mk_regime_pos(dates, idx)
    sig = _signal_func(idx, pos_fixed if fix_regime else pos_stale)
    thr = _thr_backtest if fix_threshold else _thr_live

    rows = []
    with tempfile.TemporaryDirectory(prefix="compfix_") as tmp:
        _isolation(tmp)
        sm = StateManager(tmp, "composite_momentum_v")
        sm.init_new(INITIAL_CAPITAL)
        eng = _new_engine(sm, thr, sig, risk_mode=risk_mode)
        for k in range(i0, len(dates)):
            d = dates[k]
            if d < seed:
                continue
            if d > end:
                break
            rep = eng.run_daily(etf, k, d)
            if "error" in rep:
                raise RuntimeError(f"{d}: {rep['error']}")
            rows.append({"date": d, "total": rep["total_value"], "hold": rep.get("hold_symbol") or ""})
    cap = INITIAL_CAPITAL
    return ([r["date"] for r in rows], [(r["total"] / cap - 1) * 100 for r in rows],
            [r["hold"] for r in rows], rows)


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


def _backtest_csv():
    """全周期回测基准（2024-01-01 ~ 2026-09-30，由本脚本首次运行时自动生成）。"""
    import glob
    hits = sorted(glob.glob(str(PROJECT_ROOT / "strategies" / "composite_momentum" / "output" / "*_wirecheck")))
    if not hits:
        sys.exit("缺少回测基准，先跑: python -m strategies.composite_momentum.run "
                 "--start 2024-01-01 --end 2026-09-30 --tag wirecheck")
    return Path(hits[-1]) / "daily_records.csv"


def selfcheck() -> bool:
    """A. harness 自检：现状口径重放 10-08~10-15，与 live CSV 逐日比总资产。"""
    live_csv = PROJECT_ROOT / "simulation" / "output" / "sim_log_composite_momentum.csv"
    live = {}
    for _, r in pd.read_csv(live_csv, dtype={"持仓标的": str}).iterrows():
        d = str(r.get("日期", "")).strip()[:10]
        if not d or "重启" in str(r.get("操作", "")):
            continue
        try:
            live[d] = float(str(r.get("总资产", "")).strip())
        except ValueError:
            pass
    _, _, _, rows = _run("2026-10-08", "2026-10-15", fix_regime=False, fix_threshold=False)
    rep = {r["date"]: r["total"] for r in rows}
    shared = sorted(d for d in rep if d in live)
    bad = [d for d in shared if abs(rep[d] - live[d]) > 0.5]
    print(f"[A 自检] 与 live 共有 {len(shared)} 天（{shared[0]}~{shared[-1]}）："
          + ("逐日复现 ✓" if not bad else f"❌ 分叉于 {bad[:3]}（重放不可信，先查口径）"))
    if bad:
        for d in bad[:5]:
            print(f"    {d}: 重放 {rep[d]:.2f} vs live {live[d]:.2f}")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck-only", action="store_true")
    a = ap.parse_args()

    print("═══ composite_momentum 接线缺口验证（W1 regime 错位 / W2 阈值口径）═══\n")
    ok = selfcheck()
    if a.selfcheck_only:
        return
    if not ok:
        sys.exit("自检未通过，终止（结果不可信）")

    back = pd.read_csv(_backtest_csv(), dtype={"hold_symbol": str}).fillna({"hold_symbol": ""})

    # C. W1 的 regime 判定差异（证明不是死分支）
    lo = "2024-01-02"
    etf = _load_range(ETF_SYMBOLS, (datetime.strptime(lo, "%Y-%m-%d") - timedelta(days=400)).strftime("%Y-%m-%d"), "2026-09-30", MOMENTUM_WINDOW)
    dates = _dates(etf[ETF_SYMBOLS[0]])
    idx = _load_index((datetime.strptime(lo, "%Y-%m-%d") - timedelta(days=650)).strftime("%Y-%m-%d"), "2026-09-30")
    pos_stale, pos_fixed, _ = _mk_regime_pos(dates, idx)
    k0 = _pos(etf[ETF_SYMBOLS[0]], lo)
    diff = sum(1 for k in range(k0, len(dates))
               if judge_market_regime(idx, pos_stale(k), MARKET_MA_PERIOD)["regime"]
               != judge_market_regime(idx, pos_fixed(k), MARKET_MA_PERIOD)["regime"])
    print(f"\n[C] W1 错位口径 vs 按日期：{lo}~2026-09-30 共 {len(dates)-k0} 天中 "
          f"{diff} 天 regime 判定不同（{diff/(len(dates)-k0):.0%}）")

    # B. 4 期 × 4 变体
    print(f"\n[B] 4 期对比（fresh 现金起步于各期起点）")
    print(f"{'期间':<10}{'口径':<16}{'收益%':>9}{'夏普':>7}{'MDD%':>8}{'与回测一致率':>13}{'末日缺口pp':>11}{'换仓次数':>9}")
    for label, seed, end in PERIODS:
        variants = [
            ("现状(W1错+W2松)", False, False),
            ("只修W1(regime)", True, False),
            ("只修W2(阈值)", False, True),
            ("全修", True, True),
        ]
        for tag, fx_r, fx_t in variants:
            dates_p, rets, holds, rows = _run(seed, end, fx_r, fx_t)
            st = _stats([r["total"] for r in rows])
            recs = align("composite_momentum", dates_p, rets, holds, back)
            m = sum(1 for r in recs if r["sim_hold"] == r["back_hold"])
            gap = recs[-1]["dev"] if recs else float("nan")
            switches = sum(1 for i in range(1, len(holds)) if holds[i] != holds[i - 1])
            print(f"{label:<10}{tag:<16}{st['ret']:>9.2f}{st['sharpe']:>7.2f}{st['mdd']:>8.2f}"
                  f"{m/len(recs)*100:>12.0f}%{gap:>11.2f}{switches:>9d}")
        print()


if __name__ == "__main__":
    main()
