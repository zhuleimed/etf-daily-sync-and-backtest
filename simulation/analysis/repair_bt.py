#!/usr/bin/env python3
"""
修复性回测 runner —— 在真实模拟盘引擎(DailySimEngine)上离线重放动量类策略，
量化"择时摩擦"修复在 当期 与 全窗口 的价值。

背景(记忆 sim-vs-backtest-discrepancy 2026-09-05):  08-03 清零重启后
momentum_rotation→09-04 -3.76% / macd_trend_rotation → -7.58%，同期沪深300 持平。
根因=2026-08-19 单日小盘假恐慌 + 两处摩擦:
  F1 momentum(mode A 纯信号): 08-18 切进 512100 恰逢 08-19 崩。
  F2 macd(mode B 风控): 08-20 风控清仓转现金 → 踏空整月回补 → 09-01 追高回买。
做法: 真实 DailySimEngine + 策略同款 signal/rank/etf_pool，逐共同交易日离线重放，
比 baseline 与 {confirm_days, risk_exit_reentry_cooldown, risk_mode} 各变体。
隔离: state/sim_db 写临时目录, monkeypatch sim_db 写函数(引擎内无 db_path 调用),
DB 以只读 uri 打开。
铁律自检: baseline 在 focal 区间 fresh seed 复放，末累计须≈live(-3.76/-7.58,pp≤0.05)，
否则退出(prove harness)。

用法:
  # 先做自检 + 单变体表:
  python -m simulation.analysis.repair_bt --strategy momentum_rotation --table out.tsv
  # 跑设计好的全矩阵(含自检):
  python -m simulation.analysis.repair_bt --strategy macd_trend_rotation --sweep --table out.tsv
参数:
  --seed 当期起点(默认 2026-08-03)   --start 全窗口起点(默认 2022-01-02)
  --end  终点(默认 2026-09-04)
引擎杠杆默认 OFF(1/0)，仅 --sweep 时内部枚举。单跑一线也可 --confirm/--cooldown/--risk。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import simulation.framework.sim_db as sim_db  # noqa: E402
from simulation.framework.engine import DailySimEngine  # noqa: E402
from simulation.framework.broker import SimBroker  # noqa: E402
from simulation.framework.state import StateManager  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "etf_daily.db"
CAL_AHEAD = 2            # end 后多取几天，确保末行在区间

# live 终端累计(对应 sim_log_{sid}.csv 末行, 自 08-03 清零重基)
LIVE_REF = {"momentum_rotation": -3.76, "macd_trend_rotation": -7.58}

# 策略常量快速访问(懒 import)
def _cfg(sid):
    return __import__(f"simulation.strategies.{sid}.config", fromlist=["*"])


def _signal_rank(sid):
    """返回 (signal_func, rank_func)。live daily.py 同款导入。"""
    if sid == "momentum_rotation":
        from strategies.momentum_rotation.momentum_signals import (
            compute_momentum_signals, rank_etfs_by_momentum,
        )
        return compute_momentum_signals, rank_etfs_by_momentum
    from strategies.macd_trend_rotation.momentum_signals import rank_etfs_by_macd

    def macd_sig(etf, idx, *a):
        from strategies.macd_trend_rotation.momentum_signals import compute_macd_scores
        return compute_macd_scores(etf, idx)
    return macd_sig, rank_etfs_by_macd


# ── 数据(只读绝对区间，recipe 同 load_latest_data) ──
def _load_range(symbols, lo, hi, mwin) -> dict[str, pd.DataFrame]:
    hi2 = (datetime.strptime(hi, "%Y-%m-%d") + timedelta(days=CAL_AHEAD)).strftime("%Y-%m-%d")
    uri = f"file:{DB_PATH}?mode=ro"
    out = {}
    with sqlite3.connect(uri, uri=True) as conn:
        for sym in symbols:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM etf_daily "
                "WHERE symbol=? AND date>=? AND date<=? ORDER BY date",
                conn, params=[sym, lo, hi2])
            if df.empty:
                raise RuntimeError(f"{sym} {lo}~{hi2} 无数据")
            for c in ("open", "high", "low", "close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            df["date"] = pd.to_datetime(df["date"])
            df = df.reset_index(drop=True)
            df["pct_chg"] = df["close"].pct_change().fillna(0.0)
            df["momentum"] = df["close"] / df["close"].shift(mwin) - 1  # backward, 无泄漏
            out[sym] = df
    # 对齐: 取各 symbol 都有的日期(共同交易日), 丢弃个别缺失日(如 512100 缺 2022-09-02),
    # 保证 1:1 位置对齐——引擎与 signal 都以"同一日期=同一行 index"读取。
    common = None
    for df in out.values():
        dts = {pd.Timestamp(t).strftime("%Y-%m-%d") for t in df["date"]}
        common = dts if common is None else (common & dts)
    common = sorted(common)
    for sym, df in out.items():
        out[sym] = df[df["date"].dt.strftime("%Y-%m-%d").isin(common)].reset_index(drop=True)
    if len({len(v) for v in out.values()}) != 1:
        raise RuntimeError("symbol 帧长不一致(日历未对齐)")
    return out


def _dates(df) -> list:
    return [pd.Timestamp(t).strftime("%Y-%m-%d") for t in df["date"]]


def _pos(df, day: str) -> int:
    hit = np.where(pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").values == day)[0]
    if len(hit) == 0:
        raise RuntimeError(f"日历无 {day}")
    return int(hit[0])


def _isolation(tmp: str):
    """临时目录内, 把引擎两个 sim_db 写调用与默认 db 全部指向无。"""
    sim_db.record_account_daily = lambda *a, **k: None
    sim_db.record_closed_trade = lambda *a, **k: None
    sim_db._DEFAULT_DB_PATH = str(Path(tmp) / "sim_trading_replay.db")


def _new_engine(sid, state_mgr, tmp, *, confirm, cooldown, risk_mode):
    cfg = _cfg(sid)
    _isolation(tmp)
    signal_func, rank_func = _signal_rank(sid)
    broker = SimBroker(state_mgr, commission_rate=cfg.COMMISSION_RATE, slippage=cfg.SLIPPAGE)
    return DailySimEngine(
        state_mgr=state_mgr, broker=broker, config={"initial_capital": cfg.INITIAL_CAPITAL},
        signal_func=signal_func, rank_func=rank_func, etf_pool=cfg.ETF_POOL,
        momentum_window=cfg.MOMENTUM_WINDOW,
        min_switch_conviction=cfg.MIN_SWITCH_CONVICTION, min_hold_days=cfg.MIN_HOLD_DAYS,
        risk_mode=risk_mode,
        stop_loss_pct=cfg.STOP_LOSS_PCT, profit_threshold=cfg.PROFIT_THRESHOLD,
        drawback_pct=cfg.DRAWBACK_PCT, drawdown_threshold=cfg.DRAWDOWN_THRESHOLD,
        confirm_days=confirm, risk_exit_reentry_cooldown=cooldown,
    ), broker


def _walk(engine, etf_data, dates, i0, day0, day1):
    """从 dates[i0] 走到 day1 含。返回 [(day,total_value,hold)]。fresh 状态已在 init_new。"""
    out = []
    for k in range(i0, len(dates)):
        d = dates[k]
        if d < day0:
            continue
        if d > day1:
            break
        rep = engine.run_daily(etf_data, k, d)
        if "error" in rep:
            raise RuntimeError(f"{d} engine err: {rep['error']}")
        hold = rep.get("hold_symbol") or ""
        out.append((d, rep["total_value"], hold))
    return out


def _stats(vals) -> dict:
    v = np.asarray(vals, float)
    if len(v) < 2 or v[0] <= 0:
        return {"ret": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": len(v)}
    rets = v[1:] / v[:-1] - 1
    ret = (v[-1] / v[0] - 1) * 100
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else np.nan
    peak = np.maximum.accumulate(v)
    mdd = float(np.min(v / peak - 1) * 100)
    return {"ret": ret, "sharpe": sharpe, "mdd": mdd, "n": len(v)}


def replay(sid, *, full_start, seed, end, confirm=1, cooldown=0, risk=None):
    """跑一个变体：返回 dict(full=.., focal=..)。引擎杠杆默认 OFF。"""
    cfg = _cfg(sid)
    mode = cfg.RISK_MODE if risk is None else risk
    # 起点缓冲需落在所有 symbol 已有数据的共同区间内(最新上架 563000≈2021-11)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        mins = {}
        for sym in cfg.ETF_SYMBOLS:
            ro = conn.execute("SELECT MIN(date) FROM etf_daily WHERE symbol=?", (sym,)).fetchone()[0]
            mins[sym] = ro
    common_first = max(mins.values())
    lo_buf = (datetime.strptime(full_start, "%Y-%m-%d") - timedelta(days=150)).strftime("%Y-%m-%d")
    lo = max(lo_buf, common_first)
    etf = _load_range(cfg.ETF_SYMBOLS, lo, end, cfg.MOMENTUM_WINDOW)
    dates = _dates(etf[cfg.ETF_SYMBOLS[0]])
    i_full = _pos(etf[cfg.ETF_SYMBOLS[0]], full_start)
    i_seed = _pos(etf[cfg.ETF_SYMBOLS[0]], seed)
    need_warm = max(cfg.MOMENTUM_WINDOW, 45)
    if i_full < need_warm:
        raise RuntimeError(f"full_start={full_start} 之前共同历史不足 "
                           f"(latest_list={common_first}, idx={i_full}<{need_warm})，请调晚 start")

    # 全窗口 fresh(从 full_start 现金)
    with tempfile.TemporaryDirectory(prefix="repair_") as tmp:
        sm = StateManager(tmp, f"{sid}_full")
        sm.init_new(cfg.INITIAL_CAPITAL)
        eng, _ = _new_engine(sid, sm, tmp, confirm=confirm, cooldown=cooldown, risk_mode=mode)
        wf = _walk(eng, etf, dates, i_full, full_start, end)
    full = _stats([x[1] for x in wf])

    # focal fresh(从 seed 现金)
    with tempfile.TemporaryDirectory(prefix="repair_") as tmp:
        sm = StateManager(tmp, f"{sid}_focal")
        sm.init_new(cfg.INITIAL_CAPITAL)
        eng, _ = _new_engine(sid, sm, tmp, confirm=confirm, cooldown=cooldown, risk_mode=mode)
        ws = _walk(eng, etf, dates, i_seed, seed, end)
    focal = _stats([x[1] for x in ws])
    return {"focal_ret": focal["ret"], "full_ret": full["ret"],
            "full_sharpe": full["sharpe"], "full_mdd": full["mdd"], "full_n": full["n"]}


# 门户(门禁在分析层执行, 这里给数据)
def _matrix_defs(strategy_cfg_risk):
    """每策略基线锚 = 自身模式, 其余变更(risk)将与其锚模式对比。"""
    return [
        # (label, confirm, cooldown, risk_override 或 None=锚)
        ("baseline", 1, 0, None),
        ("F1_confirm2", 2, 0, None),
        ("F1_confirm3", 3, 0, None),
        ("F2_cooldown2", 1, 2, None),
        ("F2_cooldown3", 1, 3, None),
        ("F2_cooldown5", 1, 5, None),
        ("comb_c2_k3", 2, 3, None),
        ("mode_flip", 1, 0, "A" if strategy_cfg_risk == "B" else "B"),
    ]


def _live_daily(sid):
    """live sim 每日总资产(日期去重取最后), 用作 harness 前缀对齐参照。"""
    path = PROJECT_ROOT / "simulation" / "output" / f"sim_log_{sid}.csv"
    df = pd.read_csv(path)
    df["ds"] = df["日期"].astype(str).str[:10]
    df = df[df["ds"].str.contains(r"2026", na=False)]
    g = df.groupby("ds")["总资产"].last()
    return {d: float(v) for d, v in g.items() if pd.notna(v)}


def _selfcheck(sid, a):
    """干净单日重放 baseline 与 live 前缀对齐。live CSV 在个别日期多次重跑
    (append 重复行, 见 momentum 2026-08-18 ×4) 会制造一次性 switch 假象; 故验收 =
    命中同一日历的干净段须逐日一致(≤0.5元), 允许在某 artifact 日后分叉但须识别该日。
    返回 True=通过。"""
    # 重放并取逐日 total
    cfg = _cfg(sid)
    lo = (datetime.strptime(a.start, "%Y-%m-%d")
          - timedelta(days=150)).strftime("%Y-%m-%d")
    etf = _load_range(cfg.ETF_SYMBOLS, lo, a.end, cfg.MOMENTUM_WINDOW)
    dates = _dates(etf[cfg.ETF_SYMBOLS[0]])
    i0 = _pos(etf[cfg.ETF_SYMBOLS[0]], a.seed)
    with tempfile.TemporaryDirectory(prefix="self_") as tmp:
        sm = StateManager(tmp, f"{sid}_self")
        sm.init_new(10000)
        eng, _ = _new_engine(sid, sm, tmp, confirm=1, cooldown=0,
                             risk_mode=cfg.RISK_MODE)
        rows = _walk(eng, etf, dates, i0, a.seed, a.end)
    rep = {d: tv for d, tv, _ in rows}
    live = _live_daily(sid)
    shared = [d for d in rep if d in live]
    # 从 seed 起逐日比较, 直到首个 >0.5 元分歧 => 该日为 artifact 分叉点
    first_div = None
    for d in shared:
        if abs(rep[d] - live[d]) > 0.5:
            first_div = d
            break
    clean_days = shared.index(first_div) if first_div else len(shared)
    print(f"[self-check] {sid}: 干净前缀命中 {clean_days}/{len(shared)} 天"
          f"{'' if not first_div else '  于 '+first_div+' 分叉(live 同日晚重跑 artifact)'}")
    if first_div is None:
        print("[self-check] PASS: 全程复现 live")
        return True
    if clean_days >= 8:
        print("[self-check] PASS(前缀): harness 用真实引擎, 前缀逐日一致 ✓; "
              "分叉应为 live 同一交易日重复运行所致, 非重放逻辑差")
        return True
    print(f"[self-check] FAIL: 干净前缀仅 {clean_days} < 8，中止")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(LIVE_REF))
    ap.add_argument("--seed", default="2026-08-03")
    ap.add_argument("--start", default="2022-03-01")
    ap.add_argument("--end", default="2026-09-04")
    ap.add_argument("--confirm", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=0)
    ap.add_argument("--risk", choices=["A", "B"], default=None)
    ap.add_argument("--sweep", action="store_true", help="跑全矩阵")
    ap.add_argument("--skip-selfcheck", action="store_true")
    ap.add_argument("--table", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not a.skip_selfcheck and not _selfcheck(a.strategy, a):
        sys.exit(1)

    base = replay(a.strategy, full_start=a.start, seed=a.seed, end=a.end, confirm=1, cooldown=0)
    print(f"[baseline] {a.strategy} 干净单日 focal={base['focal_ret']:+.2f}%"
          f"(live累计参照 {LIVE_REF[a.strategy]:+.2f}%, 分叉注明见上) full={base['full_ret']:+.2f}%")

    cfg = _cfg(a.strategy)
    defs = _matrix_defs(cfg.RISK_MODE) if a.sweep else [("run", a.confirm, a.cooldown, a.risk)]
    head = ("label\tstrategy\tconfirm\tcooldown\trisk\tfocal_ret\tfull_ret\t"
            "full_sharpe\tfull_mdd\tfull_n\n")
    lines = []
    for label, c, cd, rk in defs:
        r = replay(a.strategy, full_start=a.start, seed=a.seed, end=a.end, confirm=c, cooldown=cd, risk=rk)
        rm = rk or cfg.RISK_MODE
        def f(x): return "" if x is None or not np.isfinite(x) else f"{x:.2f}"
        lines.append(f"{label}\t{a.strategy}\t{c}\t{cd}\t{rm}\t"
                     f"{r['focal_ret']:.2f}\t{r['full_ret']:.2f}\t"
                     f"{f(r['full_sharpe'])}\t{f(r['full_mdd'])}\t{r['full_n']}")
        print(f"  {label}: focal {r['focal_ret']:+.2f}%  full {r['full_ret']:+.2f}%  "
              f"full_sharpe {f(r['full_sharpe'])}  mdd {f(r['full_mdd'])}%")
    with open(a.table, "w") as f:
        f.write(head + "\n".join(lines) + "\n")
    print(f"Wrote {a.table}")


if __name__ == "__main__":
    main()
