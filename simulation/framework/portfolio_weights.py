#!/usr/bin/env python3
"""策略层组合权重月报（方向 3 第二步，2026-10-02）

回答"这 20 条策略盘到底该怎么分资金"：
  1. 按近 60 交易日波动率算 **风险平价（1/σ）权重**（月度再平衡）
  2. 算各 sleeve 的**风险贡献**（w_i·(Σw)_i / σ_p）——谁在真正占用风险预算
  3. 推送到微信

数据源优先级（只用过去数据、只读、不跑回测）：
  1) 模拟盘 live 曲线：`simulation/output/sim_log_<sid>.csv`，取**最近一次清零注释行之后**
     的数据，不足 MIN_LIVE_DAYS 天则不用
  2) 回测曲线：该策略 `strategies/<sid>/output/` 下覆盖窗口的最新一次
每次推送会标注有多少条用了 live、多少条退回回测（模拟盘 2026-10-02 刚清零，初期全为回测）

依据：方向 3 第一步实验（2024-01~2026-09）——风险平价组合 夏普 1.21 / MDD -10.9%，
优于单策略中位 0.87 / -23.3%，也优于等权 1.13 / -14.9%；叠加波动率目标化为负优化。
详见 strategies/RESEARCH_DIRECTIONS_20261002.md。

用法：
    python -m simulation.framework.portfolio_weights          # 只打印
    python -m simulation.framework.portfolio_weights --push   # 打印并推送微信
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIM_OUT = ROOT / "simulation" / "output"
EXCLUDE = {"combined"}       # 聚合器（= momentum+pair），计入会重复
VOL_WINDOW = 60              # 波动率窗口（交易日）
MIN_LIVE_DAYS = VOL_WINDOW   # live 曲线至少这么多天才用
SHOW_MIN_W = 0.03            # 报告里单列的最小权重

# 中文名（缺失则用 id）
NAMES = {
    "momentum_rotation": "动量轮动", "composite_momentum": "复合动量",
    "macd_trend_rotation": "MACD趋势", "rsi_trend_rotation": "RSI趋势",
    "adaptive_rotation": "自适应轮动", "adx_trend_rotation": "ADX趋势",
    "momentum_vol_filter": "波动率过滤", "pair_trading": "配对交易",
    "dual_momentum": "双动量", "sortino_ranking": "Sortino排名",
    "sharpe_ranking": "Sharpe排名", "median_momentum": "中位数#2",
    "tail_risk": "尾部风险", "bollinger_reversion": "布林带回归",
    "spread_reversion": "价差回归", "volume_price": "量价配合",
    "gold_safe_haven": "黄金避险", "cross_border": "跨境轮动",
    "asset_allocation": "资产配置", "neural_momentum": "Neural动量",
}


def _pipeline_strategies() -> list[str]:
    import re
    src = (ROOT / "pipeline.py").read_text(encoding="utf-8")
    return [m for m in re.findall(r'"id":\s*"(\w+)"', src)
            if m not in ("sync", "backtest_align") and m not in EXCLUDE]


def _live_returns(sid: str) -> pd.Series | None:
    """模拟盘 live 曲线（最近一次清零注释行之后的 总资产 → 日收益）。"""
    p = SIM_OUT / f"sim_log_{sid}.csv"
    if not p.exists():
        return None
    rows = list(csv.reader(p.open(encoding="utf-8-sig")))
    if not rows:
        return None
    start = 0
    for i, r in enumerate(rows):
        if len(r) > 2 and str(r[2]).startswith("【"):
            start = i + 1
    body = rows[start:]
    dates, vals = [], []
    for r in body:
        if len(r) < 10:
            continue
        try:
            vals.append(float(str(r[9]).replace(",", "")))
            dates.append(str(r[0])[:10])
        except ValueError:
            continue
    if len(vals) < MIN_LIVE_DAYS:
        return None
    s = pd.Series(vals, index=pd.to_datetime(dates))
    return s[~s.index.duplicated(keep="last")].pct_change().dropna()


def _backtest_returns(sid: str, start: str, end: str) -> pd.Series | None:
    """回测曲线（覆盖窗口的最新一次；只读）。"""
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
        s = pd.Series(pd.to_numeric(df["total_value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]))
        if s.empty:
            continue
        # 起点必须与请求一致（容差 10 自然日）：动量类路径依赖，
        # 2021 起与 2024 起的"同窗口"结果可差几十个百分点 → 混用会让权重/波动率不可比
        if (s.index[0] - pd.Timestamp(start)).days > 10:
            continue
        s = s[(s.index >= start) & (s.index <= end)]
        if len(s) >= VOL_WINDOW:
            return s[~s.index.duplicated(keep="last")].pct_change().dropna()
    return None


def _engine_curve(sid: str, start: str, end: str, quiet: bool = False) -> pd.Series | None:
    """无 run.py/无输出时：直接驱动该策略**自己**的 engine 取曲线（在内存里跑一次）。

    注意必须选 `engine.py` 里**本模块定义**的引擎类——`dir(mod)` 还含 import 进来的
    基类 momentum 的 BacktestEngine，选错会跑成动量（2026-10-02 组合层实验踩过）。
    """
    import contextlib
    import io
    try:
        if sid == "asset_allocation":
            mod = __import__("strategies.asset_allocation.backtest", fromlist=["*"])
            with contextlib.redirect_stdout(io.StringIO()):
                close_df, _ = mod.load_data()
                nav = mod.backtest_equity_series(close_df, "risk_parity", start=start)
            if nav is None:
                return None
            s = pd.Series(nav.values, index=pd.to_datetime(nav.index))
            return _dedupe_ret(s)
        mod = __import__(f"strategies.{sid}.engine", fromlist=["*"])
        own = [n for n in dir(mod) if n.endswith("Engine")
               and isinstance(getattr(mod, n), type)
               and getattr(mod, n).__module__ == mod.__name__]
        if not own:
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            eng = getattr(mod, own[0])()
            eng.load_data(start_date=start, end_date=end)
            eng.run()
            df = eng.get_daily_df()
        s = pd.Series(pd.to_numeric(df["total_value"], errors="coerce").values,
                      index=pd.to_datetime(df["date"]))
        return _dedupe_ret(s)
    except Exception as e:
        if not quiet:
            print(f"  ⚠ {sid} 驱动 engine 失败: {type(e).__name__}: {e}")
        return None


def _run_py_curve(sid: str, start: str, end: str) -> pd.Series | None:
    """跑一次 `python -m strategies.<sid>.run`（解析其打印的输出目录取曲线）。

    仅用于"没有 engine.py"的策略（如 cross_border 复用 momentum 引擎 + 自己的池子）。
    """
    import subprocess
    import tempfile
    rp = ROOT / f"strategies/{sid}/run.py"
    if not rp.exists():
        return None
    try:
        proc = subprocess.run([sys.executable, "-m", f"strategies.{sid}.run",
                               "--start", start, "--end", end, "--tag", "gate2026"],
                              cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    except Exception as e:
        print(f"  ⚠ {sid} run.py 失败: {e}")
        return None
    out_dir = None
    for line in (proc.stdout or "").splitlines():
        if "输出目录" in line:
            out_dir = line.split(":", 1)[1].strip()
    if not out_dir:
        print(f"  ⚠ {sid} run.py 未打印输出目录（stderr 末尾: {(proc.stderr or '')[-200:]}）")
        return None
    f = Path(out_dir) / "daily_records.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    s = pd.Series(pd.to_numeric(df["total_value"], errors="coerce").values,
                  index=pd.to_datetime(df["date"]))
    return _dedupe_ret(s)


def _dedupe_ret(s: pd.Series) -> pd.Series:
    """去重日期 + 转日收益（各来源统一走这里，避免重复标签建 DataFrame 失败）。"""
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def collect_sleeves(start: str, end: str, run_missing: bool = False):
    """采集各 sleeve 的日收益。

    返回 (R, src, note)：R=date×sid 收益矩阵（已按共同日历对齐），
    src={sid: "live"|"backtest"|"engine"}，note=被剔除的 sleeve 说明。
    """
    rets, src, note = {}, {}, {}
    for sid in _pipeline_strategies():
        s = _live_returns(sid)
        if s is not None and len(s) >= MIN_LIVE_DAYS:
            rets[sid], src[sid] = s.tail(120), "live"
            continue
        s = _backtest_returns(sid, start, end)
        if s is not None:
            rets[sid], src[sid] = s, "backtest"
            continue
        if run_missing:
            s = _engine_curve(sid, start, end, quiet=True)
            if s is not None and len(s) >= VOL_WINDOW:
                rets[sid], src[sid] = s, "engine"
                continue
            s = _run_py_curve(sid, start, end)      # 无 engine.py 的策略（如 cross_border）
            if s is not None and len(s) >= VOL_WINDOW:
                rets[sid], src[sid] = s, "runpy"
                continue
        note[sid] = "无可用曲线（无回测输出；可加 run_missing=True 现场驱动 engine）"
    R = pd.DataFrame(rets).sort_index()
    return R, src, note


def align_window(R: pd.DataFrame, window: int = VOL_WINDOW):
    """取**共同日历**上的最近 window 个交易日，剔除窗口内缺数据的 sleeve。

    各 sleeve 的可比性要求同一天数、同一区间——否则波动率/协方差不可比
    （2026-10-02 踩过：neural 评分只到 08-04，其"近60日"与别人不是同一段）。
    """
    if R.empty:
        return R
    good = R.dropna(axis=1, thresh=window).tail(window)
    good = good.dropna(axis=1)
    return good


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """成分风险贡献 = w_i·(Σw)_i / σ_p。"""
    port_var = float(w.values @ cov.values @ w.values)
    if port_var <= 0:
        return pd.Series(0.0, index=w.index)
    mrc = cov.values @ w.values                      # 边际风险贡献
    return pd.Series(w.values * mrc / np.sqrt(port_var), index=w.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="推送微信")
    ap.add_argument("--asof", default="", help="报告日期（默认今天）")
    ap.add_argument("--bt-start", default="2024-01-01", help="回测曲线起点（可比性要求起点一致）")
    ap.add_argument("--bt-end", default="2026-09-30")
    a = ap.parse_args()

    today = a.asof or date.today().isoformat()
    R_raw, src, note = collect_sleeves(a.bt_start, a.bt_end, run_missing=True)
    tail = align_window(R_raw, VOL_WINDOW)
    if tail.shape[1] < 2:
        print(f"⚠ 可用曲线不足（{tail.shape[1]} 条），跳过组合权重报告")
        for sid, why in note.items():
            print(f"   - {sid}: {why}")
        return
    src = {k: v for k, v in src.items() if k in tail.columns}
    dropped = [c for c in R_raw.columns if c not in tail.columns]
    print(f"  （共同 {tail.shape[0]} 个交易日 × {tail.shape[1]} 条 sleeve）")
    for sid in dropped:
        print(f"   ⚠ 剔除 {NAMES.get(sid, sid)}：近{VOL_WINDOW}日窗口内曲线缺数据"
              f"（如 neural 的评分文件只到 2026-08-04）")
    for sid, why in note.items():
        print(f"   ⚠ 剔除 {NAMES.get(sid, sid)}: {why}")
    vol = tail.std() * np.sqrt(252)
    w = (1.0 / vol).replace([np.inf, -np.inf], np.nan).dropna()
    w = w / w.sum()
    cov = tail.cov() * 252
    rc = risk_contributions(w, cov.reindex(index=w.index, columns=w.index))
    port_vol = float(np.sqrt(w.values @ cov.loc[w.index, w.index].values @ w.values))

    n_live = sum(1 for v in src.values() if v == "live")
    lines = [f"📊 策略组合权重月报（风险平价）| {today}",
             f"口径：{len(w)} 条策略 · 近{VOL_WINDOW}交易日波动率倒数加权 · 月度再平衡",
             f"组合预估年化波动 {port_vol*100:.1f}%", ""]
    order = w.sort_values(ascending=False)
    shown = order[order >= SHOW_MIN_W]
    lines.append("建议权重（≥3%）:")
    for sid, wi in shown.items():
        lines.append(f"  {NAMES.get(sid, sid):<8} {wi*100:5.1f}%  "
                     f"（σ{vol[sid]*100:4.1f}% 风险贡献{rc[sid]/port_vol*100:4.1f}%）")
    rest = order[order < SHOW_MIN_W]
    if len(rest):
        lines.append(f"  其余 {len(rest)} 条合计 {rest.sum()*100:.1f}%："
                     + " ".join(f"{NAMES.get(s, s)}{order[s]*100:.1f}%" for s in rest.index[:6])
                     + ("…" if len(rest) > 6 else ""))
    if dropped:
        lines.append(f"未纳入：{'、'.join(NAMES.get(s, s) for s in dropped)}（近{VOL_WINDOW}日窗口缺数据）")
    lines += ["", f"数据源：live {n_live} 条 / 回测 {len(w)-n_live} 条"
                  + ("（模拟盘刚清零，回测为主）" if n_live == 0 else ""),
              "依据：方向3 实验——风险平价组合 夏普1.21/MDD-10.9% vs 单策略中位 0.87/-23.3%"]
    text = "\n".join(lines)
    print(text)
    if a.push:
        from simulation.framework.notify import send_message
        send_message(f"📊 策略组合权重月报 | {today}", text)
        print("\n✅ 已推送")


if __name__ == "__main__":
    main()
