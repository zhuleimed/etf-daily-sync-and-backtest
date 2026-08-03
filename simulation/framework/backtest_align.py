#!/usr/bin/env python3
"""
回测 vs 模拟盘 轨迹对齐监控（2026-08-03 新增）

目标：区分"市场环境导致模拟盘亏损"与"模拟盘逻辑 bug"。
方法：对同一策略，用回测引擎跑相同区间得到逐日轨迹，与模拟盘 CSV 逐日对齐：
  - 回测从"模拟盘起点前 45 个交易日"起步（保证动量/指标预热充分），
  - 以模拟盘起点日为锚点重锚回测收益（排除预热期对累计收益的干扰），
  - 逐日计算偏差 = 模拟盘收益 - 回测收益（百分点），超阈值 → 微信告警。

回测与模拟盘存在固有差异（回测无涨跌停、执行价=当日open vs 模拟盘=次日open 等），
正常偏差约 1~3pp；暴跌市/极端行情下可达 5~8pp。
偏差 > 阈值时告警，提示人工检查——既可能是模拟盘 bug，也可能是市场异常。

用法：
  python -m simulation.framework.backtest_align --strategy momentum_rotation
  python -m simulation.framework.backtest_align                     # 核心策略列表
  python -m simulation.framework.backtest_align --threshold 0.08    # 自定义阈值(8pp)
  python -m simulation.framework.backtest_align --push              # 超阈值时推送微信
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 默认纳入对齐监控的核心策略（需 run.py 支持 --start/--end/--tag）
DEFAULT_STRATEGIES = [
    "momentum_rotation",
    "momentum_vol_filter",
    "composite_momentum",
    "rsi_trend_rotation",
    "adx_trend_rotation",
    "cross_border",
]

DB_PATH = PROJECT_ROOT / "data" / "etf_daily.db"
OUTPUT_DIR = PROJECT_ROOT / "simulation" / "output"
PREWARM_DAYS = 45  # 回测预热交易日数（保证动量/指标充分计算）


def load_sim_series(sid: str) -> tuple[list[str], list[float], str | None]:
    """读取模拟盘 CSV，返回 (日期列表, 累计收益率%列表, 对齐锚点日期)。

    处理三类特殊情况：
      1. "历史起点"追记行（日期可能为空）——跳过
      2. "框架重构"注释行（收益序列断裂点）——跳过
      3. 收益重置跳变：累计收益率从明显亏损(< -5%)跳回≈0（如2026-07-03
         框架v2重构），从最后一个跳变点起重新锚定——重置后收益相对新起点
         累计，与回测对齐时参照系必须一致（否则产生系统性假偏差）。
    """
    path = OUTPUT_DIR / f"sim_log_{sid}.csv"
    if not path.exists():
        return [], [], None
    df = pd.read_csv(path)
    dates, rets, reset_markers = [], [], []
    for _, row in df.iterrows():
        d = str(row.get("日期", "")).strip()
        op = str(row.get("操作", "")).strip()
        # 记录"框架重构"注释行日期（收益序列断裂点，优先用于锚定）
        if "重构" in op and d:
            reset_markers.append(d[:10])
        if not d or "历史起点" in d or "重构" in op:
            continue
        # 累计收益率形如 "-10.82%"；空值跳过
        r = str(row.get("累计收益率", "")).strip().replace("%", "")
        if not r:
            continue
        try:
            rets.append(float(r))
        except ValueError:
            continue
        dates.append(d[:10])  # 去掉可能的"←历史起点"后缀

    # 锚点确定（优先可靠信号，跳变检测兜底）：
    # 1. 存在"框架重构"注释行 → 取最后一个重构行日期起（含同日真实行）
    #    重构后收益相对新起点累计，参照系必须与回测一致
    # 2. 无注释行时检测收益跳变：ret 从 < -5% 跳回 |ret| < 0.5%（手动重置）
    anchor_idx = 0
    if reset_markers:
        last_reset = reset_markers[-1]
        for i, d in enumerate(dates):
            if d >= last_reset:
                anchor_idx = i
                break
    else:
        for i in range(1, len(rets)):
            if rets[i - 1] < -5 and abs(rets[i]) < 0.5 and (rets[i] - rets[i - 1]) > 5:
                anchor_idx = i
    if anchor_idx > 0:
        print(f"  ℹ {sid}: 检测到收益重置点（{dates[anchor_idx]}），从重置后对齐"
              f"（{anchor_idx}行重置前历史不参与）")
    anchor = dates[anchor_idx] if dates else None
    return dates[anchor_idx:], rets[anchor_idx:], anchor


def get_prewarm_start(anchor_date: str, n: int = PREWARM_DAYS) -> str:
    """从数据库取 anchor_date 往前第 n 个交易日（任意 ETF 的交易日历）。"""
    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT date FROM etf_daily WHERE date < ? "
            "ORDER BY date DESC LIMIT ?",
            (anchor_date, n),
        ).fetchall()
    if len(rows) < n:
        return anchor_date  # 历史不足，退化为从锚点开始
    return rows[-1][0]  # 第 n 个（最远的一个）


def get_latest_trade_day() -> str:
    """数据库最新交易日。"""
    with sqlite3.connect(str(DB_PATH)) as conn:
        return conn.execute("SELECT MAX(date) FROM etf_daily").fetchone()[0]


def run_backtest(sid: str, start: str, end: str, tag: str) -> pd.DataFrame | None:
    """子进程调 run.py 跑回测，返回 daily_records DataFrame。"""
    cmd = [
        sys.executable, "-m", f"strategies.{sid}.run",
        "--start", start, "--end", end, "--tag", tag,
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {sid} 回测超时，跳过")
        return None
    if proc.returncode != 0:
        print(f"  ❌ {sid} 回测失败:\n{proc.stderr[-500:]}")
        return None
    # 从 run.py 输出中找输出目录
    m = None
    for line in proc.stdout.splitlines():
        if "输出目录" in line:
            m = line.split(":", 1)[1].strip()
            break
    if not m:
        print(f"  ⚠ {sid} 未找到输出目录，跳过")
        return None
    daily_path = Path(m.strip()) / "daily_records.csv"
    if not daily_path.exists():
        print(f"  ⚠ {sid} 无 daily_records.csv，跳过")
        return None
    return pd.read_csv(daily_path)


def align(sid: str, sim_dates: list[str], sim_rets: list[float],
          back_df: pd.DataFrame) -> list[dict]:
    """逐日对齐，返回偏差记录列表。

    回测侧以锚点（模拟盘首日）重锚：back_ret_anchored = (1+cum)/(1+cum@anchor) - 1。
    偏差(pp) = sim_ret(%) - back_ret_anchored(%)。
    """
    back_df = back_df.copy()
    back_df["date"] = back_df["date"].astype(str)
    back_map = dict(zip(back_df["date"], back_df["cumulative_return"]))

    anchor = sim_dates[0]
    cum_at_anchor = back_map.get(anchor)
    if cum_at_anchor is None:
        print(f"  ⚠ 回测轨迹缺少锚点 {anchor}，跳过")
        return []

    records = []
    for d, s_ret in zip(sim_dates, sim_rets):
        cum_back = back_map.get(d)
        if cum_back is None:
            continue
        back_anchored = (1 + cum_back) / (1 + cum_at_anchor) - 1
        dev = s_ret - back_anchored * 100  # 百分点
        records.append({"date": d, "sim_ret": s_ret,
                        "back_ret": back_anchored * 100, "dev": dev})
    return records


def report(sid: str, records: list[dict], threshold: float, push: bool) -> bool:
    """输出偏差报告，超阈值告警。返回是否超阈值。"""
    if not records:
        print(f"  ⚠ {sid}: 无可对齐记录")
        return False
    last = records[-1]
    max_dev = max(records, key=lambda r: abs(r["dev"]))
    over = [r for r in records if abs(r["dev"]) > threshold * 100]

    print(f"\n═══ {sid} 轨迹对齐（共{len(records)}个共同交易日）═══")
    print(f"  当前: 模拟盘{last['sim_ret']:+.2f}% vs 回测{last['back_ret']:+.2f}% → 偏差{last['dev']:+.2f}pp")
    print(f"  最大偏差: {max_dev['date']} {max_dev['dev']:+.2f}pp (模拟盘{max_dev['sim_ret']:+.1f}% vs 回测{max_dev['back_ret']:+.1f}%)")
    if over:
        print(f"  ⚠ 超阈值({threshold*100:.0f}pp) {len(over)}天: {[r['date'] for r in over[:8]]}")
        if push:
            # 用 send_message 而非 push_error_alert：后者会标"运行异常"标题
            from simulation.framework.notify import send_message
            today = date.today().strftime("%Y-%m-%d")
            lines = [
                f"⚠️ 轨迹对齐告警 {sid} | {today}",
                f"模拟盘 {last['sim_ret']:+.2f}% vs 回测 {last['back_ret']:+.2f}%",
                f"当前偏差 {last['dev']:+.2f}pp，最大 {max_dev['dev']:+.2f}pp ({max_dev['date']})",
                f"超阈值天数: {len(over)}",
                "提示：检查模拟盘日志与回测轨迹，判断是逻辑bug还是市场异常",
            ]
            send_message(f"⚠️ 轨迹对齐告警-{sid}", "\n".join(lines))
        return True
    print(f"  偏差在阈值内 ✅")
    return False


def main():
    parser = argparse.ArgumentParser(description="回测vs模拟盘轨迹对齐监控")
    parser.add_argument("--strategy", type=str, default="",
                        help="策略名（默认跑核心列表）")
    parser.add_argument("--threshold", type=float, default=0.08,
                        help="偏差阈值（默认0.08=8个百分点）")
    parser.add_argument("--push", action="store_true",
                        help="超阈值时推送微信告警")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else DEFAULT_STRATEGIES
    latest = get_latest_trade_day()
    print(f"轨迹对齐监控 | 最新交易日 {latest} | 阈值 {args.threshold*100:.0f}pp")

    any_alert = False
    for sid in strategies:
        sim_dates, sim_rets, anchor = load_sim_series(sid)
        if not sim_dates or anchor is None:
            print(f"  ⚠ {sid}: 无模拟盘记录，跳过")
            continue
        start = get_prewarm_start(anchor)
        print(f"\n  ▶ {sid}: 锚点 {anchor} → 回测 {start}~{latest}")
        back_df = run_backtest(sid, start, latest, f"align_{sid}")
        if back_df is None:
            continue
        records = align(sid, sim_dates, sim_rets, back_df)
        if report(sid, records, args.threshold, args.push):
            any_alert = True

    # 有告警也正常退出：告警已通过微信推送，监控步骤本身"完成"
    print(f"\n完成：{'存在超阈值偏差 ⚠（已推送告警）' if any_alert else '全部在阈值内 ✅'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
