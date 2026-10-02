#!/usr/bin/env python3
"""
回测 vs 模拟盘 轨迹对齐监控（2026-08-03 新增）

目标：区分"市场环境导致模拟盘亏损"与"模拟盘逻辑 bug"。
方法：对同一策略，用回测引擎跑相同区间得到逐日轨迹，与模拟盘 CSV 逐日对齐：
  - 回测从"模拟盘起点前 45 个交易日"起步（保证动量/指标预热充分），
  - 以模拟盘起点日为锚点重锚回测收益（排除预热期对累计收益的干扰），
  - 逐日计算偏差 = 模拟盘收益 - 回测收益（百分点），超阈值 → 微信告警。

回测与模拟盘存在固有差异（回测无涨跌停、执行价=当日open vs 模拟盘=次日open、
回测渐进调仓 vs 模拟盘一次性切换等），正常偏差约 1~3pp；暴跌市/极端行情下可达 5~8pp。

告警判定（2026-10-02 重构）：**只看"当下是否真的脱离轨迹"**，不看累计缺口——
累计缺口是路径依赖的，历史一次性偏离（如同日重跑伪切换、风控在暴跌日卖出）会永久
沉淀；此后即使持仓与回测完全一致也不会收敛，旧逻辑"缺口>8pp 即告警"因此每晚重复
推送假阳性。现改为：
  (1) 持仓与回测不一致且连续 ≥3 个共同交易日 → 告警（真·脱离）
  (2) 近 10 个共同交易日偏差漂移 ≥5pp → 告警（同标的但路径快速拉开）
累计缺口/历史越线天数仅作打印参考（--threshold 即该参考线），不触发告警。

用法：
  python -m simulation.framework.backtest_align --strategy momentum_rotation
  python -m simulation.framework.backtest_align                     # 核心策略列表
  python -m simulation.framework.backtest_align --threshold 0.08    # 自定义缺口参考线(8pp)
  python -m simulation.framework.backtest_align --push              # 超阈值时推送微信
"""

from __future__ import annotations

import argparse
import json
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

# 告警判定参数（2026-10-02 重构：改看"当下是否脱离"，不看累计缺口）
MISMATCH_ALERT_DAYS = 3    # 持仓与回测不一致需连续≥N个共同交易日才告警
                           # （切换过渡期/T+1 执行差 1~2 天属正常，不算脱离）
DRIFT_WINDOW = 10          # 近期漂移窗口（共同交易日）
DRIFT_ALERT_PP = 5.0       # 窗口内偏差漂移≥此值(pp)告警（同标的但路径快速拉开）
REMIND_DAYS = 10           # 同一未解决状态每隔 N 个共同交易日再提醒一次（边沿触发用）
ALERT_STATE_PATH = OUTPUT_DIR / "backtest_align_state.json"  # 上次推送状态（防重复打扰）


def load_sim_series(sid: str) -> tuple[list[str], list[float], str | None, list[str]]:
    """读取模拟盘 CSV，返回 (日期列表, 累计收益率%列表, 对齐锚点日期, 每日持仓标的列表)。

    处理三类特殊情况：
      1. "历史起点"追记行（日期可能为空）——跳过
      2. "框架重构"注释行（收益序列断裂点）——跳过
      3. 收益重置跳变：累计收益率从明显亏损(< -5%)跳回≈0（如2026-07-03
         框架v2重构），从最后一个跳变点起重新锚定——重置后收益相对新起点
         累计，与回测对齐时参照系必须一致（否则产生系统性假偏差）。
    """
    path = OUTPUT_DIR / f"sim_log_{sid}.csv"
    if not path.exists():
        return [], [], None, []
    # 持仓标的按字符串读：否则 pandas 会把 510050 推成 510050.0，空仓行变 nan，
    # 与回测侧 "510050"/"" 的字符串比较必然不等 → 全窗口假"持仓不一致"
    df = pd.read_csv(path, dtype={"持仓标的": str})
    dates, rets, reset_markers, holds = [], [], [], []
    for _, row in df.iterrows():
        d = str(row.get("日期", "")).strip()
        op = str(row.get("操作", "")).strip()
        # 记录"重置注释行"日期（收益序列断裂点，优先用于锚定）
        # 识别两类：框架重构注释行 / 用户清零重启注释行
        if ("重构" in op or "清零重启" in op) and d:
            reset_markers.append(d[:10])
        if not d or "历史起点" in d or "重构" in op or "清零重启" in op:
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
        h = row.get("持仓标的", "")
        holds.append("" if pd.isna(h) else str(h).strip())

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
    return dates[anchor_idx:], rets[anchor_idx:], anchor, holds[anchor_idx:]


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
    # hold_symbol 按字符串读：否则带空值的列会被推成 float（512100 → 512100.0），
    # 与模拟盘侧字符串比较必然不等 → 全窗口假"持仓不一致"
    return pd.read_csv(daily_path, dtype={"hold_symbol": str})


def align(sid: str, sim_dates: list[str], sim_rets: list[float],
          sim_holds: list[str], back_df: pd.DataFrame) -> list[dict]:
    """逐日对齐，返回偏差记录列表（含双方当日持仓）。

    回测侧以锚点（模拟盘首日）重锚：back_ret_anchored = (1+cum)/(1+cum@anchor) - 1。
    偏差(pp) = sim_ret(%) - back_ret_anchored(%)。
    """
    back_df = back_df.copy()
    back_df["date"] = back_df["date"].astype(str)
    back_map = dict(zip(back_df["date"], back_df["cumulative_return"]))
    if "hold_symbol" in back_df.columns:
        back_hold_map = {
            d: ("" if pd.isna(h) else str(h))
            for d, h in zip(back_df["date"], back_df["hold_symbol"])
        }
    else:
        back_hold_map = {}

    anchor = sim_dates[0]
    cum_at_anchor = back_map.get(anchor)
    if cum_at_anchor is None:
        print(f"  ⚠ 回测轨迹缺少锚点 {anchor}，跳过")
        return []

    records = []
    for d, s_ret, s_hold in zip(sim_dates, sim_rets, sim_holds):
        cum_back = back_map.get(d)
        if cum_back is None:
            continue
        back_anchored = (1 + cum_back) / (1 + cum_at_anchor) - 1
        dev = s_ret - back_anchored * 100  # 百分点
        records.append({"date": d, "sim_ret": s_ret,
                        "back_ret": back_anchored * 100, "dev": dev,
                        "sim_hold": s_hold, "back_hold": back_hold_map.get(d, "")})
    return records


def _load_alert_state() -> dict:
    """读取上次告警状态（不存在/损坏则返回空，不影响主流程）。"""
    try:
        return json.loads(ALERT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_alert_state(state: dict) -> None:
    ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def report(sid: str, records: list[dict], threshold: float, push: bool,
           alert_state: dict) -> dict:
    """输出偏差报告，按"当下是否真的脱离轨迹"告警。返回 {"over": 是否越线, "pushed": 是否推送}。

    2026-10-02 重构：原逻辑只看"最新交易日累计收益缺口>阈值"。累计缺口是路径
    依赖的——历史一次性偏离（如同日重跑造成的伪切换、风控在暴跌日卖出）会把
    缺口永久沉淀在账户里，即使此后持仓与回测完全一致、逐日同步运行，缺口也
    不会收敛 → 每晚照常告警，成为永久假阳性（09-30 的 composite/cross_border
    即此情形：持仓与回测一致，-8.4/-8.1pp 缺口是八月事件的沉淀）。

    新判定（只看"当下"，threshold 仅作缺口参考线打印）：
      1. 持仓不一致且连续 ≥ MISMATCH_ALERT_DAYS 个共同交易日 → 告警（真·脱离轨迹）
      2. 近 DRIFT_WINDOW 个交易日偏差漂移 ≥ DRIFT_ALERT_PP → 告警（同标的但路径快速拉开）
    窗口内历史越线天数仅作打印参考，不触发。

    推送为"边沿触发"：同一未解决状态只在**首次出现**和每隔 REMIND_DAYS 个交易日
    提醒一次，不每晚重复（判定条件编码为 key 存 ALERT_STATE_PATH）。
    """
    if not records:
        print(f"  ⚠ {sid}: 无可对齐记录")
        return {"over": False, "pushed": False}
    last = records[-1]
    max_dev = max(records, key=lambda r: abs(r["dev"]))
    hist_over = [r["date"] for r in records if abs(r["dev"]) > threshold * 100]

    # 近期漂移：最新偏差 相对 窗口前偏差 的变化
    n = min(DRIFT_WINDOW, len(records) - 1)
    drift = last["dev"] - records[-1 - n]["dev"] if n > 0 else 0.0
    # 持仓不一致的连续天数（从最新交易日往前数）
    mismatch_days = 0
    for r in reversed(records):
        if r["sim_hold"] != r["back_hold"]:
            mismatch_days += 1
        else:
            break

    print(f"\n═══ {sid} 轨迹对齐（共{len(records)}个共同交易日）═══")
    print(f"  当前: 模拟盘{last['sim_ret']:+.2f}% vs 回测{last['back_ret']:+.2f}% → 累计缺口{last['dev']:+.2f}pp（参考线{threshold*100:.0f}pp）")
    print(f"  最大缺口(历史): {max_dev['date']} {max_dev['dev']:+.2f}pp (模拟盘{max_dev['sim_ret']:+.1f}% vs 回测{max_dev['back_ret']:+.1f}%)")
    print(f"  持仓: 模拟盘{last['sim_hold'] or '空仓'} vs 回测{last['back_hold'] or '空仓'}"
          + ("  一致 ✅" if mismatch_days == 0 else f"  不一致（已连续{mismatch_days}个交易日）"))
    print(f"  近{n}日漂移: {drift:+.2f}pp（告警线{DRIFT_ALERT_PP:.0f}pp）")
    if hist_over:
        print(f"  ℹ 窗口内历史曾超参考线 {len(hist_over)}天: {hist_over[:8]}（只作参考，不触发告警）")

    reasons = []
    if mismatch_days >= MISMATCH_ALERT_DAYS:
        reasons.append(f"持仓脱离回测轨迹已连续{mismatch_days}个交易日"
                       f"（模拟盘{last['sim_hold'] or '空仓'} vs 回测{last['back_hold'] or '空仓'}）")
    if abs(drift) >= DRIFT_ALERT_PP:
        reasons.append(f"近{n}日偏差漂移{drift:+.2f}pp超{DRIFT_ALERT_PP:.0f}pp")

    # ── 告警状态键（边沿触发）：键不变 = 同一个未解决的问题，不重复打扰 ──
    if reasons:
        key = f"mismatch:{last['sim_hold'] or '空仓'}->{last['back_hold'] or '空仓'}" \
            if mismatch_days >= MISMATCH_ALERT_DAYS else "drift_over"
    else:
        key = "ok"

    prev = alert_state.get(sid, {}) if isinstance(alert_state.get(sid), dict) else {}
    last_push = str(prev.get("last_push_date", ""))
    since_push = sum(1 for r in records if r["date"] > last_push)
    changed = prev.get("key") != key
    should_push = bool(reasons) and push and (changed or since_push >= REMIND_DAYS)
    pushed = False

    if reasons:
        print(f"  ⚠ 脱离轨迹告警：{'；'.join(reasons)}")
        if push and not should_push:
            print(f"  ℹ 同一状态已于 {last_push} 推送过（其后第{since_push}个交易日），"
                  f"未满{REMIND_DAYS}个交易日不重复推送")
        if should_push:
            # 用 send_message 而非 push_error_alert：后者会标"运行异常"标题
            from simulation.framework.notify import send_message
            today = date.today().strftime("%Y-%m-%d")
            lines = [
                f"⚠️ 轨迹对齐告警 {sid} | {today}",
                *reasons,
                f"累计缺口 {last['dev']:+.2f}pp（模拟盘{last['sim_ret']:+.2f}% vs 回测{last['back_ret']:+.2f}%）",
                "提示：检查模拟盘持仓与回测持仓，判断是逻辑bug还是市场异常",
            ]
            send_message(f"⚠️ 轨迹对齐告警-{sid}", "\n".join(lines))
            pushed = True
            alert_state[sid] = {"key": key, "last_push_date": last["date"]}
        elif push:
            # 状态仍是同一问题：只更新时间戳基线，不改 key
            alert_state[sid] = {"key": key, "last_push_date": last_push}
        return {"over": True, "pushed": pushed}

    print(f"  持仓一致、无快速漂移 → 不告警（累计缺口{last['dev']:+.2f}pp 属历史沉淀，不随每日同步收敛）")
    if push:
        alert_state[sid] = {"key": "ok", "last_push_date": ""}
    return {"over": False, "pushed": False}


def main():
    parser = argparse.ArgumentParser(description="回测vs模拟盘轨迹对齐监控")
    parser.add_argument("--strategy", type=str, default="",
                        help="策略名（默认跑核心列表）")
    parser.add_argument("--threshold", type=float, default=0.08,
                        help="累计缺口参考线（默认0.08=8pp，仅打印提示，不触发告警）")
    parser.add_argument("--push", action="store_true",
                        help="超阈值时推送微信告警")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else DEFAULT_STRATEGIES
    latest = get_latest_trade_day()
    print(f"轨迹对齐监控 | 最新交易日 {latest} | 阈值 {args.threshold*100:.0f}pp")

    alert_state = _load_alert_state()
    any_alert, any_push = False, False
    for sid in strategies:
        sim_dates, sim_rets, anchor, sim_holds = load_sim_series(sid)
        if not sim_dates or anchor is None:
            print(f"  ⚠ {sid}: 无模拟盘记录，跳过")
            continue
        start = get_prewarm_start(anchor)
        print(f"\n  ▶ {sid}: 锚点 {anchor} → 回测 {start}~{latest}")
        back_df = run_backtest(sid, start, latest, f"align_{sid}")
        if back_df is None:
            continue
        records = align(sid, sim_dates, sim_rets, sim_holds, back_df)
        result = report(sid, records, args.threshold, args.push, alert_state)
        any_alert |= result["over"]
        any_push |= result["pushed"]

    if args.push:
        _save_alert_state(alert_state)  # 记录已推送状态，防下一晚重复打扰

    # 有告警也正常退出：告警已通过微信推送（或已推过处于静默期），监控步骤本身"完成"
    tail = "" if not args.push else ("（已推送）" if any_push else "（已推送过，本次静默）")
    print(f"\n完成：{'存在脱离轨迹告警 ⚠' + tail if any_alert else '全部持仓一致、无快速漂移 ✅'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
