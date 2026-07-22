"""
ETF 模拟盘月度报告模块

每月最后一天管线结束后自动运行，汇总所有策略的月度/累计绩效。

数据来源：simulation/output/sim_log_*.csv（每个策略独立的 CSV 日志）
输出：一条微信推送，包含全策略月度对比表。
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from simulation.framework.notify import send_message

# ── 策略名称映射 ──
STRATEGY_NAMES: dict[str, str] = {
    "momentum_rotation": "动量轮动",
    "composite_momentum": "复合动量",
    "macd_trend_rotation": "MACD趋势",
    "adx_trend_rotation": "ADX趋势",
    "rsi_trend_rotation": "RSI趋势",
    "momentum_vol_filter": "波动率过滤",
    "pair_trading": "配对交易",
    "adaptive_rotation": "自适应轮动",
    "gold_safe_haven": "黄金避险 🥇",
    "cross_border": "跨境轮动 🌏",
    "combined": "组合策略",
}


def _compute_monthly_metrics(
    df: pd.DataFrame,
    initial_capital: float,
    month_start: str,
    month_end: str,
) -> dict:
    """从月度CSV切片计算当月绩效。"""
    mdf = df[(df["日期"] >= month_start) & (df["日期"] <= month_end)]
    if mdf.empty or len(mdf) < 2:
        return {}

    tv = mdf["总资产"].values
    month_start_val = tv[0]
    month_end_val = tv[-1]
    month_return = month_end_val / month_start_val - 1 if month_start_val > 0 else 0

    # 当月最大回撤
    cum = tv / tv[0]
    peak = np.maximum.accumulate(cum)
    month_dd = (cum - peak).min() / peak[np.argmin(cum - peak)] if len(peak) > 0 else 0

    # 总累计
    total_return = tv[-1] / initial_capital - 1 if initial_capital > 0 else 0

    # 年化（需要至少20个数据点）
    n_all = len(df)
    annual_return = (1 + total_return) ** (252 / n_all) - 1 if n_all >= 20 else None

    # 全周期夏普
    if n_all >= 2:
        tv_all = df["总资产"].values
        daily_ret = pd.Series(tv_all).pct_change().dropna()
        if daily_ret.std() > 1e-10:
            sharpe = round(
                (daily_ret.mean() - 0.03 / 252) / daily_ret.std() * np.sqrt(252), 2
            )
        else:
            sharpe = None
    else:
        sharpe = None

    # 全周期最大回撤
    tv_all = df["总资产"].values
    cum_all = tv_all / tv_all[0]
    peak_all = np.maximum.accumulate(cum_all)
    total_dd = (cum_all - peak_all).min() / peak_all[np.argmin(cum_all - peak_all)] if len(peak_all) > 0 else 0

    return {
        "month_return": month_return,
        "month_dd": month_dd,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "total_dd": total_dd,
        "n_days": n_all,
    }


def build_monthly_report(output_dir: Optional[str] = None) -> str:
    """
    读取所有策略CSV，生成月度报告文本。

    Args:
        output_dir: simulation/output 目录路径，默认自动查找。

    Returns:
        格式化的月度报告文本。
    """
    if output_dir is None:
        output_dir = str(
            Path(__file__).resolve().parent.parent / "output"
        )

    out = Path(output_dir)
    if not out.exists():
        return "📊 ETF模拟盘月度报告\n\n暂无数据"

    today = date.today()
    # 确定报告月份
    if today.day <= 3:
        # 月初前几天，报告上个月
        report_month = today.replace(day=1) - timedelta(days=1)
    else:
        report_month = today
    month_label = report_month.strftime("%Y年%m月")
    month_start = report_month.strftime("%Y-%m-01")
    # 当月最后一天
    if report_month.month == 12:
        month_end = report_month.replace(year=report_month.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        month_end = report_month.replace(month=report_month.month + 1, day=1) - timedelta(days=1)
    month_end_str = month_end.strftime("%Y-%m-%d")

    # 收集所有策略数据
    strategies = []
    for csv_file in sorted(out.glob("sim_log_*.csv")):
        sid = csv_file.stem.replace("sim_log_", "")
        name = STRATEGY_NAMES.get(sid, sid)

        try:
            df = pd.read_csv(str(csv_file), encoding="utf-8-sig")
        except Exception:
            continue

        if df.empty or "总资产" not in df.columns:
            continue

        # 初始资金（从 state JSON 或 CSV 推断）
        state_file = out / f"state_{sid}.json"
        initial_capital = 10000.0
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                initial_capital = float(state.get("initial_capital", 10000))
            except Exception:
                pass

        metrics = _compute_monthly_metrics(df, initial_capital, month_start, month_end_str)
        if not metrics:
            continue

        # 当日状态
        today_str = today.strftime("%Y-%m-%d")
        today_row = df[df["日期"] == today_str]
        hold_info = ""
        if not today_row.empty:
            row = today_row.iloc[-1]
            holding = str(row.get("操作", ""))
            pending = str(row.get("明日待执行", ""))
            if holding and holding not in ["nan", ""]:
                hold_info = holding
            if pending and pending not in ["nan", ""]:
                hold_info += f" 待:{pending}"

        strategies.append({
            "name": name,
            "sid": sid,
            **metrics,
            "hold_info": hold_info,
            "initial_capital": initial_capital,
        })

    if not strategies:
        return f"📊 ETF模拟盘月度报告 | {month_label}\n\n暂无有效数据"

    # 按累计收益排序
    strategies.sort(key=lambda s: s.get("total_return", -1), reverse=True)

    # ── 构建文本 ──
    lines = [
        f"📊 ETF模拟盘月度报告 | {month_label}",
        "═" * 42,
        "",
    ]

    # 表头
    lines.append(
        f"  {'策略':<16} {'当月':>7} {'累计':>8} {'夏普':>5} {'回撤':>7}"
    )
    lines.append("  " + "─" * 44)

    # 分类显示
    independent = {"黄金避险 🥇", "跨境轮动 🌏"}
    batch_strategies = []
    independent_strategies = []

    for s in strategies:
        if s["name"] in independent:
            independent_strategies.append(s)
        else:
            batch_strategies.append(s)

    # 动量类
    for s in batch_strategies:
        mr = s.get("month_return", 0)
        tr = s.get("total_return", 0)
        sh = s.get("sharpe")
        dd = s.get("total_dd", 0)
        sh_str = f"{sh:.2f}" if sh is not None else "  N/A"
        lines.append(
            f"  {s['name']:<16} {mr:>+6.1%} {tr:>+7.1%} {sh_str:>5} {dd:>6.1%}"
        )

    # 分隔
    if independent_strategies:
        lines.append("")
        for s in independent_strategies:
            mr = s.get("month_return", 0)
            tr = s.get("total_return", 0)
            sh = s.get("sharpe")
            dd = s.get("total_dd", 0)
            sh_str = f"{sh:.2f}" if sh is not None else "  N/A"
            lines.append(
                f"  {s['name']:<16} {mr:>+6.1%} {tr:>+7.1%} {sh_str:>5} {dd:>6.1%}"
            )

    lines.append("")
    lines.append("═" * 42)
    lines.append(f"  数据截止: {today.strftime('%Y-%m-%d')}")

    return "\n".join(lines)


def push_monthly_report(output_dir: Optional[str] = None) -> bool:
    """生成并推送月度报告。"""
    text = build_monthly_report(output_dir)
    today = date.today()
    month_label = today.strftime("%Y年%m月")
    return send_message(f"📊 ETF月度报告 | {month_label}", text)


# ── CLI入口 ──
if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None
    success = push_monthly_report(out_dir)
    print("月度报告推送成功" if success else "月度报告推送失败")
