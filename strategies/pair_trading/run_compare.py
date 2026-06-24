#!/usr/bin/env python
"""
配对交易 → 纯多头轮动 — 三模式对比回测

同时运行 A/B/C 三种模式，输出对比结果。

用法：
    python -m strategies.pair_trading.run_compare
    python -m strategies.pair_trading.run_compare --start 2024-01-01 --end 2026-06-22
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .config import (
    PAIRS, INITIAL_CAPITAL, CAPITAL_PER_PAIR,
    ZSCORE_OPEN, ZSCORE_CLOSE, ZSCORE_STOP,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, OUTPUT_DIR, ZSCORE_PERIOD,
)
from strategies.momentum_rotation.data import load_all_etf_data
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from .engine_switch import (
    run_mode_a, run_mode_b, run_mode_c, SwitchRecord,
)


def parse_args():
    p = argparse.ArgumentParser(description="配对交易纯多头轮动 — 三模式对比")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    return p.parse_args()


def records_to_df(records: list[SwitchRecord], mode: str) -> pd.DataFrame:
    """SwitchRecord list → DataFrame"""
    rows = []
    for r in records:
        rows.append({
            "date": r.date,
            "mode": r.mode,
            "hold_symbols": r.hold_symbols,
            "cash": r.cash,
            "stock_value": r.stock_value,
            "total_value": r.total_value,
            "daily_return": r.daily_return,
            "cumulative_return": r.cumulative_return,
            "action": r.action,
            "z_scores": r.z_scores,
            "signals": r.signals,
        })
    return pd.DataFrame(rows)


def compute_metrics(daily_df: pd.DataFrame, initial_capital: float) -> dict:
    """简易指标计算（不依赖 MetricsCalculator，避免格式兼容问题）。"""
    if daily_df.empty:
        return {"error": "无数据"}

    values = daily_df["total_value"].values
    rets = daily_df["daily_return"].values

    total_ret = values[-1] / initial_capital - 1

    # 年化（交易日 ~245）
    n_days = len(values)
    annual_ret = (values[-1] / initial_capital) ** (245 / max(n_days, 1)) - 1 if n_days > 0 else 0

    # 夏普
    excess = rets.mean() - 0.03 / 245  # 无风险利率 3%
    std = rets.std()
    sharpe = (excess / std) * np.sqrt(245) if std > 1e-8 else 0

    # 最大回撤
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    max_dd = np.max(dd) if len(dd) > 0 else 0

    # 交易统计
    actions = daily_df["action"].values
    trade_days = sum(1 for a in actions if a != "hold" and a != "")
    hold_days = sum(1 for a in actions if a == "hold" or a == "")
    win_days = sum(1 for r in rets if r > 0)
    loss_days = sum(1 for r in rets if r < 0)
    win_rate = win_days / max(win_days + loss_days, 1)

    # 持仓天数占比
    has_position = sum(1 for r in daily_df["hold_symbols"].values if "现金" not in str(r) and r != "")
    position_pct = has_position / max(len(daily_df), 1)

    return {
        "总收益": f"{total_ret*100:+.2f}%",
        "年化收益": f"{annual_ret*100:+.2f}%",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd*100:.2f}%",
        "交易天数": trade_days,
        "持仓天数比": f"{position_pct*100:.0f}%",
        "涨跌比": f"{win_rate*100:.0f}%",
        "总净值": f"{values[-1]:.2f}",
    }


def print_comparison(all_results: dict):
    """打印对比表。"""
    print("\n" + "=" * 65)
    print("  配对交易 → 纯多头轮动 | 三模式对比")
    print("=" * 65)

    headers = ["指标", "A.单对最强+全仓", "B.三对均分+轮动", "C.加权分配"]
    col_width = 22
    print()
    print(f"  {'':<12} {'A.单对最强+全仓':<{col_width}} {'B.三对均分+轮动':<{col_width}} {'C.加权分配':<{col_width}}")
    print(f"  {'':-<12} {'':-<{col_width}} {'':-<{col_width}} {'':-<{col_width}}")

    metrics_keys = ["总收益", "年化收益", "夏普比率", "最大回撤", "交易天数", "持仓天数比", "涨跌比"]

    for key in metrics_keys:
        vals = []
        for m in ["A", "B", "C"]:
            d = all_results.get(m, {})
            vals.append(d.get(key, "—"))
        print(f"  {key:<10} {vals[0]:<{col_width}} {vals[1]:<{col_width}} {vals[2]:<{col_width}}")

    print(f"  {'':-<12} {'':-<{col_width}} {'':-<{col_width}} {'':-<{col_width}}")
    print()

    # 交易次数明细
    print(f"  {'交易统计':-^50}")
    for m in ["A", "B", "C"]:
        trades = all_results.get(m, {}).get("_trades", [])
        if trades:
            buy_count = sum(1 for t in trades if "买入" in t.get("action", "") or "买" in t.get("action", ""))
            sell_count = sum(1 for t in trades if "卖出" in t.get("action", "") or "平" in t.get("action", "") or "损" in t.get("action", ""))
            print(f"  {m}模式: {len(trades)}笔操作 (买入{buy_count}次/卖出{sell_count}次)")
        else:
            print(f"  {m}模式: 无交易记录")

    print("\n" + "=" * 65)


def main():
    args = parse_args()

    print(f"\n{'=' * 55}")
    print(f"  配对交易 → 纯多头轮动 | 三模式对比")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  配对: {len(PAIRS)} 对")
    for p in PAIRS:
        print(f"    {p['name']}: {p['a']} ↔ {p['b']}")
    print(f"  参数: z开>{ZSCORE_OPEN} 平<{ZSCORE_CLOSE} 停>{ZSCORE_STOP}")
    print(f"  窗口: {ZSCORE_PERIOD}日 | 佣金{COMMISSION_RATE:.2%} | 滑点{SLIPPAGE:.2%}")
    print(f"  {'=' * 55}")

    # ── 加载数据 ──
    symbols = set()
    for p in PAIRS:
        symbols.add(p["a"])
        symbols.add(p["b"])
    symbols = list(symbols)

    print("\n  [1/4] 加载数据…")
    etf_data, dates = load_all_etf_data(
        symbols=symbols, start_date=args.start,
        end_date=args.end, db_path=DB_PATH,
        momentum_window=ZSCORE_PERIOD,
    )
    print(f"         {len(dates)} 个交易日 | {len(symbols)} 只 ETF")

    # ── 运行回测 ──
    all_results = {}

    for mode_name, mode_func in [("A", run_mode_a), ("B", run_mode_b), ("C", run_mode_c)]:
        print(f"\n  [2/4] 运行模式 {mode_name}…")
        records, trades = mode_func(etf_data, dates, initial_capital=args.money)
        df = records_to_df(records, mode_name)
        metrics = compute_metrics(df, args.money)
        metrics["_records"] = records
        metrics["_trades"] = trades
        all_results[mode_name] = metrics
        print(f"         收益 {metrics['总收益']} | 回撤 {metrics['最大回撤']} | 夏普 {metrics['夏普比率']} | 交易 {metrics['交易天数']}天")

    # ── 打印对比 ──
    print("\n  [3/4] 生成对比报告…")
    print_comparison(all_results)

    # ── 输出 CSV ──
    print("  [4/4] 保存明细…")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, f"compare_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    for m in ["A", "B", "C"]:
        df = records_to_df(all_results[m]["_records"], m)
        # 只保留必要列
        out_df = df[["date", "action", "hold_symbols", "cash", "stock_value",
                      "total_value", "daily_return", "cumulative_return", "z_scores"]]
        out_df.to_csv(os.path.join(out_dir, f"mode_{m}.csv"), index=False)

    # 保存对比摘要
    summary = pd.DataFrame({
        "指标": ["总收益", "年化收益", "夏普比率", "最大回撤", "交易天数", "持仓天数比", "涨跌比"],
        "A.单对最强+全仓": [all_results["A"].get(k, "") for k in ["总收益", "年化收益", "夏普比率", "最大回撤", "交易天数", "持仓天数比", "涨跌比"]],
        "B.三对均分+轮动": [all_results["B"].get(k, "") for k in ["总收益", "年化收益", "夏普比率", "最大回撤", "交易天数", "持仓天数比", "涨跌比"]],
        "C.加权分配": [all_results["C"].get(k, "") for k in ["总收益", "年化收益", "夏普比率", "最大回撤", "交易天数", "持仓天数比", "涨跌比"]],
    })
    summary.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)

    print(f"  输出目录: {os.path.abspath(out_dir)}")
    print(f"\n{'=' * 55}")
    print(f"  回测完成 ✓")
    print(f"  {'=' * 55}\n")


if __name__ == "__main__":
    main()
