#!/usr/bin/env python3
"""
反转/均值回归策略回测 — 入口脚本

用法：
  # 默认参数回测
  python -m strategies.contrarian_reversion.run

  # 网格搜索最优参数
  python -m strategies.contrarian_reversion.run --scan

  # 单参数覆盖
  python -m strategies.contrarian_reversion.run --window 10 --entry -0.03
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .engine import ContrarianEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import (
    load_benchmark_data, compute_equal_weight_benchmark,
)

from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="反转/均值回归策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--window", type=int, default=cfg.REVERSION_WINDOW,
                   help="反转回看窗口（日）")
    p.add_argument("--entry", type=float, default=cfg.ENTRY_THRESHOLD,
                   help="入场阈值（如 -0.02 = 跌2%才入场）")
    p.add_argument("--profit", type=float, default=cfg.PROFIT_TARGET,
                   help="止盈目标（如 0.03 = 涨3%止盈）")
    p.add_argument("--stop", type=float, dest="stop_loss", default=cfg.STOP_LOSS,
                   help="止损线（如 -0.08 = 跌8%止损）")
    p.add_argument("--max-hold", type=int, default=cfg.MAX_HOLD_DAYS,
                   help="最大持仓天数")
    p.add_argument("--min-hold", type=int, default=cfg.MIN_HOLD_DAYS,
                   help="最小持仓天数")
    p.add_argument("--conviction", type=float, default=cfg.MIN_SWITCH_CONVICTION,
                   help="切换置信度")
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A", "B", "C"])
    p.add_argument("--scan", action="store_true", help="网格搜索最优参数")
    p.add_argument("--tag", default="")
    return p.parse_args()


# ============================================================================
# 单次回测
# ============================================================================

def run_backtest(start_date, end_date, initial_capital,
                 reversion_window, entry_threshold, profit_target,
                 stop_loss, max_hold_days, min_hold_days,
                 min_switch_conviction, risk_mode,
                 verbose=True):
    """运行单次回测，返回 {start, end, metrics, daily_df, trade_df, ...}"""
    engine = ContrarianEngine(
        initial_capital=initial_capital,
        risk_mode=risk_mode,
        momentum_window=reversion_window,
        reversion_window=reversion_window,
        entry_threshold=entry_threshold,
        profit_target=profit_target,
        stop_loss=stop_loss,
        max_hold_days=max_hold_days,
        min_hold_days=min_hold_days,
        min_switch_conviction=min_switch_conviction,
    )

    # 加载数据
    from strategies.momentum_rotation.data import load_all_etf_data as _load
    etf_data, dates = _load(
        symbols=cfg.ETF_SYMBOLS,
        start_date=start_date,
        end_date=end_date,
        db_path=cfg.DB_PATH,
        momentum_window=reversion_window,
    )
    engine.etf_data = etf_data
    engine.dates = dates

    if verbose:
        print(f"加载 {len(etf_data)} 只ETF，{len(dates)} 个交易日")

    engine.run()

    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()

    # 基准
    try:
        bench_data = load_benchmark_data(start_date=start_date, end_date=end_date)
        bench_return = bench_data["cumulative_returns"].iloc[-1] - 1 if len(bench_data) > 0 else None
    except Exception:
        bench_data, bench_return = None, None

    try:
        ew_data = compute_equal_weight_benchmark(etf_data)
        ew_return = ew_data["cumulative_returns"].iloc[-1] - 1 if len(ew_data) > 0 else None
    except Exception:
        ew_data, ew_return = None, None

    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(
        engine.daily_records, engine.trade_records,
        initial_capital=initial_capital,
        benchmark_return=bench_return,
        ew_benchmark_return=ew_return,
    )

    return {
        "start": start_date,
        "end": end_date if end_date else daily_df["date"].iloc[-1],
        "metrics": metrics,
        "daily_df": daily_df,
        "trade_df": trade_df,
        "bench_data": bench_data,
        "ew_data": ew_data,
    }


# ============================================================================
# 网格搜索
# ============================================================================

def grid_search():
    """网格搜索最优反转参数组合。"""
    print("\n" + "=" * 70)
    print("  反转/均值回归策略 — 网格搜索")
    print("=" * 70)

    # 搜索空间
    windows = [5, 10, 15]              # 反转回看窗口
    entries = [-0.02, -0.03, -0.05]    # 入场阈值
    profits = [0.02, 0.03, 0.05]       # 止盈目标
    stops = [-0.05, -0.08, -0.10]      # 止损线
    max_holds = [10, 15, 20]           # 最大持仓天数
    min_holds = [3, 5]                 # 最小持仓天数

    total = len(windows) * len(entries) * len(profits) * len(stops) * len(max_holds) * len(min_holds)
    print(f"  总组合数: {total}")
    print(f"  预计耗时: ~{total * 0.15:.0f}s\n")

    results = []
    count = 0

    for w, ent, prf, stp, mxh, mnh in product(windows, entries, profits, stops, max_holds, min_holds):
        count += 1
        try:
            r = run_backtest(
                start_date="2024-01-01", end_date="",
                initial_capital=cfg.INITIAL_CAPITAL,
                reversion_window=w,
                entry_threshold=ent,
                profit_target=prf,
                stop_loss=stp,
                max_hold_days=mxh,
                min_hold_days=mnh,
                min_switch_conviction=cfg.MIN_SWITCH_CONVICTION,
                risk_mode="B",
                verbose=False,
            )
            m = r["metrics"]
            results.append({
                "window": w, "entry": ent, "profit": prf, "stop": stp,
                "max_hold": mxh, "min_hold": mnh,
                "return": m.total_return,
                "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown,
                "trades": m.total_trades,
                "win_rate": m.win_rate,
            })
            print(f"  [{count:3d}/{total}] w={w} ent={ent:.0%} prf={prf:.0%} "
                  f"stp={stp:.0%} mxh={mxh} mnh={mnh} | "
                  f"收益={m.total_return:+.1%} 夏普={m.sharpe_ratio:.2f} "
                  f"DD={m.max_drawdown:.1%} 交易={m.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:3d}/{total}] ✗ 错误: {e}", flush=True)

    # 排序输出
    results.sort(key=lambda x: x["return"], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP 20 参数组合（按收益排序）")
    print("=" * 70)
    print(f"  {'排名':<4} {'窗':>3} {'入场':>6} {'止盈':>6} {'止损':>6} "
          f"{'最大持':>5} {'最小持':>5} {'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    print(f"  {'-' * 65}")

    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<4} {r['window']:>3} {r['entry']:>6.0%} {r['profit']:>6.0%} "
              f"{r['stop']:>6.0%} {r['max_hold']:>5} {r['min_hold']:>5} "
              f"{r['return']:>+7.1%} {r['sharpe']:>5.2f} {r['max_dd']:>6.1%} "
              f"{r['trades']:>4} {r['win_rate']:>5.1%}")

    return results


# ============================================================================
# 主入口
# ============================================================================

def main():
    args = parse_args()

    if args.scan:
        grid_search()
        return

    print(f"\n{'=' * 55}")
    print(f"  反转/均值回归策略回测")
    print(f"  {'=' * 55}")
    print(f"  ETF池: {len(cfg.ETF_SYMBOLS)}只宽基ETF")
    print(f"  反转窗口: {args.window}日  入场阈值: {args.entry:.0%}")
    print(f"  止盈: {args.profit:.0%}  止损: {args.stop_loss:.0%}")
    print(f"  最大持有: {args.max_hold}天  最小持有: {args.min_hold}天")
    print(f"  风控模式: {args.risk_mode}")
    print(f"  {'=' * 55}")

    result = run_backtest(
        start_date=args.start, end_date=args.end,
        initial_capital=args.money,
        reversion_window=args.window,
        entry_threshold=args.entry,
        profit_target=args.profit,
        stop_loss=args.stop_loss,
        max_hold_days=args.max_hold,
        min_hold_days=args.min_hold,
        min_switch_conviction=args.conviction,
        risk_mode=args.risk_mode,
    )

    metrics = result["metrics"]
    daily_df = result["daily_df"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = os.path.join(cfg.OUTPUT_DIR, f"{timestamp}{tag}")
    reporter = Reporter(output_dir=output_dir)

    reporter.save_daily_records(daily_df)
    reporter.save_trade_records(result["trade_df"])
    reporter.save_metrics(metrics)
    reporter.plot_equity_curve(daily_df, result["bench_data"], result["ew_data"])
    reporter.plot_drawdown(daily_df)
    reporter.print_summary(metrics)

    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
