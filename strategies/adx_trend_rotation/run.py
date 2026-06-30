#!/usr/bin/env python3
"""
ADX 趋势强度轮动策略回测 — 入口脚本

用法:
  python strategies/adx_trend_rotation/run.py
  python strategies/adx_trend_rotation/run.py --risk-mode B --tag test
"""

import argparse
import os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import (
    INITIAL_CAPITAL, START_DATE, COMMISSION_RATE, SLIPPAGE,
    OUTPUT_DIR, RISK_MODE, ADX_MIN_STRENGTH,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ADX趋势强度轮动策略回测")
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--risk-mode", type=str, default="", choices=["", "A", "B", "C"])
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    risk_mode = args.risk_mode if args.risk_mode else RISK_MODE

    print(f"\n{'=' * 55}")
    print(f"  ADX趋势强度轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  ADX周期: 14, 最小强度: {ADX_MIN_STRENGTH}")
    mode_names = {"A": "纯信号（无风控）", "B": "全开", "C": "仅极端回撤"}
    print(f"  风控模式: {risk_mode} = {mode_names.get(risk_mode, risk_mode)}")
    print(f"  {'=' * 55}")

    engine = BacktestEngine(initial_capital=args.money, risk_mode=risk_mode)
    print("  [1/4] 加载数据…")
    engine.load_data(start_date=args.start, end_date=args.end)
    print("  [2/4] 运行回测…")
    engine.run()

    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()
    index_data = engine.index_data
    ew_data = engine.equal_weight_data
    bench_total_return = index_data["cumulative_returns"].iloc[-1] - 1 if index_data is not None and not index_data.empty else None
    ew_total_return = ew_data["cumulative_returns"].iloc[-1] - 1 if ew_data is not None and not ew_data.empty else None

    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(
        engine.daily_records, engine.trade_records,
        initial_capital=args.money,
        benchmark_return=bench_total_return,
        ew_benchmark_return=ew_total_return,
    )

    print("  [4/4] 生成报告…")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = os.path.join(OUTPUT_DIR, f"{timestamp}{tag}")
    reporter = Reporter(output_dir=output_dir)
    reporter.save_daily_records(daily_df)
    reporter.save_trade_records(trade_df)
    reporter.save_metrics(metrics)
    reporter.plot_equity_curve(daily_df, index_data, ew_data)
    reporter.plot_drawdown(daily_df)
    reporter.plot_holding_heatmap(daily_df)
    reporter.plot_monthly_returns(daily_df)
    reporter.print_summary(metrics)
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
