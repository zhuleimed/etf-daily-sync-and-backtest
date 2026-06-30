#!/usr/bin/env python3
"""
均值回归轮动策略回测 — 入口脚本

用法:
  python -m strategies.mean_reversion_rotation.run
  python -m strategies.mean_reversion_rotation.run --start 2024-01-01 --end 2026-06-30
"""

import argparse, os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import INITIAL_CAPITAL, START_DATE, COMMISSION_RATE, SLIPPAGE, OUTPUT_DIR, RISK_MODE


def parse_args():
    parser = argparse.ArgumentParser(description="均值回归轮动策略回测")
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
    print(f"  均值回归轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  核心逻辑: 买超卖(%B<0+RSI<{35}) → 等回归(%B>{0.5}) → 获利了结")
    print(f"  市场过滤: 沪深300>MA200，熊市不做均值回归")
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
    bench_return = index_data["cumulative_returns"].iloc[-1] - 1 if index_data is not None and not index_data.empty else None
    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(engine.daily_records, engine.trade_records, args.money, benchmark_return=bench_return)
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
    reporter.print_summary(metrics)
    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
