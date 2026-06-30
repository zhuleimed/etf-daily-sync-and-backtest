#!/usr/bin/env python3
"""
复合动量轮动策略回测 — 入口脚本

用法:
  # 默认参数回测
  python strategies/composite_momentum/run.py

  # 自定义参数
  python strategies/composite_momentum/run.py \\
    --start 2024-01-01 \\
    --end 2026-06-29 \\
    --money 10000 \\
    --risk-mode B

  # 标记不同参数组合
  python strategies/composite_momentum/run.py --tag test_v1
"""

import argparse
import os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import (
    INITIAL_CAPITAL,
    START_DATE,
    COMMISSION_RATE,
    SLIPPAGE,
    BENCHMARK_SYMBOL,
    OUTPUT_DIR,
    RISK_MODE,
)
from .data import load_index_data, compute_equal_weight_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="复合动量轮动策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=str, default=START_DATE,
                        help=f"回测开始日期，默认 {START_DATE}")
    parser.add_argument("--end", type=str, default="",
                        help="回测结束日期，默认到最新数据")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL,
                        help=f"初始资金，默认 {INITIAL_CAPITAL} 元")
    parser.add_argument("--risk-mode", type=str, default="",
                        choices=["", "A", "B", "C"],
                        help="风控模式: A=纯信号, B=全开, C=仅极端回撤")
    parser.add_argument("--tag", type=str, default="",
                        help="回测标记，用于输出目录命名")
    return parser.parse_args()


def main():
    args = parse_args()
    risk_mode = args.risk_mode if args.risk_mode else RISK_MODE

    print(f"\n{'=' * 55}")
    print(f"  复合动量轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  交易费用: 佣金{COMMISSION_RATE:.2%}, 滑点{SLIPPAGE:.2%}, 无印花税")
    mode_names = {"A": "纯信号（无风控）", "B": "全开", "C": "仅极端回撤"}
    print(f"  风控模式: {risk_mode} = {mode_names.get(risk_mode, risk_mode)}")
    print(f"  {'=' * 55}")

    # 1. 创建引擎
    engine = BacktestEngine(initial_capital=args.money, risk_mode=risk_mode)

    # 2. 加载数据
    print("  [1/4] 加载数据…")
    engine.load_data(start_date=args.start, end_date=args.end)

    # 3. 运行回测
    print("  [2/4] 运行回测…")
    engine.run()

    # 4. 计算绩效
    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()

    # 基准数据
    index_data = engine.index_data
    ew_data = engine.equal_weight_data

    bench_total_return = None
    if index_data is not None and not index_data.empty:
        bench_total_return = index_data["cumulative_returns"].iloc[-1] - 1

    ew_total_return = None
    if ew_data is not None and not ew_data.empty:
        ew_total_return = ew_data["cumulative_returns"].iloc[-1] - 1

    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(
        engine.daily_records,
        engine.trade_records,
        initial_capital=args.money,
        benchmark_return=bench_total_return,
        ew_benchmark_return=ew_total_return,
    )

    # 5. 生成报告
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
