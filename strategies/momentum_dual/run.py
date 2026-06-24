#!/usr/bin/env python3
"""
宽基ETF动量轮动策略回测 — 入口脚本

用法:
  # 默认参数回测
  python strategies/momentum_rotation/run.py

  # 自定义参数
  python strategies/momentum_rotation/run.py \\
    --start 2024-01-01 \\
    --end 2026-06-23 \\
    --money 10000 \\
    --momentum 20 \\
    --adjust-days 5

  # 带自定义标记保存不同参数组合的结果
  python strategies/momentum_rotation/run.py --tag test_v1 --momentum 10
"""

import argparse
import os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import (
    ETF_SYMBOLS,
    INITIAL_CAPITAL,
    START_DATE,
    MOMENTUM_WINDOW,
    COMMISSION_RATE,
    SLIPPAGE,
    ADJUSTMENT_DAYS,
    BENCHMARK_SYMBOL,
    OUTPUT_DIR,
    RISK_MODE,
    TOP_N,
    DYNAMIC_WINDOW_ENABLED,
    ABSOLUTE_MOMENTUM_FILTER,
)
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="宽基ETF动量轮动策略回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 时间参数
    parser.add_argument("--start", type=str, default=START_DATE,
                        help=f"回测开始日期，默认 {START_DATE}")
    parser.add_argument("--end", type=str, default="",
                        help="回测结束日期，默认到最新数据")

    # 策略参数
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL,
                        help=f"初始资金，默认 {INITIAL_CAPITAL} 元")
    parser.add_argument("--momentum", type=int, default=MOMENTUM_WINDOW,
                        help=f"动量窗口（N日），默认 {MOMENTUM_WINDOW}")
    parser.add_argument("--adjust-days", type=int, default=ADJUSTMENT_DAYS,
                        help=f"调仓周期（渐进天数），默认 {ADJUSTMENT_DAYS}")

    # 输出
    parser.add_argument("--tag", type=str, default="",
                        help="回测标记，用于输出目录命名")
    parser.add_argument("--risk-mode", type=str, default="",
                        choices=["", "A", "B", "C"],
                        help="风控模式: A=纯信号, B=全开, C=仅极端回撤")
    parser.add_argument("--top-n", type=int, default=0,
                        help="持有动量前N只ETF（0=用config默认）")
    parser.add_argument("--no-dynamic-window", action="store_true",
                        help="禁用动态动量窗口")

    return parser.parse_args()


def main():
    args = parse_args()

    # 风控模式：命令行优先，否则用 config 默认
    risk_mode = args.risk_mode if args.risk_mode else RISK_MODE
    top_n = args.top_n if args.top_n > 0 else TOP_N
    dynamic_window = not args.no_dynamic_window if args.no_dynamic_window else DYNAMIC_WINDOW_ENABLED

    # 显示参数
    print(f"\n{'=' * 55}")
    print(f"  宽基ETF动量轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    if dynamic_window:
        print(f"  动量窗口: 动态(10/20日)")
    else:
        print(f"  动量窗口: {args.momentum} 日")
    print(f"  调仓周期: {args.adjust_days} 日")
    print(f"  交易费用: 佣金{COMMISSION_RATE:.2%}, 滑点{SLIPPAGE:.2%}, 无印花税")
    mode_names = {"A": "纯信号（无风控）", "B": "全开", "C": "仅极端回撤"}
    print(f"  风控模式: {risk_mode} = {mode_names.get(risk_mode, risk_mode)}")
    print(f"  TOP-N:   {top_n}（持有前{top_n}名等权）")
    if ABSOLUTE_MOMENTUM_FILTER:
        print(f"  双动量:  开启（动量≤0则空仓，不扛下跌）")
    else:
        print(f"  双动量:  关闭")
    print(f"  {'=' * 55}")

    # ---- 创建引擎 ----
    engine = BacktestEngine(initial_capital=args.money,
                            risk_mode=risk_mode,
                            momentum_window=args.momentum,
                            top_n=top_n,
                            dynamic_window=dynamic_window)

    # ---- 加载数据 ----
    print("  [1/4] 加载数据…")
    engine.load_data(start_date=args.start, end_date=args.end)

    # ---- 运行回测 ----
    print("  [2/4] 运行回测…")
    engine.run()

    # ---- 计算绩效 ----
    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()

    # 对齐基准数据
    benchmark_data = engine.benchmark_data
    ew_data = engine.equal_weight_data

    bench_total_return = None
    if not benchmark_data.empty:
        bench_total_return = benchmark_data["cumulative_returns"].iloc[-1] - 1

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

    # ---- 生成报告 ----
    print("  [4/4] 生成报告…")

    # 输出目录按时间戳组织
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = os.path.join(OUTPUT_DIR, f"{timestamp}{tag}")
    reporter = Reporter(output_dir=output_dir)

    # 保存CSV
    reporter.save_daily_records(daily_df)
    reporter.save_trade_records(trade_df)
    reporter.save_metrics(metrics)

    # 绘制图表
    reporter.plot_equity_curve(daily_df, benchmark_data, ew_data)
    reporter.plot_drawdown(daily_df)
    reporter.plot_holding_heatmap(daily_df)
    reporter.plot_monthly_returns(daily_df)

    # 打印摘要
    reporter.print_summary(metrics)

    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
