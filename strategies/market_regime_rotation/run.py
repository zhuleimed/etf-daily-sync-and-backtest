"""
市场状态感知动量轮动策略 — 入口脚本

用法:
  # 默认参数
  python strategies/market_regime_rotation/run.py

  # 自定义区间
  python strategies/market_regime_rotation/run.py --start 2024-01-01 --end 2026-06-23

  # 与 momentum_rotation 对比测试
  python strategies/market_regime_rotation/run.py --tag test1
"""

import argparse
import os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import (
    COMMISSION_RATE,
    SLIPPAGE,
    INITIAL_CAPITAL,
    START_DATE,
    OUTPUT_DIR,
    BULL_MA_PERIOD,
    BULL_MOMENTUM_WINDOW,
    BULL_MOMENTUM_THRESHOLD,
)
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="市场状态感知动量轮动策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 55}")
    print(f"  市场状态感知动量轮动策略")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  牛市判定: 沪深300>{BULL_MA_PERIOD}日均线+"
          f"{BULL_MOMENTUM_WINDOW}日动量>{BULL_MOMENTUM_THRESHOLD:.0%}")
    print(f"  牛市时锁仓不动，熊市时动量轮动")
    print(f"  交易费用: 佣金{COMMISSION_RATE:.2%}, 滑点{SLIPPAGE:.2%}")
    print(f"  {'=' * 55}")

    engine = BacktestEngine(initial_capital=args.money)
    print("  [1/4] 加载数据…")
    engine.load_data(start_date=args.start, end_date=args.end)
    print("  [2/4] 运行回测…")
    engine.run()

    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()

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
    reporter.plot_equity_curve(daily_df, benchmark_data, ew_data)
    reporter.plot_drawdown(daily_df)
    reporter.plot_holding_heatmap(daily_df)
    reporter.plot_monthly_returns(daily_df)
    reporter.print_summary(metrics)

    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
