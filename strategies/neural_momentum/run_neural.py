"""Neural Momentum 回测入口：动量 + 神经网络混合评分

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.neural_momentum.run_neural \
      --weight-w 0.5 --start 2024-01-01 --tag w05_2024
  --weight-w 1.0 = 纯动量（同池同引擎基准）
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from strategies.neural_momentum.config import (
    INITIAL_CAPITAL, MOMENTUM_WINDOW, ADJUSTMENT_DAYS,
    DYNAMIC_WINDOW_ENABLED, TOP_N, START_DATE, RISK_MODE,
    COMMISSION_RATE, SLIPPAGE, NEURAL_SCORES_PATH, WEIGHT_W,
)
from strategies.neural_momentum.engine import BacktestEngine


def load_neural_scores() -> pd.DataFrame:
    p = Path(PROJECT_ROOT) / NEURAL_SCORES_PATH
    if not p.exists():
        print(f"  ⚠ 神经评分文件不存在: {p}（将退化为纯动量）")
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = df.index.strftime("%Y-%m-%d")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Momentum：动量+神经网络混合评分回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--momentum", type=int, default=MOMENTUM_WINDOW)
    parser.add_argument("--adjust-days", type=int, default=ADJUSTMENT_DAYS)
    parser.add_argument("--risk-mode", type=str, default="", choices=["", "A", "B", "C"])
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--no-dynamic-window", action="store_true")
    parser.add_argument("--weight-w", type=float, default=WEIGHT_W,
                        help="混合权重：w×动量z + (1-w)×神经z（1.0=纯动量）")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    risk_mode = args.risk_mode if args.risk_mode else RISK_MODE
    top_n = args.top_n if args.top_n > 0 else TOP_N
    dynamic_window = not args.no_dynamic_window if args.no_dynamic_window else DYNAMIC_WINDOW_ENABLED

    neural = load_neural_scores()

    print(f"\n{'=' * 55}")
    print(f"  Neural Momentum 回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  混合权重: w={args.weight_w}（{'纯动量' if args.weight_w >= 1.0 else f'{args.weight_w}×动量z + {1-args.weight_w:.1f}×神经z'}）")
    print(f"  神经评分: {len(neural) if neural is not None else 0} 天")
    print(f"  风控模式: {risk_mode} | TOP-N: {top_n} | 调仓周期: {args.adjust_days} 日")
    print(f"  {'=' * 55}")

    engine = BacktestEngine(
        initial_capital=args.money,
        risk_mode=risk_mode,
        momentum_window=args.momentum,
        top_n=top_n,
        dynamic_window=dynamic_window,
        neural_scores=neural,
        weight_w=args.weight_w,
    )

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

    from strategies.neural_momentum.metrics import MetricsCalculator
    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(
        engine.daily_records, engine.trade_records,
        initial_capital=args.money,
        benchmark_return=bench_total_return,
        ew_benchmark_return=ew_total_return,
    )

    print("  [4/4] 生成报告…")
    from datetime import datetime
    import os
    from strategies.neural_momentum.reporter import Reporter
    from strategies.neural_momentum.config import OUTPUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else f"_w{args.weight_w:g}"
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
