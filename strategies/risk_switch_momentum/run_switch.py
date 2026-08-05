"""风险状态择时回测入口：动量轮动 + LightGBM 风险开关（基于原引擎副本）。

对比基准：strategies/momentum_rotation（原引擎 +84% 2024-01起）。

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.risk_switch_momentum.run_switch \
      --risk-threshold 0.65 --start 2024-01-01 --tag risk065_2024
  --risk-threshold 1.0 = 关闭风险开关（等价原引擎，用于校验副本一致性）
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.risk_switch_momentum.config import (
    INITIAL_CAPITAL, MOMENTUM_WINDOW, ADJUSTMENT_DAYS,
    DYNAMIC_WINDOW_ENABLED, TOP_N, START_DATE, RISK_MODE,
    COMMISSION_RATE, SLIPPAGE,
)
from strategies.risk_switch_momentum.engine import BacktestEngine
from strategies.risk_switch_momentum.metrics import MetricsCalculator


def load_risk_scores(path: str) -> dict:
    """读取 risk_scores.csv（date, score）→ {date_str: float}"""
    import pandas as pd
    df = pd.read_csv(path)
    return {str(r["date"])[:10]: float(r["score"]) for _, r in df.iterrows()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="动量轮动 + 风险状态择时回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=str, default=START_DATE)
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--momentum", type=int, default=MOMENTUM_WINDOW)
    parser.add_argument("--adjust-days", type=int, default=ADJUSTMENT_DAYS)
    parser.add_argument("--risk-mode", type=str, default="",
                        choices=["", "A", "B", "C"])
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--no-dynamic-window", action="store_true")
    # ── 风险择时参数 ──
    parser.add_argument("--risk-threshold", type=float, default=1.0,
                        help="风险分阈值（>阈值空仓）。1.0=关闭")
    parser.add_argument("--risk-scores", type=str,
                        default=str(Path(__file__).parent.parent / "risk_timing" / "output" / "risk_scores.csv"),
                        help="风险分 CSV 路径")
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    risk_mode = args.risk_mode if args.risk_mode else RISK_MODE
    top_n = args.top_n if args.top_n > 0 else TOP_N
    dynamic_window = not args.no_dynamic_window if args.no_dynamic_window else DYNAMIC_WINDOW_ENABLED

    risk_scores = load_risk_scores(args.risk_scores) if args.risk_threshold < 1.0 else {}

    print(f"\n{'=' * 55}")
    print(f"  动量轮动 + 风险状态择时回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  风险阈值: {args.risk_threshold} ({'关闭' if args.risk_threshold >= 1.0 else f'风险分>{args.risk_threshold}空仓'})")
    print(f"  风险分数: {len(risk_scores)} 天" if risk_scores else "  风险分数: 未启用")
    print(f"  风控模式: {risk_mode} | TOP-N: {top_n} | 调仓周期: {args.adjust_days} 日")
    print(f"  {'=' * 55}")

    engine = BacktestEngine(
        initial_capital=args.money,
        risk_mode=risk_mode,
        momentum_window=args.momentum,
        top_n=top_n,
        dynamic_window=dynamic_window,
        risk_scores=risk_scores,
        risk_threshold=args.risk_threshold,
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

    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(
        engine.daily_records,
        engine.trade_records,
        initial_capital=args.money,
        benchmark_return=bench_total_return,
        ew_benchmark_return=ew_total_return,
    )

    # ---- 生成报告（与原 run.py 同构） ----
    print("  [4/4] 生成报告…")
    from datetime import datetime
    import os
    from strategies.risk_switch_momentum.reporter import Reporter
    from strategies.risk_switch_momentum.config import OUTPUT_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else f"_risk{args.risk_threshold:g}"
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
