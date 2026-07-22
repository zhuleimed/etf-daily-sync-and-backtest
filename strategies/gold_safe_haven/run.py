#!/usr/bin/env python3
"""
黄金避险轮动策略回测 — 入口脚本

用法:
  # 默认参数回测
  python -m strategies.gold_safe_haven.run

  # 自定义参数
  python -m strategies.gold_safe_haven.run --start 2024-01-01 --money 10000

  # 分年份回测（4期对比用）
  python -m strategies.gold_safe_haven.run --start 2024-01-01 --end 2024-12-31
"""
import argparse
import os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .data import load_benchmark_data, compute_equal_weight_benchmark
from . import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="黄金避险轮动策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=str, default=cfg.START_DATE,
                        help=f"回测开始日期，默认 {cfg.START_DATE}")
    parser.add_argument("--end", type=str, default="",
                        help="回测结束日期，默认到最新数据")
    parser.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL,
                        help=f"初始资金，默认 {cfg.INITIAL_CAPITAL} 元")
    parser.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW,
                        help=f"动量窗口，默认 {cfg.MOMENTUM_WINDOW} 日")
    parser.add_argument("--tag", type=str, default="",
                        help="回测标记，用于输出目录命名")
    parser.add_argument("--panic-threshold", type=float, default=cfg.PANIC_THRESHOLD,
                        help=f"恐慌触发阈值，默认 {cfg.PANIC_THRESHOLD}")
    return parser.parse_args()


def run_backtest(
    start_date: str,
    end_date: str,
    initial_capital: float,
    momentum_window: int,
    output_dir: str,
    panic_threshold: float,
    verbose: bool = True,
) -> dict:
    """
    运行单次回测，返回结果字典。

    可用于批量运行（4期对比等）。
    """
    # 临时覆盖恐慌阈值
    original_threshold = cfg.PANIC_THRESHOLD
    cfg.PANIC_THRESHOLD = panic_threshold

    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            momentum_window=momentum_window,
        )

        engine.load_data(start_date=start_date, end_date=end_date)
        engine.run()

        daily_df = engine.get_daily_df()
        trade_df = engine.get_trade_df()

        # 基准数据
        try:
            bench_data = load_benchmark_data(start_date=start_date, end_date=end_date)
            # 对齐日期
            strategy_dates = set(daily_df["date"].values)
            bench_data = bench_data[bench_data["date"].astype(str).isin(strategy_dates)]
            bench_return = bench_data["cumulative_returns"].iloc[-1] - 1 if len(bench_data) > 0 else None
        except Exception:
            bench_data = None
            bench_return = None

        try:
            ew_data = compute_equal_weight_benchmark(engine.etf_data)
            ew_return = ew_data["cumulative_returns"].iloc[-1] - 1 if len(ew_data) > 0 else None
        except Exception:
            ew_data = None
            ew_return = None

        calc = MetricsCalculator(risk_free_rate=0.03)
        metrics = calc.compute(
            engine.daily_records,
            engine.trade_records,
            initial_capital=initial_capital,
            benchmark_return=bench_return,
            ew_benchmark_return=ew_return,
        )

        # 恐慌统计
        panic_days = len(daily_df[daily_df["mode"].str.startswith("gold")])
        panic_entries = len(daily_df[daily_df["action"] == "panic_entry"])
        total_days = len(daily_df)

        result = {
            "start": start_date,
            "end": end_date if end_date else daily_df["date"].iloc[-1],
            "metrics": metrics,
            "daily_df": daily_df,
            "trade_df": trade_df,
            "bench_data": bench_data,
            "ew_data": ew_data,
            "total_days": total_days,
            "panic_days": panic_days,
            "panic_entries": panic_entries,
            "panic_pct": panic_days / total_days * 100 if total_days > 0 else 0,
        }

        if verbose:
            print(f"\n  恐慌统计: 触发{panic_entries}次, "
                  f"黄金持仓{panic_days}天({result['panic_pct']:.1f}%)")

        return result

    finally:
        cfg.PANIC_THRESHOLD = original_threshold


def main():
    args = parse_args()

    print(f"\n{'=' * 55}")
    print(f"  黄金避险轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  动量窗口: {args.momentum} 日")
    print(f"  恐慌阈值: {args.panic_threshold}")
    print(f"  黄金标的: {cfg.GOLD_SYMBOL} ({cfg.ETF_POOL[cfg.GOLD_SYMBOL]})")
    print(f"  最小黄金持有: {cfg.MIN_GOLD_HOLD}天  最长: {cfg.GOLD_MAX_HOLD}天")
    print(f"  黄金止损: {cfg.GOLD_STOP_LOSS*100:.0f}%")
    print(f"  {'=' * 55}")

    print("  [1/4] 加载数据…")
    print("  [2/4] 运行回测…")
    print("  [3/4] 计算绩效…")
    print("  [4/4] 生成报告…")

    result = run_backtest(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.money,
        momentum_window=args.momentum,
        output_dir=cfg.OUTPUT_DIR,
        panic_threshold=args.panic_threshold,
        verbose=True,
    )

    metrics = result["metrics"]
    daily_df = result["daily_df"]
    trade_df = result["trade_df"]

    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    output_dir = os.path.join(cfg.OUTPUT_DIR, f"{timestamp}{tag}")
    reporter = Reporter(output_dir=output_dir)

    reporter.save_daily_records(daily_df)
    reporter.save_trade_records(trade_df)
    reporter.save_metrics(metrics)

    reporter.plot_equity_curve(daily_df, result["bench_data"], result["ew_data"])
    reporter.plot_drawdown(daily_df)

    reporter.print_summary(metrics)

    # 恐慌模式详细统计
    print(f"\n  [恐慌避险统计]")
    print(f"    恐慌触发次数: {result['panic_entries']}")
    print(f"    黄金持仓天数: {result['panic_days']}/{result['total_days']} ({result['panic_pct']:.1f}%)")

    # 黄金持仓期间收益 vs 宽基持仓收益
    gold_days = daily_df[daily_df["mode"].str.startswith("gold")]
    if len(gold_days) > 0:
        gold_ret = gold_days["cumulative_return"].iloc[-1] - gold_days["cumulative_return"].iloc[0] if len(gold_days) > 1 else 0
        print(f"    黄金持仓期策略收益: {gold_ret*100:.2f}%")

    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
