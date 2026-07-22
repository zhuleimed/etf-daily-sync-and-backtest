#!/usr/bin/env python3
"""
市场宽度择时策略回测 — 入口脚本

用法：
  python -m strategies.market_breadth.run           # 默认参数
  python -m strategies.market_breadth.run --scan    # 网格搜索
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .engine import BreadthTimingEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark

from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="市场宽度择时策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    p.add_argument("--ma", type=int, default=cfg.BREADTH_MA_PERIOD,
                   help="宽度MA周期（如20日）")
    p.add_argument("--strong", type=float, default=cfg.BREADTH_STRONG,
                   help="强市阈值（如0.7=70%）")
    p.add_argument("--weak", type=float, default=cfg.BREADTH_WEAK,
                   help="弱市阈值（如0.3=30%）")
    p.add_argument("--neutral", default=cfg.NEUTRAL_MODE,
                   choices=["cash", "half"], help="中性市策略")
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A", "B", "C"])
    p.add_argument("--scan", action="store_true", help="网格搜索")
    p.add_argument("--tag", default="")
    return p.parse_args()


def run_backtest(start_date, end_date, initial_capital,
                 momentum_window, ma_period, breadth_strong,
                 breadth_weak, neutral_mode, risk_mode,
                 verbose=True):
    """运行单次回测。"""
    # 覆盖 momentum_rotation 模块中的变量（父类引擎使用的）
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine

    orig_mr_risk = mr_cfg.RISK_MODE
    orig_mr_mom = mr_cfg.MOMENTUM_WINDOW

    mr_cfg.RISK_MODE = risk_mode
    mr_cfg.MOMENTUM_WINDOW = momentum_window
    mr_engine.RISK_MODE = risk_mode
    mr_engine.MOMENTUM_WINDOW = momentum_window

    try:
        engine = BreadthTimingEngine(
            initial_capital=initial_capital,
            risk_mode=risk_mode,
            momentum_window=momentum_window,
            ma_period=ma_period,
            breadth_strong=breadth_strong,
            breadth_weak=breadth_weak,
            neutral_mode=neutral_mode,
        )

        from strategies.momentum_rotation.data import load_all_etf_data as _load
        etf_data, dates = _load(
            symbols=cfg.ETF_SYMBOLS,
            start_date=start_date,
            end_date=end_date,
            db_path=cfg.DB_PATH,
            momentum_window=momentum_window,
        )
        engine.etf_data = etf_data
        engine.dates = dates

        if verbose:
            print(f"加载 {len(etf_data)} 只ETF，{len(dates)} 个交易日")

        engine.run()
        daily_df = engine.get_daily_df()
        trade_df = engine.get_trade_df()

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
            "start": start_date, "end": end_date if end_date else daily_df["date"].iloc[-1],
            "metrics": metrics, "daily_df": daily_df, "trade_df": trade_df,
            "bench_data": bench_data, "ew_data": ew_data,
        }
    finally:
        mr_cfg.RISK_MODE = orig_mr_risk
        mr_cfg.MOMENTUM_WINDOW = orig_mr_mom
        mr_engine.RISK_MODE = orig_mr_risk
        mr_engine.MOMENTUM_WINDOW = orig_mr_mom


def grid_search():
    """网格搜索最优宽度择时参数。"""
    print("\n" + "=" * 70)
    print("  市场宽度择时策略 — 网格搜索")
    print("=" * 70)

    ma_periods = [20, 30, 60]
    strongs = [0.60, 0.70, 0.80]
    weaks = [0.20, 0.30, 0.40]
    neutrals = ["cash", "half"]

    total = len(ma_periods) * len(strongs) * len(weaks) * len(neutrals)
    print(f"  总组合数: {total}\n")

    results = []
    count = 0
    for ma, st, wk, nu in product(ma_periods, strongs, weaks, neutrals):
        if wk >= st:
            continue  # 弱市阈值必须小于强市阈值
        count += 1
        try:
            r = run_backtest(
                start_date="2024-01-01", end_date="",
                initial_capital=cfg.INITIAL_CAPITAL,
                momentum_window=20,
                ma_period=ma, breadth_strong=st,
                breadth_weak=wk, neutral_mode=nu,
                risk_mode="A",
                verbose=False,
            )
            m = r["metrics"]
            results.append({
                "ma": ma, "strong": st, "weak": wk, "neutral": nu,
                "return": m.total_return, "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown, "trades": m.total_trades,
                "win_rate": m.win_rate,
            })
            print(f"  [{count:3d}] MA={ma} st={st:.0%} wk={wk:.0%} {nu:<4} | "
                  f"ret={m.total_return:+.1%} sh={m.sharpe_ratio:.2f} "
                  f"dd={m.max_drawdown:.1%} tr={m.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:3d}] ✗ {e}", flush=True)

    results.sort(key=lambda x: x["return"], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP 15 参数组合")
    print("=" * 70)
    print(f"  {'MA':>3} {'强':>5} {'弱':>5} {'中性':>5} "
          f"{'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    print(f"  {'-'*52}")
    for r in results[:15]:
        print(f"  {r['ma']:>3} {r['strong']:>4.0%} {r['weak']:>4.0%} {r['neutral']:>5} "
              f"{r['return']:>+7.1%} {r['sharpe']:>5.2f} {r['max_dd']:>6.1%} "
              f"{r['trades']:>4} {r['win_rate']:>5.1%}")

    return results


def main():
    args = parse_args()

    if args.scan:
        grid_search()
        return

    print(f"\n{'='*55}")
    print(f"  市场宽度择时策略回测")
    print(f"  {'='*55}")
    print(f"  宽度MA: {args.ma}日  强市: {args.strong:.0%}  弱市: {args.weak:.0%}")
    print(f"  中性模式: {args.neutral}  动量窗口: {args.momentum}日")
    print(f"  {'='*55}")

    result = run_backtest(
        start_date=args.start, end_date=args.end,
        initial_capital=args.money, momentum_window=args.momentum,
        ma_period=args.ma, breadth_strong=args.strong,
        breadth_weak=args.weak, neutral_mode=args.neutral,
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
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
