#!/usr/bin/env python3
"""
量价配合增强策略回测 — 入口脚本

用法：
  python -m strategies.volume_price.run           # 默认参数
  python -m strategies.volume_price.run --scan    # 网格搜索
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

from .engine import VolumePriceEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark

from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="量价配合增强策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    p.add_argument("--vol-short", type=int, default=cfg.VOL_SHORT_PERIOD)
    p.add_argument("--vol-long", type=int, default=cfg.VOL_LONG_PERIOD)
    p.add_argument("--vol-threshold", type=float, default=cfg.VOL_THRESHOLD)
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A", "B", "C"])
    p.add_argument("--scan", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def run_backtest(start_date, end_date, initial_capital,
                 momentum_window, vol_short, vol_long,
                 vol_threshold, risk_mode, verbose=True):
    """运行单次回测。"""
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine

    orig_mr_risk = mr_cfg.RISK_MODE
    orig_mr_mom = mr_cfg.MOMENTUM_WINDOW
    mr_cfg.RISK_MODE = risk_mode
    mr_cfg.MOMENTUM_WINDOW = momentum_window
    mr_engine.RISK_MODE = risk_mode
    mr_engine.MOMENTUM_WINDOW = momentum_window

    try:
        engine = VolumePriceEngine(
            initial_capital=initial_capital,
            risk_mode=risk_mode,
            momentum_window=momentum_window,
            vol_short=vol_short,
            vol_long=vol_long,
            vol_threshold=vol_threshold,
        )
        from strategies.momentum_rotation.data import load_all_etf_data as _load
        etf_data, dates = _load(
            symbols=cfg.ETF_SYMBOLS, start_date=start_date,
            end_date=end_date, db_path=cfg.DB_PATH,
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
    """网格搜索最优量价参数。"""
    print("\n" + "=" * 70)
    print("  量价配合增强策略 — 网格搜索")
    print("=" * 70)

    windows = [15, 20]
    vol_shorts = [5, 10]
    vol_longs = [20, 30]
    thresholds = [1.0, 1.2, 1.5]
    risks = ["A", "B"]

    total = len(windows) * len(vol_shorts) * len(vol_longs) * len(thresholds) * len(risks)
    print(f"  总组合数: {total}\n")

    results = []
    count = 0
    for w, vs, vl, th, rm in product(windows, vol_shorts, vol_longs, thresholds, risks):
        if vs >= vl:
            continue  # 短周期必须 < 长周期
        count += 1
        try:
            r = run_backtest(
                start_date="2024-01-01", end_date="",
                initial_capital=cfg.INITIAL_CAPITAL,
                momentum_window=w,
                vol_short=vs, vol_long=vl,
                vol_threshold=th, risk_mode=rm,
                verbose=False,
            )
            m = r["metrics"]
            results.append({
                "window": w, "vol_s": vs, "vol_l": vl, "thresh": th, "risk": rm,
                "return": m.total_return, "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown, "trades": m.total_trades,
                "win_rate": m.win_rate,
            })
            print(f"  [{count:3d}] w={w} vs={vs} vl={vl} th={th:.0%} "
                  f"risk={rm} | ret={m.total_return:+.1%} "
                  f"sh={m.sharpe_ratio:.2f} dd={m.max_drawdown:.1%} "
                  f"tr={m.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:3d}] ✗ {e}", flush=True)

    results.sort(key=lambda x: x["return"], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP 15 参数组合")
    print("=" * 70)
    print(f"  {'窗':>3} {'VS':>3} {'VL':>3} {'阈值':>5} {'风控':>4} "
          f"{'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    print(f"  {'-'*55}")
    for r in results[:15]:
        print(f"  {r['window']:>3} {r['vol_s']:>3} {r['vol_l']:>3} "
              f"{r['thresh']:>4.0%} {r['risk']:>4} "
              f"{r['return']:>+7.1%} {r['sharpe']:>5.2f} "
              f"{r['max_dd']:>6.1%} {r['trades']:>4} {r['win_rate']:>5.1%}")

    return results


def main():
    args = parse_args()
    if args.scan:
        grid_search()
        return

    print(f"\n{'='*55}")
    print(f"  量价配合增强策略回测")
    print(f"  {'='*55}")
    print(f"  量比: {args.vol_short}日/{args.vol_long}日  阈值: {args.vol_threshold:.0%}")
    print(f"  动量窗口: {args.momentum}日  风控: {args.risk_mode}")
    print(f"  {'='*55}")

    result = run_backtest(
        start_date=args.start, end_date=args.end,
        initial_capital=args.money,
        momentum_window=args.momentum,
        vol_short=args.vol_short, vol_long=args.vol_long,
        vol_threshold=args.vol_threshold, risk_mode=args.risk_mode,
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
