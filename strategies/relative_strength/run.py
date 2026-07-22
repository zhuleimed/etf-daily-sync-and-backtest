#!/usr/bin/env python3
"""相对强度持久性策略回测 — 入口脚本"""
import argparse, os, sys
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .engine import RelativeStrengthEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark
from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="相对强度持久性策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    p.add_argument("--rel-window", type=int, default=cfg.RELATIVE_WINDOW)
    p.add_argument("--min-persistence", type=float, default=cfg.MIN_PERSISTENCE)
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A","B","C"])
    p.add_argument("--scan", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def run_backtest(start_date, end_date, initial_capital,
                 momentum_window, rel_window, min_persistence,
                 risk_mode, verbose=True):
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine
    orig_risk = mr_cfg.RISK_MODE; orig_mom = mr_cfg.MOMENTUM_WINDOW
    mr_cfg.RISK_MODE = risk_mode; mr_cfg.MOMENTUM_WINDOW = momentum_window
    mr_engine.RISK_MODE = risk_mode; mr_engine.MOMENTUM_WINDOW = momentum_window
    try:
        engine = RelativeStrengthEngine(
            initial_capital=initial_capital, risk_mode=risk_mode,
            momentum_window=momentum_window, relative_window=rel_window,
            min_persistence=min_persistence,
        )
        from strategies.momentum_rotation.data import load_all_etf_data as _load
        etf_data, dates = _load(
            symbols=cfg.ETF_SYMBOLS, start_date=start_date,
            end_date=end_date, db_path=cfg.DB_PATH,
            momentum_window=momentum_window,
        )
        engine.etf_data = etf_data; engine.dates = dates
        if verbose:
            print(f"加载 {len(etf_data)} 只ETF，{len(dates)} 个交易日")
        engine.run()
        daily_df = engine.get_daily_df(); trade_df = engine.get_trade_df()
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
        metrics = calc.compute(engine.daily_records, engine.trade_records,
                               initial_capital=initial_capital,
                               benchmark_return=bench_return,
                               ew_benchmark_return=ew_return)
        return {"start": start_date, "end": end_date if end_date else daily_df["date"].iloc[-1],
                "metrics": metrics, "daily_df": daily_df, "trade_df": trade_df,
                "bench_data": bench_data, "ew_data": ew_data}
    finally:
        mr_cfg.RISK_MODE = orig_risk; mr_cfg.MOMENTUM_WINDOW = orig_mom
        mr_engine.RISK_MODE = orig_risk; mr_engine.MOMENTUM_WINDOW = orig_mom


def grid_search():
    print("\n" + "=" * 70)
    print("  相对强度持久性策略 — 网格搜索")
    print("=" * 70)
    rel_windows = [10, 20, 30]
    persistences = [0.50, 0.60, 0.70]
    risks = ["A", "B"]
    total = len(rel_windows) * len(persistences) * len(risks)
    print(f"  总组合数: {total}\n")

    results = []; count = 0
    for rw, mp, rm in product(rel_windows, persistences, risks):
        count += 1
        try:
            r = run_backtest(start_date="2024-01-01", end_date="",
                             initial_capital=cfg.INITIAL_CAPITAL,
                             momentum_window=20, rel_window=rw,
                             min_persistence=mp, risk_mode=rm,
                             verbose=False)
            m = r["metrics"]
            results.append({"rel_win": rw, "min_p": mp, "risk": rm,
                            "return": m.total_return, "sharpe": m.sharpe_ratio,
                            "max_dd": m.max_drawdown, "trades": m.total_trades,
                            "win_rate": m.win_rate})
            print(f"  [{count:2d}] rw={rw} mp={mp:.0%} risk={rm} | "
                  f"ret={m.total_return:+.1%} sh={m.sharpe_ratio:.2f} "
                  f"dd={m.max_drawdown:.1%} tr={m.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:2d}] ✗ {e}", flush=True)

    results.sort(key=lambda x: x["return"], reverse=True)
    print("\n" + "=" * 70)
    print("  TOP 10")
    print("=" * 70)
    print(f"  {'窗':>3} {'持续性':>6} {'风控':>4} "
          f"{'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    for r in results[:10]:
        print(f"  {r['rel_win']:>3} {r['min_p']:>5.0%} {r['risk']:>4} "
              f"{r['return']:>+7.1%} {r['sharpe']:>5.2f} "
              f"{r['max_dd']:>6.1%} {r['trades']:>4} {r['win_rate']:>5.1%}")
    return results


def main():
    args = parse_args()
    if args.scan:
        grid_search()
        return
    print(f"\n{'='*55}")
    print(f"  相对强度持久性策略回测")
    print(f"  相对窗口: {args.rel_window}日  最小持续性: {args.min_persistence:.0%}")
    print(f"  动量窗口: {args.momentum}日  风控: {args.risk_mode}")
    print(f"  {'='*55}")
    result = run_backtest(start_date=args.start, end_date=args.end,
        initial_capital=args.money, momentum_window=args.momentum,
        rel_window=args.rel_window, min_persistence=args.min_persistence,
        risk_mode=args.risk_mode)
    m = result["metrics"]; d = result["daily_df"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg.OUTPUT_DIR, f"{ts}{args.tag or ''}")
    Reporter(output_dir=out).save_daily_records(d)
    Reporter(output_dir=out).save_trade_records(result["trade_df"])
    Reporter(output_dir=out).save_metrics(m)
    Reporter(output_dir=out).print_summary(m)
    print(f"  输出: {os.path.abspath(out)}\n{'='*55}\n")


if __name__ == "__main__":
    main()
