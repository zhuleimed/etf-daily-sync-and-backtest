#!/usr/bin/env python3
"""多周期动量共振策略回测 — 入口脚本"""

import argparse, os, sys
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .engine import MultiPeriodEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark
from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="多周期动量共振策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A","B","C"])
    p.add_argument("--scan", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def run_backtest(start_date, end_date, initial_capital,
                 mom_short, mom_medium, mom_long,
                 min_resonance, risk_mode, verbose=True):
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine
    orig_risk = mr_cfg.RISK_MODE; orig_mom = mr_cfg.MOMENTUM_WINDOW
    mr_cfg.RISK_MODE = risk_mode; mr_cfg.MOMENTUM_WINDOW = mom_long
    mr_engine.RISK_MODE = risk_mode; mr_engine.MOMENTUM_WINDOW = mom_long

    try:
        engine = MultiPeriodEngine(
            initial_capital=initial_capital, risk_mode=risk_mode,
            momentum_short=mom_short, momentum_medium=mom_medium,
            momentum_long=mom_long, min_resonance=min_resonance,
        )
        from strategies.momentum_rotation.data import load_all_etf_data as _load
        etf_data, dates = _load(
            symbols=cfg.ETF_SYMBOLS, start_date=start_date,
            end_date=end_date, db_path=cfg.DB_PATH,
            momentum_window=mom_long,
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
    print("  多周期动量共振策略 — 网格搜索")
    print("=" * 70)

    # 有效周期组合（短 < 中 < 长）
    periods = [(5,10,20),(5,10,30),(5,15,20),(5,15,30),(10,15,20),(10,15,30)]
    resonances = [2, 3]
    risks = ["A", "B"]

    total = len(periods) * len(resonances) * len(risks)
    print(f"  总组合数: {total}\n")

    results = []; count = 0
    for (s, m, l), res, rm in product(periods, resonances, risks):
        count += 1
        try:
            r = run_backtest(start_date="2024-01-01", end_date="",
                             initial_capital=cfg.INITIAL_CAPITAL,
                             mom_short=s, mom_medium=m, mom_long=l,
                             min_resonance=res, risk_mode=rm, verbose=False)
            mt = r["metrics"]
            results.append({"short": s, "medium": m, "long": l, "res": res,
                            "risk": rm, "return": mt.total_return,
                            "sharpe": mt.sharpe_ratio, "max_dd": mt.max_drawdown,
                            "trades": mt.total_trades, "win_rate": mt.win_rate})
            print(f"  [{count:2d}] s={s} m={m} l={l} res={res} risk={rm} | "
                  f"ret={mt.total_return:+.1%} sh={mt.sharpe_ratio:.2f} "
                  f"dd={mt.max_drawdown:.1%} tr={mt.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:2d}] ✗ {e}", flush=True)

    results.sort(key=lambda x: x["return"], reverse=True)
    print("\n" + "=" * 70)
    print("  TOP 15 参数组合")
    print("=" * 70)
    print(f"  {'短':>3} {'中':>3} {'长':>3} {'共振':>4} {'风控':>4} "
          f"{'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    print(f"  {'-'*55}")
    for r in results[:15]:
        print(f"  {r['short']:>3} {r['medium']:>3} {r['long']:>3} "
              f"{r['res']:>4} {r['risk']:>4} "
              f"{r['return']:>+7.1%} {r['sharpe']:>5.2f} "
              f"{r['max_dd']:>6.1%} {r['trades']:>4} {r['win_rate']:>5.1%}")
    return results


def main():
    args = parse_args()
    if args.scan:
        grid_search()
        return
    print(f"\n{'='*55}")
    print(f"  多周期动量共振策略回测")
    print(f"  周期: {cfg.MOMENTUM_SHORT}/{cfg.MOMENTUM_MEDIUM}/{cfg.MOMENTUM_LONG}日")
    print(f"  最小共振: {cfg.MIN_RESONANCE}/3  风控: {args.risk_mode}")
    print(f"  {'='*55}")
    result = run_backtest(
        start_date=args.start, end_date=args.end,
        initial_capital=args.money,
        mom_short=cfg.MOMENTUM_SHORT, mom_medium=cfg.MOMENTUM_MEDIUM,
        mom_long=cfg.MOMENTUM_LONG, min_resonance=cfg.MIN_RESONANCE,
        risk_mode=args.risk_mode,
    )
    metrics = result["metrics"]; daily_df = result["daily_df"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(cfg.OUTPUT_DIR, f"{ts}{tag}")
    reporter = Reporter(output_dir=out)
    reporter.save_daily_records(daily_df)
    reporter.save_trade_records(result["trade_df"])
    reporter.save_metrics(metrics)
    reporter.plot_equity_curve(daily_df, result["bench_data"], result["ew_data"])
    reporter.plot_drawdown(daily_df)
    reporter.print_summary(metrics)
    print(f"  输出目录: {os.path.abspath(out)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
