#!/usr/bin/env python3
"""HS300均线择时策略 — 入口脚本"""
import argparse, os, sys
from datetime import datetime
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from .engine import HS300MATimingEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_all_etf_data, load_benchmark_data, compute_equal_weight_benchmark
from . import config as cfg

def parse_args():
    p = argparse.ArgumentParser(description="HS300均线择时策略回测")
    p.add_argument("--start", default=cfg.START_DATE); p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--ma", type=int, default=cfg.MA_PERIOD)
    p.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A","B","C"])
    p.add_argument("--scan", action="store_true"); p.add_argument("--tag", default="")
    return p.parse_args()

def run_backtest(start_date, end_date, initial_capital,
                 ma_period, momentum_window, risk_mode, verbose=True):
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine
    o_r, o_m = mr_cfg.RISK_MODE, mr_cfg.MOMENTUM_WINDOW
    mr_cfg.RISK_MODE = risk_mode; mr_cfg.MOMENTUM_WINDOW = momentum_window
    mr_engine.RISK_MODE = risk_mode; mr_engine.MOMENTUM_WINDOW = momentum_window
    try:
        engine = HS300MATimingEngine(initial_capital=initial_capital,
                                     risk_mode=risk_mode, momentum_window=momentum_window,
                                     ma_period=ma_period)
        etf_data, dates = load_all_etf_data(
            symbols=cfg.ETF_SYMBOLS, start_date=start_date, end_date=end_date,
            db_path=cfg.DB_PATH, momentum_window=momentum_window,
        )
        engine.etf_data = etf_data; engine.dates = dates
        engine.hs300_data = load_benchmark_data(start_date=start_date, end_date=end_date,
                                                momentum_window=momentum_window)
        if verbose: print(f"加载 {len(etf_data)} 只ETF，{len(dates)} 个交易日")
        engine.run()
        daily_df = engine.get_daily_df(); trade_df = engine.get_trade_df()
        try:
            bd = load_benchmark_data(start_date=start_date, end_date=end_date)
            br = bd["cumulative_returns"].iloc[-1] - 1 if len(bd) > 0 else None
        except Exception: bd, br = None, None
        try:
            ew = compute_equal_weight_benchmark(etf_data)
            ewr = ew["cumulative_returns"].iloc[-1] - 1 if len(ew) > 0 else None
        except Exception: ew, ewr = None, None
        calc = MetricsCalculator(risk_free_rate=0.03)
        metrics = calc.compute(engine.daily_records, engine.trade_records,
                               initial_capital=initial_capital,
                               benchmark_return=br, ew_benchmark_return=ewr)
        return {"start": start_date, "end": end_date if end_date else daily_df["date"].iloc[-1],
                "metrics": metrics, "daily_df": daily_df, "trade_df": trade_df,
                "bench_data": bd, "ew_data": ew}
    finally:
        mr_cfg.RISK_MODE = o_r; mr_cfg.MOMENTUM_WINDOW = o_m
        mr_engine.RISK_MODE = o_r; mr_engine.MOMENTUM_WINDOW = o_m

def grid_search():
    print("\n" + "=" * 70)
    print("  HS300均线择时 — MA周期扫描")
    print("=" * 70)
    for ma in [5, 10, 15, 20, 30, 60, 90, 120]:
        r = run_backtest(start_date="2024-01-01", end_date="", initial_capital=cfg.INITIAL_CAPITAL,
                         ma_period=ma, momentum_window=20, risk_mode="A", verbose=False)
        m = r["metrics"]
        print(f"  MA={ma:>3}: ret={m.total_return:+.1%} sh={m.sharpe_ratio:.2f} "
              f"dd={m.max_drawdown:.1%} tr={m.total_trades}")

def main():
    args = parse_args()
    if args.scan: grid_search(); return
    print(f"\n{'='*55}")
    print(f"  HS300均线择时策略回测")
    print(f"  MA周期: {args.ma}日  动量: {args.momentum}日  风控: {args.risk_mode}")
    print(f"  {'='*55}")
    r = run_backtest(start_date=args.start, end_date=args.end,
        initial_capital=args.money, ma_period=args.ma,
        momentum_window=args.momentum, risk_mode=args.risk_mode)
    m = r["metrics"]; d = r["daily_df"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(cfg.OUTPUT_DIR, f"{ts}{args.tag or ''}")
    rp = Reporter(output_dir=out)
    rp.save_daily_records(d); rp.save_trade_records(r["trade_df"])
    rp.save_metrics(m); rp.plot_equity_curve(d, r["bench_data"], r["ew_data"])
    rp.plot_drawdown(d); rp.print_summary(m)
    print(f"  输出: {os.path.abspath(out)}\n{'='*55}\n")

if __name__ == "__main__": main()
