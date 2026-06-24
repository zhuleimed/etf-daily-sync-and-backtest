"""入口脚本"""
import argparse, os
from datetime import datetime
from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import (
    ETF_SYMBOLS, INITIAL_CAPITAL, START_DATE,
    COMMISSION_RATE, SLIPPAGE, ADJUSTMENT_DAYS,
    OUTPUT_DIR, RISK_MODE, TOP_N,
)
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default=START_DATE)
    p.add_argument("--end", type=str, default="")
    p.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"\n{'='*55}")
    print(f"  {'mean_reversion'}")
    print(f"  {'='*55}")
    engine = BacktestEngine(initial_capital=args.money)
    print("  [1/4] 加载数据...")
    engine.load_data(start_date=args.start, end_date=args.end)
    print("  [2/4] 运行回测...")
    engine.run()
    print("  [3/4] 计算绩效...")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()
    bm = engine.benchmark_data
    ew = engine.equal_weight_data
    bmr = bm["cumulative_returns"].iloc[-1] - 1 if not bm.empty else None
    ewr = ew["cumulative_returns"].iloc[-1] - 1 if ew is not None and not ew.empty else None
    calc = MetricsCalculator(risk_free_rate=0.03)
    metrics = calc.compute(engine.daily_records, engine.trade_records,
        initial_capital=args.money, benchmark_return=bmr, ew_benchmark_return=ewr)
    print("  [4/4] 生成报告...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(OUTPUT_DIR, f"{ts}{tag}")
    rep = Reporter(output_dir=out)
    rep.save_daily_records(daily_df); rep.save_trade_records(trade_df)
    rep.save_metrics(metrics); rep.plot_equity_curve(daily_df, bm, ew)
    rep.plot_drawdown(daily_df); rep.plot_monthly_returns(daily_df)
    rep.print_summary(metrics)
    print(f"  输出目录: {os.path.abspath(out)}\n")

if __name__ == "__main__":
    main()
