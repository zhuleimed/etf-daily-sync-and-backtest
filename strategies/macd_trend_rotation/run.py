#!/usr/bin/env python3
"""
MACD 趋势轮动策略 — 入口脚本

用法:
  python strategies/macd_trend_rotation/run.py
  python strategies/macd_trend_rotation/run.py --risk-mode A --tag test
"""

import argparse
import os
from datetime import datetime
from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import INITIAL_CAPITAL, START_DATE, COMMISSION_RATE, SLIPPAGE, OUTPUT_DIR, RISK_MODE


def parse_args():
    p = argparse.ArgumentParser(description="MACD趋势轮动策略回测")
    p.add_argument("--start", type=str, default=START_DATE)
    p.add_argument("--end", type=str, default="")
    p.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--risk-mode", type=str, default="", choices=["", "A", "B", "C"])
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args(); rm = args.risk_mode if args.risk_mode else RISK_MODE
    print(f"\n{'=' * 55}")
    print(f"  MACD趋势轮动策略回测"); print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    mn = {"A": "纯信号（无风控）", "B": "全开", "C": "仅极端回撤"}
    print(f"  风控模式: {rm} = {mn.get(rm, rm)}"); print(f"  {'=' * 55}")

    engine = BacktestEngine(initial_capital=args.money, risk_mode=rm)
    print("  [1/4] 加载数据…"); engine.load_data(start_date=args.start, end_date=args.end)
    print("  [2/4] 运行回测…"); engine.run()
    print("  [3/4] 计算绩效…")
    dd = engine.get_daily_df(); td = engine.get_trade_df()
    idx = engine.index_data; ew = engine.equal_weight_data
    btr = idx["cumulative_returns"].iloc[-1] - 1 if idx is not None and not idx.empty else None
    ewr = ew["cumulative_returns"].iloc[-1] - 1 if ew is not None and not ew.empty else None
    calc = MetricsCalculator()
    m = calc.compute(engine.daily_records, engine.trade_records, initial_capital=args.money, benchmark_return=btr, ew_benchmark_return=ewr)

    print("  [4/4] 生成报告…")
    od = os.path.join(OUTPUT_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}" if args.tag else f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    r = Reporter(od)
    r.save_daily(dd); r.save_trades(td); r.save_metrics(m)
    r.plot_equity(dd, idx, ew); r.plot_drawdown(dd); r.plot_heatmap(dd); r.plot_monthly(dd)
    r.print_summary(m)
    print(f"  输出目录: {os.path.abspath(od)}"); print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
