#!/usr/bin/env python3
"""
自适应轮动策略回测 — 入口脚本

用法:
  python -m strategies.adaptive_rotation.run
  python -m strategies.adaptive_rotation.run --start 2024-01-01 --end 2026-06-30
"""

import argparse, os
from datetime import datetime

from .engine import BacktestEngine
from .metrics import MetricsCalculator
from .reporter import Reporter
from .config import INITIAL_CAPITAL, START_DATE, COMMISSION_RATE, SLIPPAGE, OUTPUT_DIR, RISK_MODE


def parse_args():
    p = argparse.ArgumentParser(description="自适应轮动策略回测")
    p.add_argument("--start", type=str, default=START_DATE)
    p.add_argument("--end", type=str, default="")
    p.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--risk-mode", type=str, default="", choices=["", "A", "B", "C"])
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    risk_mode = args.risk_mode or RISK_MODE

    print(f"\n{'=' * 55}")
    print(f"  自适应轮动策略回测")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  核心逻辑: 牛市→动量  |  震荡→均值回归  |  熊市→空仓")
    print(f"  {'=' * 55}")

    engine = BacktestEngine(args.money, risk_mode)
    print("  [1/4] 加载数据…")
    engine.load_data(args.start, args.end)
    print("  [2/4] 运行回测…")
    engine.run()
    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()
    idx = engine.index_data
    bench = idx["cumulative_returns"].iloc[-1] - 1 if idx is not None and not idx.empty else None
    calc = MetricsCalculator(0.03)
    m = calc.compute(engine.daily_records, engine.trade_records, args.money, benchmark_return=bench)
    print("  [4/4] 生成报告…")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    od = os.path.join(OUTPUT_DIR, f"{ts}{tag}")
    r = Reporter(od)
    r.save_daily_records(daily_df)
    r.save_trade_records(trade_df)
    r.save_metrics(m)
    r.plot_equity_curve(daily_df, engine.index_data, engine.equal_weight_data)
    r.plot_drawdown(daily_df)
    r.print_summary(m)
    print(f"  输出目录: {os.path.abspath(od)}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
