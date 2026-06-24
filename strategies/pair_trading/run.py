"""配对交易策略 — 入口脚本"""
import argparse, os, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from .engine import PairTradingEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from .config import (
    PAIRS, INITIAL_CAPITAL, CAPITAL_PER_PAIR,
    ZSCORE_OPEN, ZSCORE_CLOSE, ZSCORE_STOP,
    COMMISSION_RATE, SLIPPAGE, OUTPUT_DIR, ZSCORE_PERIOD,
)
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark

def parse_args():
    p = argparse.ArgumentParser(description="ETF 配对交易策略")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="")
    p.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"\n{'=' * 55}")
    print(f"  ETF 配对交易策略")
    print(f"  {'=' * 55}")
    print(f"  回测区间: {args.start} → {args.end or '最新'}")
    print(f"  初始资金: {args.money:,.0f} 元")
    print(f"  配对数量: {len(PAIRS)} 对 | 每对 {CAPITAL_PER_PAIR} 元")
    print(f"  z-score: 开>{ZSCORE_OPEN} 关<{ZSCORE_CLOSE} 停>{ZSCORE_STOP}")
    print(f"  交易费用: 佣金{COMMISSION_RATE:.2%}, 滑点{SLIPPAGE:.2%}")
    print(f"  配对列表:")
    for p in PAIRS:
        print(f"    {p['name']}: {p['a']} ↔ {p['b']}")
    print(f"  {'=' * 55}")

    engine = PairTradingEngine(initial_capital=args.money)
    print("  [1/4] 加载数据…")
    engine.load_data(start_date=args.start, end_date=args.end)
    print("  [2/4] 运行回测…")
    engine.run()

    print("  [3/4] 计算绩效…")
    daily_df = engine.get_daily_df()
    trade_df = engine.get_trade_df()

    # 基准数据（用于对比）
    try:
        bm = load_benchmark_data(start_date=args.start, end_date=args.end)
        bmr = bm["cumulative_returns"].iloc[-1] - 1 if not bm.empty else None
    except:
        bmr = None

    calc = MetricsCalculator(risk_free_rate=0.03)
    # 将 DailyRecord 转成 MetricsCalculator 需要的格式
    metrics = calc.compute(
        engine.daily_records, [],  # trade_records 为空
        initial_capital=args.money,
        benchmark_return=bmr,
    )

    print("  [4/4] 生成报告…")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out = os.path.join(OUTPUT_DIR, f"{ts}{tag}")
    rep = Reporter(output_dir=out)
    rep.save_daily_records(daily_df)
    rep.save_metrics(metrics)
    rep.plot_equity_curve(daily_df, bm if 'bm' in dir() else pd.DataFrame())
    rep.print_summary(metrics)
    print(f"  输出目录: {os.path.abspath(out)}\n")

if __name__ == "__main__":
    main()
