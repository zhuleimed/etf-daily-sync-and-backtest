#!/usr/bin/env python3
"""
跨境轮动策略回测 — 入口脚本

A股+美股+港股三市场动量轮动，复用 momentum_rotation 引擎。

用法：
  # 默认参数回测
  python -m strategies.cross_border.run

  # 网格搜索
  python -m strategies.cross_border.run --scan

  # 单参数覆盖
  python -m strategies.cross_border.run --momentum 15 --risk-mode B
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.metrics import MetricsCalculator
from strategies.momentum_rotation.reporter import Reporter
from strategies.momentum_rotation.data import load_benchmark_data, compute_equal_weight_benchmark

from . import config as cfg


def parse_args():
    p = argparse.ArgumentParser(description="跨境轮动策略回测")
    p.add_argument("--start", default=cfg.START_DATE)
    p.add_argument("--end", default="")
    p.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    p.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    p.add_argument("--tag", default="")
    p.add_argument("--risk-mode", default=cfg.RISK_MODE, choices=["A", "B", "C"])
    p.add_argument("--scan", action="store_true", help="网格搜索最优参数")
    return p.parse_args()


# ============================================================================
# 单次回测
# ============================================================================

def run_backtest(start_date, end_date, initial_capital, momentum_window,
                 min_hold_days=None, min_conviction=None, risk_mode="A",
                 verbose=True):
    """运行单次回测。复用动量引擎，仅替换 ETF 池。"""
    # 保存原始值
    orig_hold = cfg.MIN_HOLD_DAYS
    orig_conv = cfg.MIN_SWITCH_CONVICTION
    orig_risk = cfg.RISK_MODE
    orig_mom = cfg.MOMENTUM_WINDOW

    if min_hold_days is not None:
        cfg.MIN_HOLD_DAYS = min_hold_days
    if min_conviction is not None:
        cfg.MIN_SWITCH_CONVICTION = min_conviction
    cfg.RISK_MODE = risk_mode
    cfg.MOMENTUM_WINDOW = momentum_window

    # 覆盖 momentum_rotation 模块中的 ETF 池变量（import binding 陷阱！）
    import strategies.momentum_rotation.config as mr_cfg
    import strategies.momentum_rotation.engine as mr_engine

    orig_mr_symbols = mr_cfg.ETF_SYMBOLS
    orig_mr_pool = mr_cfg.ETF_POOL
    orig_mr_hold = mr_cfg.MIN_HOLD_DAYS
    orig_mr_conv = mr_cfg.MIN_SWITCH_CONVICTION
    orig_mr_risk = mr_cfg.RISK_MODE
    orig_mr_mom = mr_cfg.MOMENTUM_WINDOW

    # 同步 config 模块
    mr_cfg.ETF_SYMBOLS = cfg.ETF_SYMBOLS
    mr_cfg.ETF_POOL = cfg.ETF_POOL
    mr_cfg.MIN_HOLD_DAYS = cfg.MIN_HOLD_DAYS
    mr_cfg.MIN_SWITCH_CONVICTION = cfg.MIN_SWITCH_CONVICTION
    mr_cfg.RISK_MODE = cfg.RISK_MODE
    mr_cfg.MOMENTUM_WINDOW = cfg.MOMENTUM_WINDOW

    # 同步 engine 模块（from .config import X 的本地副本）
    mr_engine.ETF_SYMBOLS = cfg.ETF_SYMBOLS
    mr_engine.ETF_POOL = cfg.ETF_POOL
    for var in ['ETF_SYMBOLS', 'ETF_POOL', 'MOMENTUM_WINDOW', 'MIN_HOLD_DAYS',
                 'MIN_SWITCH_CONVICTION', 'RISK_MODE', 'SHORT_TERM_MOMENTUM_CHECK',
                 'ADJUSTMENT_DAYS', 'TOP_N', 'USE_RELATIVE_MOMENTUM',
                 'DYNAMIC_WINDOW_ENABLED']:
        if hasattr(mr_engine, var) and hasattr(cfg, var):
            setattr(mr_engine, var, getattr(cfg, var))

    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            risk_mode=risk_mode,
            momentum_window=momentum_window,
            top_n=cfg.TOP_N,
            dynamic_window=cfg.DYNAMIC_WINDOW_ENABLED,
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

        # 基准（沪深300作为A股基准参考）
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
            "start": start_date,
            "end": end_date if end_date else daily_df["date"].iloc[-1],
            "metrics": metrics, "daily_df": daily_df, "trade_df": trade_df,
            "bench_data": bench_data, "ew_data": ew_data,
        }
    finally:
        # 恢复原始值
        cfg.MIN_HOLD_DAYS = orig_hold
        cfg.MIN_SWITCH_CONVICTION = orig_conv
        cfg.RISK_MODE = orig_risk
        cfg.MOMENTUM_WINDOW = orig_mom
        mr_cfg.ETF_SYMBOLS = orig_mr_symbols
        mr_cfg.ETF_POOL = orig_mr_pool
        mr_cfg.MIN_HOLD_DAYS = orig_mr_hold
        mr_cfg.MIN_SWITCH_CONVICTION = orig_mr_conv
        mr_cfg.RISK_MODE = orig_mr_risk
        mr_cfg.MOMENTUM_WINDOW = orig_mr_mom
        mr_engine.ETF_SYMBOLS = orig_mr_symbols
        mr_engine.ETF_POOL = orig_mr_pool


# ============================================================================
# 网格搜索
# ============================================================================

def grid_search():
    """网格搜索最优跨境轮动参数。"""
    print("\n" + "=" * 70)
    print("  跨境轮动策略 — 网格搜索")
    print("=" * 70)

    windows = [10, 15, 20, 25]
    holds = [5, 10, 15]
    convictions = [0.02, 0.03, 0.05]
    risk_modes = ["A", "B"]

    total = len(windows) * len(holds) * len(convictions) * len(risk_modes)
    print(f"  总组合数: {total}\n")

    results = []
    count = 0
    for w, h, conv, rm in product(windows, holds, convictions, risk_modes):
        count += 1
        try:
            r = run_backtest(
                start_date="2024-01-01", end_date="",
                initial_capital=cfg.INITIAL_CAPITAL,
                momentum_window=w,
                min_hold_days=h,
                min_conviction=conv,
                risk_mode=rm,
                verbose=False,
            )
            m = r["metrics"]
            results.append({
                "window": w, "hold": h, "conviction": conv, "risk": rm,
                "return": m.total_return, "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown, "trades": m.total_trades,
                "win_rate": m.win_rate,
            })
            print(f"  [{count:3d}/{total}] w={w} hold={h} conv={conv:.0%} "
                  f"risk={rm} | ret={m.total_return:+.1%} "
                  f"sh={m.sharpe_ratio:.2f} dd={m.max_drawdown:.1%} "
                  f"tr={m.total_trades}", flush=True)
        except Exception as e:
            print(f"  [{count:3d}/{total}] ✗ {e}", flush=True)

    results.sort(key=lambda x: x["return"], reverse=True)

    print("\n" + "=" * 70)
    print("  TOP 20 参数组合（按收益排序）")
    print("=" * 70)
    print(f"  {'排名':<4} {'窗':>3} {'持仓':>4} {'置信':>6} {'风控':>4} "
          f"{'收益':>8} {'夏普':>6} {'回撤':>7} {'交易':>4} {'胜率':>6}")
    print(f"  {'-'*58}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<4} {r['window']:>3} {r['hold']:>4} {r['conviction']:>6.0%} "
              f"{r['risk']:>4} {r['return']:>+7.1%} {r['sharpe']:>5.2f} "
              f"{r['max_dd']:>6.1%} {r['trades']:>4} {r['win_rate']:>5.1%}")

    return results


# ============================================================================
# 主入口
# ============================================================================

def main():
    args = parse_args()

    if args.scan:
        grid_search()
        return

    print(f"\n{'='*55}")
    print(f"  跨境轮动策略回测（A股+美股+港股）")
    print(f"  {'='*55}")
    print(f"  ETF池: {len(cfg.ETF_SYMBOLS)}只（A股×2 + 美股×2 + 港股×1）")
    print(f"  动量窗口: {args.momentum}日  置信度: {cfg.MIN_SWITCH_CONVICTION:.0%}")
    print(f"  最小持仓: {cfg.MIN_HOLD_DAYS}天  风控: {args.risk_mode}")
    print(f"  {'='*55}")

    result = run_backtest(
        start_date=args.start, end_date=args.end,
        initial_capital=args.money, momentum_window=args.momentum,
        risk_mode=args.risk_mode,
    )

    metrics = result["metrics"]
    daily_df = result["daily_df"]
    trade_df = result["trade_df"]

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

    print(f"  输出目录: {os.path.abspath(output_dir)}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
