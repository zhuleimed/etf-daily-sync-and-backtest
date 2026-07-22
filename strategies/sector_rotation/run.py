#!/usr/bin/env python3
"""
行业ETF轮动策略回测 — 入口脚本

复用 momentum_rotation 引擎，仅替换 ETF 池为 12 只行业 ETF。
"""
import argparse
import os
import sys
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="行业ETF轮动策略回测")
    parser.add_argument("--start", type=str, default=cfg.START_DATE)
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    parser.add_argument("--momentum", type=int, default=cfg.MOMENTUM_WINDOW)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--risk-mode", type=str, default=cfg.RISK_MODE, choices=["A","B","C"])
    return parser.parse_args()


def run_backtest(start_date, end_date, initial_capital, momentum_window,
                 min_hold_days=None, min_conviction=None, risk_mode="A", verbose=True):
    """运行单次回测（可覆盖参数用于网格搜索）。"""
    # 临时覆盖 config
    orig_hold = cfg.MIN_HOLD_DAYS
    orig_conv = cfg.MIN_SWITCH_CONVICTION
    orig_risk = cfg.RISK_MODE
    orig_mom = cfg.MOMENTUM_WINDOW
    orig_start = cfg.START_DATE
    orig_end = cfg.END_DATE

    if min_hold_days is not None:
        cfg.MIN_HOLD_DAYS = min_hold_days
    if min_conviction is not None:
        cfg.MIN_SWITCH_CONVICTION = min_conviction
    cfg.RISK_MODE = risk_mode
    cfg.MOMENTUM_WINDOW = momentum_window
    cfg.START_DATE = start_date
    cfg.END_DATE = end_date

    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            risk_mode=risk_mode,
            momentum_window=momentum_window,
            top_n=cfg.TOP_N,
            dynamic_window=cfg.DYNAMIC_WINDOW_ENABLED,
        )

        # 加载数据（传入行业ETF列表）
        from strategies.momentum_rotation.data import load_all_etf_data as _load
        # 临时覆盖 momentum_rotation 的 ETF_SYMBOLS（引擎内部使用）
        import strategies.momentum_rotation.config as mr_cfg
        import strategies.momentum_rotation.engine as mr_engine
        orig_mr_symbols = mr_cfg.ETF_SYMBOLS
        orig_mr_pool = mr_cfg.ETF_POOL
        orig_mr_hold = mr_cfg.MIN_HOLD_DAYS
        orig_mr_conv = mr_cfg.MIN_SWITCH_CONVICTION
        orig_mr_risk = mr_cfg.RISK_MODE
        orig_mr_mom = mr_cfg.MOMENTUM_WINDOW
        # 同步更新 config 和 engine 模块中的变量（import binding 陷阱！）
        mr_cfg.ETF_SYMBOLS = cfg.ETF_SYMBOLS
        mr_cfg.ETF_POOL = cfg.ETF_POOL
        mr_engine.ETF_SYMBOLS = cfg.ETF_SYMBOLS
        mr_engine.ETF_POOL = cfg.ETF_POOL
        mr_cfg.MIN_HOLD_DAYS = cfg.MIN_HOLD_DAYS
        mr_cfg.MIN_SWITCH_CONVICTION = cfg.MIN_SWITCH_CONVICTION
        mr_cfg.RISK_MODE = cfg.RISK_MODE
        mr_cfg.MOMENTUM_WINDOW = cfg.MOMENTUM_WINDOW
        # 引擎模块级别的变量也需要更新（from .config import X 的本地副本——import binding 陷阱！）
        for var in ['ETF_SYMBOLS','ETF_POOL','MOMENTUM_WINDOW','MIN_HOLD_DAYS',
                     'MIN_SWITCH_CONVICTION','RISK_MODE','SHORT_TERM_MOMENTUM_CHECK',
                     'ADJUSTMENT_DAYS','TOP_N','USE_RELATIVE_MOMENTUM',
                     'DYNAMIC_WINDOW_ENABLED']:
            if hasattr(mr_engine, var) and hasattr(cfg, var):
                setattr(mr_engine, var, getattr(cfg, var))
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

        # 基准
        try:
            bench_data = load_benchmark_data(start_date=start_date, end_date=end_date)
            bench_return = bench_data["cumulative_returns"].iloc[-1] - 1 if len(bench_data) > 0 else None
        except Exception:
            bench_data = None
            bench_return = None

        try:
            ew_data = compute_equal_weight_benchmark(etf_data, cfg.ETF_SYMBOLS)
            ew_return = ew_data["cumulative_returns"].iloc[-1] - 1 if len(ew_data) > 0 else None
        except Exception:
            ew_data = None
            ew_return = None

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
        cfg.MIN_HOLD_DAYS = orig_hold
        cfg.MIN_SWITCH_CONVICTION = orig_conv
        cfg.RISK_MODE = orig_risk
        cfg.MOMENTUM_WINDOW = orig_mom
        cfg.START_DATE = orig_start
        cfg.END_DATE = orig_end
        mr_cfg.ETF_SYMBOLS = orig_mr_symbols
        mr_cfg.ETF_POOL = orig_mr_pool
        mr_engine.ETF_SYMBOLS = orig_mr_symbols
        mr_engine.ETF_POOL = orig_mr_pool
        mr_cfg.MIN_HOLD_DAYS = orig_mr_hold
        mr_cfg.MIN_SWITCH_CONVICTION = orig_mr_conv
        mr_cfg.RISK_MODE = orig_mr_risk
        mr_cfg.MOMENTUM_WINDOW = orig_mr_mom


def main():
    args = parse_args()
    print(f"\n{'='*55}")
    print(f"  行业ETF轮动策略回测")
    print(f"  {'='*55}")
    print(f"  ETF池: {len(cfg.ETF_SYMBOLS)}只行业ETF")
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
