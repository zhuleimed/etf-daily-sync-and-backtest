#!/usr/bin/env python3
"""
黄金避险轮动策略 — 参数网格搜索

搜索硬阈值的三个参数 + 牛市过滤器MA周期，找出最优组合。
"""
import itertools
import time
import pandas as pd
from datetime import datetime

from strategies.gold_safe_haven import config as cfg
from strategies.gold_safe_haven.run import run_backtest


# ============================================================================
# 网格定义
# ============================================================================

DD_THRESHOLDS = [-0.02, -0.03, -0.04, -0.05, -0.06]
VOL_THRESHOLDS = [1.1, 1.2, 1.3, 1.4, 1.5]
BREADTH_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
BULL_FILTER_MAS = [0, 60, 120, 250]  # 0 = 不使用牛市过滤器

# ============================================================================
# 搜索
# ============================================================================

def run_grid_search():
    combinations = list(itertools.product(
        DD_THRESHOLDS, VOL_THRESHOLDS, BREADTH_THRESHOLDS, BULL_FILTER_MAS
    ))
    total = len(combinations)
    print(f"网格搜索: {total} 组合 (dd×{len(DD_THRESHOLDS)} vol×{len(VOL_THRESHOLDS)} "
          f"brd×{len(BREADTH_THRESHOLDS)} ma×{len(BULL_FILTER_MAS)})")
    print(f"预计耗时: ~{total * 3 // 60} 分钟")
    print(f"开始: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 80)

    results = []
    t0 = time.time()

    for i, (dd, vol, brd, ma) in enumerate(combinations):
        # 临时覆盖参数
        cfg.PANIC_DD_THRESHOLD = dd
        cfg.PANIC_VOL_THRESHOLD = vol
        cfg.PANIC_BREADTH_THRESHOLD = brd
        cfg.PANIC_MODE = "hard"
        if ma > 0:
            cfg.USE_BULL_FILTER = True
            cfg.BULL_FILTER_MA = ma
        else:
            cfg.USE_BULL_FILTER = False

        try:
            r = run_backtest("2024-01-01", "", 10000, 20, "", 1.5, verbose=False)
            m = r["metrics"]

            results.append({
                "dd_thr": dd,
                "vol_thr": vol,
                "brd_thr": brd,
                "bull_ma": ma,
                "total_return": m.total_return,
                "sharpe": m.sharpe_ratio,
                "max_dd": m.max_drawdown,
                "trades": m.total_trades,
                "switches": m.switch_count,
                "panic_entries": r["panic_entries"],
                "panic_days": r["panic_days"],
                "panic_pct": r["panic_pct"],
                "annual_return": m.annualized_return,
                "calmar": m.calmar_ratio,
            })
        except Exception as e:
            print(f"  ⚠ [{i+1}/{total}] dd={dd} vol={vol} brd={brd} ma={ma} 失败: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {datetime.now().strftime('%H:%M:%S')} 耗时 {elapsed:.0f}s")

    # 恢复默认
    cfg.PANIC_DD_THRESHOLD = -0.04
    cfg.PANIC_VOL_THRESHOLD = 1.2
    cfg.PANIC_BREADTH_THRESHOLD = 0.7
    cfg.USE_BULL_FILTER = True
    cfg.BULL_FILTER_MA = 60

    return pd.DataFrame(results)


def print_top_results(df: pd.DataFrame, n: int = 20):
    """打印 top-N 结果。"""
    print(f"\n{'=' * 90}")
    print(f"  TOP {n} 参数组合（按收益率排序）")
    print(f"{'=' * 90}")
    print(f"{'排名':<4s} {'dd':>5s} {'vol':>5s} {'brd':>5s} {'MA':>4s} "
          f"{'收益':>8s} {'夏普':>7s} {'回撤':>7s} {'交易':>5s} {'恐慌':>5s} {'金天':>5s}")
    print("-" * 90)

    top = df.nlargest(n, "total_return")
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        ma_str = str(int(row["bull_ma"])) if row["bull_ma"] > 0 else "OFF"
        print(f"{rank:<4d} {row['dd_thr']:>5.2f} {row['vol_thr']:>5.1f} "
              f"{row['brd_thr']:>5.1f} {ma_str:>4s} "
              f"{row['total_return']:>7.2%} {row['sharpe']:>7.3f} "
              f"{row['max_dd']:>7.2%} {row['trades']:>5d} "
              f"{row['panic_entries']:>5d} {row['panic_days']:>5d}")

    # 按夏普排序
    print(f"\n{'=' * 90}")
    print(f"  TOP {n} 参数组合（按夏普比率排序）")
    print(f"{'=' * 90}")
    print(f"{'排名':<4s} {'dd':>5s} {'vol':>5s} {'brd':>5s} {'MA':>4s} "
          f"{'收益':>8s} {'夏普':>7s} {'回撤':>7s} {'交易':>5s} {'恐慌':>5s} {'金天':>5s}")
    print("-" * 90)

    top_sharpe = df.nlargest(n, "sharpe")
    for rank, (_, row) in enumerate(top_sharpe.iterrows(), 1):
        ma_str = str(int(row["bull_ma"])) if row["bull_ma"] > 0 else "OFF"
        print(f"{rank:<4d} {row['dd_thr']:>5.2f} {row['vol_thr']:>5.1f} "
              f"{row['brd_thr']:>5.1f} {ma_str:>4s} "
              f"{row['total_return']:>7.2%} {row['sharpe']:>7.3f} "
              f"{row['max_dd']:>7.2%} {row['trades']:>5d} "
              f"{row['panic_entries']:>5d} {row['panic_days']:>5d}")

    # 基准对比
    print(f"\n{'=' * 90}")
    print(f"  基准对比")
    print(f"{'=' * 90}")
    # 纯动量基准（无恐慌，直接用动量轮动）
    print(f"  {'纯动量轮动(无恐慌)':<20s} 收益≈+132.9%  夏普≈1.19  回撤≈-25.8%  交易≈38")
    print(f"  {'Z-score版(原始)':<20s} 收益≈+66.2%   夏普≈0.71  回撤≈-22.0%  交易≈158")
    print(f"  {'硬阈值版(默认参数)':<20s} 收益≈+90.7%   夏普≈0.88  回撤≈-21.1%  交易≈150")

    # 最优组合总结
    best = top.iloc[0]
    print(f"\n  ★ 最优参数（按收益率）: dd={best['dd_thr']:.2f} vol={best['vol_thr']:.1f} "
          f"brd={best['brd_thr']:.1f} MA={best['bull_ma']:.0f}")
    print(f"    收益={best['total_return']:.2%} 夏普={best['sharpe']:.3f} "
          f"回撤={best['max_dd']:.2%} 交易={best['trades']} "
          f"恐慌{best['panic_entries']}次/{best['panic_days']}天")

    return top, top_sharpe


if __name__ == "__main__":
    df = run_grid_search()
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"strategies/gold_safe_haven/output/grid_search_{ts}.csv"
    import os
    os.makedirs("strategies/gold_safe_haven/output", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")

    top_return, top_sharpe = print_top_results(df, 15)
