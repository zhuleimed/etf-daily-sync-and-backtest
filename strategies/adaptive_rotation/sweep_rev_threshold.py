"""adaptive_rotation 均值回归入场阈值 — 4 期对比扫描

背景：均值回归分支的硬过滤要求「RSI < REV_OVERSOLD_RSI 且 %B < REV_OVERSOLD_PCT_B」。
      当前 REV_OVERSOLD_PCT_B = 0.0，意思是"价格必须跌破布林带下轨"（约 2σ 事件）。
      实测自 2026-08-03 起 36 个交易日里该组合只满足过 1 次（且当天是熊市被门禁挡掉），
      策略因此连续 41 天零交易。本脚本回答："把阈值放宽会不会更好？"

为什么现在才做得了这个对比：
      `_compute_reversion_scores` 里 %B 分量原写作 (-pct_b)，等于把阈值硬编码成 0，
      改阈值不生效（2026-09-25 已修为 (阈值 - pct_b)）。修完阈值才真正可控。

用法：
    python -m strategies.adaptive_rotation.sweep_rev_threshold
    python -m strategies.adaptive_rotation.sweep_rev_threshold --values 0.0 0.2 0.4
"""

from __future__ import annotations

import argparse
import time

from . import config as cfg
from .engine import BacktestEngine
from .metrics import MetricsCalculator

# 4 种期间（与项目既定的策略评估流程一致）
PERIODS = [
    ("2024全年", "2024-01-01", "2024-12-31"),
    ("2025全年", "2025-01-01", "2025-12-31"),
    ("2026至今", "2026-01-01", ""),
    ("全周期", "2024-01-01", ""),
]


def run_once(threshold: float, start: str, end: str, capital: float,
             rsi_threshold: float | None = None) -> dict:
    """单次回测。阈值通过运行时改写 cfg 生效（signals.py 用 `import config as cfg`）。"""
    cfg.REV_OVERSOLD_PCT_B = threshold
    if rsi_threshold is not None:
        cfg.REV_OVERSOLD_RSI = rsi_threshold

    eng = BacktestEngine(capital, cfg.RISK_MODE)
    eng.load_data(start, end)
    eng.run()

    idx = eng.index_data
    bench = (idx["cumulative_returns"].iloc[-1] - 1) if idx is not None and not idx.empty else None
    m = MetricsCalculator(0.03).compute(eng.daily_records, eng.trade_records,
                                        capital, benchmark_return=bench)
    d = m.to_dict()

    # 区分"均值回归开仓"与其他开仓，看放宽阈值到底有没有带来新交易
    rev_trades = sum(1 for t in eng.trade_records if "均值回归" in str(getattr(t, "reason", "")))
    return {
        "累计收益率": d.get("累计收益率", "--"),
        "年化收益率": d.get("年化收益率", "--"),
        "最大回撤": d.get("最大回撤", "--"),
        "夏普比率": d.get("夏普比率", "--"),
        "切换次数": d.get("调仓切换次数", 0),
        "均值回归开仓": rev_trades,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3],
                    help="要扫描的 REV_OVERSOLD_PCT_B 取值")
    ap.add_argument("--money", type=float, default=cfg.INITIAL_CAPITAL)
    ap.add_argument("--rsi", type=float, nargs="+", default=None,
                    help="同时扫描 REV_OVERSOLD_RSI（不传则只扫 %B）")
    args = ap.parse_args()

    baseline = cfg.REV_OVERSOLD_PCT_B
    print(f"\n{'=' * 78}")
    print(f"  adaptive_rotation 均值回归阈值扫描  (%B 入场阈值)")
    print(f"  扫描值: {args.values}   （当前线上值 = {baseline}）")
    print(f"  说明: 阈值越大越容易触发 —— 0.0=跌破下轨, 0.2=接近下轨20%区间内")
    print(f"{'=' * 78}")

    combos = [(th, rs) for th in args.values for rs in (args.rsi or [None])]
    results: dict[tuple, dict[str, dict]] = {}
    t0 = time.time()
    total = len(combos) * len(PERIODS)
    n = 0
    for th, rs in combos:
        results[(th, rs)] = {}
        for pname, start, end in PERIODS:
            n += 1
            ts = time.time()
            try:
                r = run_once(th, start, end, args.money, rs)
            except Exception as e:  # 单个失败不中断整轮扫描
                r = {"累计收益率": f"ERR {type(e).__name__}", "年化收益率": "--",
                     "最大回撤": "--", "夏普比率": "--", "切换次数": 0, "均值回归开仓": 0}
            results[(th, rs)][pname] = r
            print(f"  [{n}/{total}] %B={th:<4} RSI={rs!s:<5} {pname:<8} "
                  f"累计={r['累计收益率']:>9} 回撤={r['最大回撤']:>7} "
                  f"夏普={r['夏普比率']!s:>7} 切换={r['切换次数']:>3} "
                  f"均值回归开仓={r['均值回归开仓']:>2}  ({time.time() - ts:.1f}s)")

    # ── 汇总表 ──
    for pname, _, _ in PERIODS:
        print(f"\n── {pname} ──")
        print(f"  {'%B阈值':<8}{'RSI阈值':<8}{'累计收益':>10}{'年化':>10}{'最大回撤':>10}{'夏普':>8}{'切换':>6}{'均值回归开仓':>12}")
        for th, rs in combos:
            r = results[(th, rs)][pname]
            mark = "  ← 当前" if (abs(th - baseline) < 1e-9 and rs is None) else ""
            print(f"  {th:<8}{rs!s:<8}{r['累计收益率']:>10}{r['年化收益率']:>10}"
                  f"{r['最大回撤']:>10}{r['夏普比率']!s:>8}{r['切换次数']:>6}"
                  f"{r['均值回归开仓']:>12}{mark}")

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    print("注：本例为单参数扫描，未做多重比较校正；样本为单条历史路径，"
          "结论不等于未来表现。决策请结合机制合理性判断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
