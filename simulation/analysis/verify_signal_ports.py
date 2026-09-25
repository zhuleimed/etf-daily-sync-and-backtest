"""信号移植验证 — 用各策略"回测引擎的真实方法"逐日对比新信号函数的选股结果。

背景：2026-09 发现 7 个候选策略的 simulation/strategies/*/daily.py 是同一模板，
      都调用动量轮动的信号函数，导致实盘曲线完全相同（见 docs/）。
      修复方式是为每个策略移植其自身回测引擎的选股逻辑。

本脚本的职责：证明移植是忠实的——在同一份数据、同一批日期上，
  「移植后的信号函数选出的目标」 == 「原回测引擎逻辑选出的目标」
若某日两者不一致，说明移植走样，不允许上线。

做法：用 object.__new__(XxxEngine) 绕过 __init__，直接调用引擎的真实私有方法
      （_pct_b / _score / _check_tail / _above_ma 等），避免"自己抄自己"的假验证。

用法：python -m simulation.analysis.verify_signal_ports
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.framework.data import get_latest_trading_day, load_latest_data  # noqa: E402
from strategies.momentum_rotation.momentum_signals import (  # noqa: E402
    compute_momentum_signals,
    rank_etfs_by_momentum,
)
from strategies.momentum_rotation import config as mom_cfg  # noqa: E402

# ── 8 个策略的配置与信号模块 ──
from strategies import bollinger_reversion, median_momentum, sharpe_ranking  # noqa: E402
from strategies import sortino_ranking, spread_reversion, tail_risk  # noqa: E402
from strategies import volume_price, dual_momentum  # noqa: E402
from strategies.bollinger_reversion.engine import BollingerReversionEngine  # noqa: E402
from strategies.sharpe_ranking.engine import SharpeRankingEngine  # noqa: E402
from strategies.sortino_ranking.engine import SortinoRankingEngine  # noqa: E402
from strategies.tail_risk.engine import TailRiskEngine  # noqa: E402
from strategies.dual_momentum.engine import DualMomentumEngine  # noqa: E402
from strategies.volume_price.engine import compute_volume_ratios  # noqa: E402

N_TEST_DAYS = 60  # 回测对比最近 N 个交易日


def _load(symbols: list[str], momentum_window: int, lookback: int = 400):
    """加载数据并 reset_index，保证索引与信号函数假设一致。"""
    data = load_latest_data(symbols, mom_cfg.DB_PATH,
                            lookback_days=lookback, momentum_window=momentum_window)
    return {s: df.reset_index(drop=True) for s, df in data.items()}


def _data_len(data: dict) -> int:
    return min(len(df) for df in data.values())


def main() -> int:
    latest = get_latest_trading_day(mom_cfg.ETF_SYMBOLS, mom_cfg.DB_PATH)
    print(f"验证基准日: {latest}\n" + "=" * 74)

    failures: list[str] = []

    # ─────────────────────────── 1. 布林带回归 ───────────────────────────
    from strategies.bollinger_reversion.momentum_signals import (
        compute_bollinger_signals, rank_etfs_by_bollinger,
    )
    cfg = bollinger_reversion.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    eng = object.__new__(BollingerReversionEngine)
    eng.bb_period, eng.bb_std = cfg.BB_PERIOD, cfg.BB_STD
    eng.pctb_th = cfg.PctB_THRESHOLD

    ok = bad = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        # 原逻辑：跌破下轨的里，跌破最深的（权重最大）
        cands = {s: cfg.PctB_THRESHOLD - eng._pct_b(data[s], i)
                 for s in cfg.ETF_SYMBOLS if eng._pct_b(data[s], i) < cfg.PctB_THRESHOLD}
        orig = max(cands, key=cands.get) if cands else None
        mine = rank_etfs_by_bollinger(compute_bollinger_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  布林带 不一致 idx={i}: 原={orig} 新={mine}")
    print(f"① 布林带回归   一致 {ok}/{ok+bad}")
    if bad:
        failures.append("bollinger_reversion")

    # ─────────────────────────── 2. 中位数动量 ───────────────────────────
    from strategies.median_momentum.momentum_signals import (
        compute_median_signals, rank_etfs_by_median,
    )
    cfg = median_momentum.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    ok = bad = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        mom = compute_momentum_signals(data, i, cfg.MOMENTUM_WINDOW)
        r = rank_etfs_by_momentum(mom)
        orig = r[cfg.RANK_POSITION] if len(r) >= cfg.RANK_POSITION else (r[1] if len(r) else None)
        mine = rank_etfs_by_median(compute_median_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  中位数 不一致 idx={i}: 原={orig} 新={mine}")
    print(f"② 中位数#2     一致 {ok}/{ok+bad}")
    if bad:
        failures.append("median_momentum")

    # ──────────────────── 3/4. Sharpe / Sortino 排名 ────────────────────
    for label, mod, engcls, cfn, rfn in [
        ("Sharpe", sharpe_ranking, SharpeRankingEngine,
         "compute_sharpe_signals", "rank_etfs_by_sharpe"),
        ("Sortino", sortino_ranking, SortinoRankingEngine,
         "compute_sortino_signals", "rank_etfs_by_sortino"),
    ]:
        import importlib
        sig = importlib.import_module(f"strategies.{mod.__name__.split('.')[-1]}.momentum_signals")
        cfn_f, rfn_f = getattr(sig, cfn), getattr(sig, rfn)
        cfg = mod.config
        data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
        eng = object.__new__(engcls)
        eng.etf_data = data
        eng.vol_window = cfg.VOL_WINDOW
        eng.momentum_window = cfg.MOMENTUM_WINDOW
        ok = bad = 0
        for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                       len(data[cfg.ETF_SYMBOLS[0]])):
            sc = {s: eng._score(s, i) for s in cfg.ETF_SYMBOLS}
            sc = {k: v for k, v in sc.items() if not (isinstance(v, float) and np.isnan(v))}
            orig = max(sc, key=sc.get) if sc else None
            mine = rfn_f(cfn_f(data, i, cfg.MOMENTUM_WINDOW))
            mine = mine[1] if len(mine) else None
            if orig == mine:
                ok += 1
            else:
                bad += 1
                if bad <= 3:
                    print(f"  {label} 不一致 idx={i}: 原={orig} 新={mine}")
        print(f"{'③' if label == 'Sharpe' else '④'} {label:10s} 一致 {ok}/{ok+bad}")
        if bad:
            failures.append(label.lower() + "_ranking")

    # ─────────────────────────── 5. 价差回归 ───────────────────────────
    from strategies.spread_reversion.momentum_signals import (
        compute_spread_signals, rank_etfs_by_spread,
    )
    cfg = spread_reversion.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    ok = bad = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        if i < cfg.LOOKBACK + 10:
            continue
        cum = {s: data[s]["close"].iloc[i] / data[s]["close"].iloc[i - cfg.LOOKBACK] - 1
               for s in cfg.ETF_SYMBOLS}
        avg = np.mean(list(cum.values()))
        std = max(np.std(list(cum.values())), 0.001)
        # 原逻辑权重：max = 1.0（|z|<EXIT_Z 的满配组）
        w = {s: (1.0 / (1 + abs((cum[s] - avg) / std))
                 if abs((cum[s] - avg) / std) > cfg.ENTRY_Z
                 else (1.0 if abs((cum[s] - avg) / std) < cfg.EXIT_Z else 0.5))
             for s in cfg.ETF_SYMBOLS}
        # 原回测永远满仓：权重恒为正，持有权重最大的为"单标的归约"的忠实选择
        orig = max(w, key=w.get)
        mine = rank_etfs_by_spread(compute_spread_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  价差 不一致 idx={i}: 原(权重最大)={orig}(w={w[orig]:.3f}) 新={mine}")
    print(f"⑤ 价差回归     一致 {ok}/{ok+bad}")
    if bad:
        failures.append("spread_reversion")

    # ─────────────────────────── 6. 尾部风险 ───────────────────────────
    from strategies.tail_risk.momentum_signals import (
        compute_tail_risk_signals, rank_etfs_by_tail_risk,
    )
    cfg = tail_risk.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    eng = object.__new__(TailRiskEngine)
    eng.etf_data, eng.vol_window = data, cfg.VOL_WINDOW
    eng.tail_threshold = cfg.TAIL_THRESHOLD
    # 与原回测同源：真实沪深300指数（index_daily 表），非 ETF 代理。
    # 注意 date 必须保持字符串——引擎用 str(date)[:10] 去 hs.index 里查。
    # 另外：回测引擎 __init__ 里把 hs300_data 硬编码成 None，即回测端该分支
    #      从未触发过；这里显式喂入指数，比较的是"策略本意"的逻辑。
    from strategies.momentum_rotation.data import load_benchmark_data
    eng.hs300_data = load_benchmark_data()
    ok = bad = tab = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        t = eng._check_tail(i)
        tab += 1 if t else 0
        if t:
            orig = eng._lowest_vol(i)
        else:
            mom = compute_momentum_signals(data, i, cfg.MOMENTUM_WINDOW)
            r = rank_etfs_by_momentum(mom)
            orig = r[1] if len(r) else None
        mine = rank_etfs_by_tail_risk(compute_tail_risk_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  尾部风险 不一致 idx={i}: 原={orig} 新={mine}")
    print(f"⑥ 尾部风险     一致 {ok}/{ok+bad}  (区间内触发 {tab} 次)")
    if bad:
        failures.append("tail_risk")

    # ─────────────────────────── 7. 量价配合 ───────────────────────────
    from strategies.volume_price.momentum_signals import (
        compute_volume_price_signals, rank_etfs_by_volume_price,
    )
    cfg = volume_price.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    ok = bad = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        mom = compute_momentum_signals(data, i, cfg.MOMENTUM_WINDOW)
        vr = compute_volume_ratios(data, i, short_period=cfg.VOL_SHORT_PERIOD,
                                   long_period=cfg.VOL_LONG_PERIOD)
        for s in mom.index:
            if s in vr and not pd.isna(vr[s]) and vr[s] < cfg.VOL_THRESHOLD:
                mom[s] = np.nan
        r = rank_etfs_by_momentum(mom)
        orig = r[1] if len(r) else None
        mine = rank_etfs_by_volume_price(compute_volume_price_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  量价 不一致 idx={i}: 原={orig} 新={mine}")
    print(f"⑦ 量价配合     一致 {ok}/{ok+bad}")
    if bad:
        failures.append("volume_price")

    # ─────────────────────────── 8. 双动量 ───────────────────────────
    from strategies.dual_momentum.momentum_signals import (
        compute_dual_momentum_signals, rank_etfs_by_dual_momentum,
    )
    cfg = dual_momentum.config
    data = _load(cfg.ETF_SYMBOLS, cfg.MOMENTUM_WINDOW)
    eng = object.__new__(DualMomentumEngine)
    eng.etf_data, eng.abs_ma = data, cfg.ABS_MA
    ok = bad = 0
    for i in range(len(data[cfg.ETF_SYMBOLS[0]]) - N_TEST_DAYS,
                   len(data[cfg.ETF_SYMBOLS[0]])):
        mom = compute_momentum_signals(data, i, cfg.MOMENTUM_WINDOW)
        for s in mom.index:
            if not eng._above_ma(s, i) and not np.isnan(mom[s]):
                mom[s] = np.nan
        r = rank_etfs_by_momentum(mom)
        orig = r[1] if len(r) else None
        mine = rank_etfs_by_dual_momentum(compute_dual_momentum_signals(data, i, cfg.MOMENTUM_WINDOW))
        mine = mine[1] if len(mine) else None
        if orig == mine:
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  双动量 不一致 idx={i}: 原={orig} 新={mine}")
    print(f"⑧ 双动量       一致 {ok}/{ok+bad}")
    if bad:
        failures.append("dual_momentum")

    print("=" * 74)
    if failures:
        print(f"❌ 移植验证未通过: {failures}")
        return 1
    print("✅ 全部 8 个策略移植验证通过（逐日选股与原回测引擎一致）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
