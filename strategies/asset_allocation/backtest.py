"""资产配置策略 — 回测（月度再平衡，4 种配置方法对比）

信号时序：月末最后交易日收盘 → 计算目标权重 → 下月首个交易日开盘执行（T+1）。
成本：换手 × (佣金万2 + 滑点万1)，与 019 全项目一致。

对比基准：
  - 动量轮动（019 真实引擎，2024-01 起 +84%/-25.49%）
  - 等权持有（1/N）

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.asset_allocation.backtest
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from strategies.asset_allocation.config import (
    ASSETS, CONSTANT_WEIGHTS, INITIAL_CAPITAL, VOL_WINDOW,
    ANNUAL_FACTOR, TARGET_VOL, START_DATE, COMMISSION, SLIPPAGE, DB_PATH,
    MAX_WEIGHT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("asset_alloc")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "allocation_results.json"


# ════════════════════════════════════════════════════════
# 数据
# ════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载资产日线 → (close_df, 对齐后的交易日)。"""
    conn = sqlite3.connect(str(PROJECT_ROOT / DB_PATH))
    closes = {}
    for sym in ASSETS:
        df = pd.read_sql_query(
            "SELECT date, open, close FROM etf_daily WHERE symbol=? ORDER BY date",
            conn, params=(sym,),
        )
        df["date"] = pd.to_datetime(df["date"])
        closes[sym] = df.set_index("date")["close"]
    conn.close()

    close_df = pd.DataFrame(closes).dropna(how="all")
    # 用货币 ETF（511880 全勤）作为交易日基准
    return close_df, close_df.index


# ════════════════════════════════════════════════════════
# 权重计算（4 种方法）
# ════════════════════════════════════════════════════════

def _cap_weights(weights: dict[str, float], max_w: float) -> dict[str, float]:
    """单资产权重上限（迭代截断：超限资产固定上限，剩余按比例补）。"""
    keys = list(weights.keys())
    w = np.array([weights[k] for k in keys], dtype=float)
    for _ in range(100):
        over = w > max_w
        if not over.any():
            break
        excess = float((w[over] - max_w).sum())
        w[over] = max_w
        free = ~over & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    # 归一化（浮点误差修正）
    w = w / w.sum()
    return dict(zip(keys, w))


def risk_parity_weights(vols: pd.Series) -> dict[str, float]:
    """风险平价：权重 ∝ 1/波动率（归一化 + 单资产上限）。

    NaN 处理：个别资产停牌（如拆分期）导致波动率 NaN 时用中位数填充，
    避免整组权重被 NaN 污染。
    """
    v = vols.fillna(vols.median())
    inv = 1.0 / v.clip(lower=1e-6)
    w = inv / inv.sum()
    return _cap_weights(w.to_dict(), MAX_WEIGHT)


def constant_weights() -> dict[str, float]:
    """恒定比例（config 定义）。"""
    return {k: v for k, v in CONSTANT_WEIGHTS.items() if v > 0}


def target_vol_weights(vols: pd.Series) -> dict[str, float]:
    """目标波动率：风险平价权重整体缩放至目标组合波动率，剩余配货币。

    组合波动率 ≈ Σ w_i × σ_i（加权近似），缩放因子 = TARGET_VOL / 组合加权波动率。
    """
    rp = risk_parity_weights(vols)
    combo_vol = sum(w * v for w, v in zip(rp.values(), vols.values))
    scale = min(1.0, TARGET_VOL / (combo_vol + 1e-6))
    w = {k: v * scale for k, v in rp.items()}
    # 剩余配货币
    if "511880" in ASSETS:
        w["511880"] = w.get("511880", 0.0) + (1 - sum(w.values()))
    return {k: v for k, v in w.items() if v > 0.005}


def equal_weights() -> dict[str, float]:
    """等权 1/N（不含货币）。"""
    non_cash = [s for s in ASSETS if s != "511880"]
    return {s: 1.0 / len(non_cash) for s in non_cash}


# ════════════════════════════════════════════════════════
# 回测引擎（月度再平衡）
# ════════════════════════════════════════════════════════

def backtest(
    close_df: pd.DataFrame,
    method: str,
    start: str = START_DATE,
    end: str = "",
    return_equity: bool = False,
) -> dict | tuple[dict, pd.Series]:
    """月度再平衡回测。

    - 月末收盘算权重（用过去 VOL_WINDOW 日波动率）
    - 下月首个交易日开盘执行（T+1）
    - 每日按权重×收盘价估值；调仓日按换手收成本
    """
    dates = close_df.index[close_df.index >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < VOL_WINDOW + 60:
        return {"status": "skipped", "reason": "日期不足"}

    rets = close_df.pct_change()
    # min_periods=30：拆分停牌等短暂缺失不导致整个窗口 NaN
    vols = rets.rolling(VOL_WINDOW, min_periods=30).std() * np.sqrt(ANNUAL_FACTOR)

    # 月末日期映射 {Period: 该月最后交易日}
    month_last = {p: dates[dates.to_period("M") == p].max() for p in dates.to_period("M").unique()}

    # 目标权重计算函数
    if method == "constant":
        w_func = constant_weights
    elif method == "risk_parity":
        w_func = None  # 需波动率
    elif method == "target_vol":
        w_func = None
    elif method == "equal":
        w_func = equal_weights
    else:
        raise ValueError(method)

    # 回测状态：收益率累乘法（免疫拆分/复权基准问题——
    # 513100 纳指ETF 2022 年拆分曾导致起点基准法假跳变 -78%）
    weights: dict[str, float] = {}
    equity: list[float] = []
    trade_days = 0
    cost_accum = 0.0   # 累计成本比例
    nav = 1.0          # 组合净值（初始 1.0）
    prev_date = None

    prev_month = None
    for i, date in enumerate(dates):
        month = date.to_period("M")

        # ── 月度再平衡：本月首日执行（权重用上月月末数据计算，T-1 信息） ──
        if month != prev_month:
            if prev_month is not None:
                # 上月月末的波动率（信号时点 = 上月最后交易日）
                last_me = month_last.get(prev_month, date)
                if method in ("risk_parity", "target_vol"):
                    v = vols.loc[last_me] if last_me in vols.index else pd.Series(dtype=float)
                    if v is not None and v.notna().sum() >= 3:
                        new_w = (risk_parity_weights(v) if method == "risk_parity"
                                 else target_vol_weights(v))
                    else:
                        new_w = weights
                else:
                    new_w = w_func()

                # 换手成本：Σ|Δw| × (佣金+滑点)（一次调仓的买卖成本）
                turnover = sum(
                    abs(new_w.get(k, 0) - weights.get(k, 0))
                    for k in set(new_w) | set(weights)
                )
                cost_accum += turnover * (COMMISSION + SLIPPAGE)
                trade_days += 1
                weights = new_w
            else:
                # 首日建仓：等权/恒定直接定；波动法用当日波动率
                if method in ("risk_parity", "target_vol"):
                    v = vols.loc[date] if date in vols.index else pd.Series(dtype=float)
                    if v.notna().sum() >= 3:
                        weights = (risk_parity_weights(v) if method == "risk_parity"
                                   else target_vol_weights(v))
                    else:
                        weights = equal_weights()
                else:
                    weights = w_func()
            prev_month = month

        # ── 每日估值（收益率累乘法） ──
        if prev_date is not None and weights:
            day_ret = 0.0
            for sym, w in weights.items():
                p_prev = close_df.loc[prev_date, sym]
                p_cur = close_df.loc[date, sym]
                if not pd.isna(p_prev) and not pd.isna(p_cur) and p_prev > 0:
                    day_ret += w * (p_cur / p_prev - 1)
            nav *= (1 + day_ret)
        equity.append(INITIAL_CAPITAL * nav * (1 - cost_accum))
        prev_date = date

    # ── 指标 ──
    eq = pd.Series(equity, index=dates)
    total_ret = eq.iloc[-1] / INITIAL_CAPITAL - 1
    rets_series = eq.pct_change().dropna()
    n_years = len(dates) / 252
    annual = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets_series.mean() / rets_series.std() * np.sqrt(252) if rets_series.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()

    result = {
        "status": "ok",
        "method": method,
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual), 4),
        "max_drawdown": round(float(dd), 4),
        "sharpe": round(float(sharpe), 3),
        "n_rebalances": trade_days,
        "n_days": len(dates),
    }
    if return_equity:
        return result, eq
    return result


def backtest_equity_series(
    close_df: pd.DataFrame, method: str, start: str = START_DATE,
) -> pd.Series | None:
    """返回回测净值序列（供逐月明细分析）。"""
    r, eq = backtest(close_df, method, start, return_equity=True)
    return eq if r.get("status") == "ok" else None


# ════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("资产配置策略回测（月度再平衡）")
    logger.info(f"Python: {sys.executable} | 资产: {len(ASSETS)} 类")

    t0 = time.time()
    close_df, _ = load_data()
    logger.info(f"数据: {len(close_df)} 行 ({close_df.index[0].date()} ~ {close_df.index[-1].date()})")

    results = []
    # (名称, 起, 止)：各自然年当年表现 + 全周期
    periods = [
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026", "2026-01-01", "2026-07-31"),
        ("full", "2021-01-01", ""),
    ]
    for method in ("equal", "constant", "risk_parity", "target_vol"):
        for period, start, end in periods:
            r = backtest(close_df, method, start, end)
            if r.get("status") == "ok":
                r["period"] = period
                results.append(r)

    # 2026 年逐月收益明细（风险平价，看 6-7 月下行表现）
    try:
        from strategies.asset_allocation.backtest import backtest_equity_series
        eq26 = backtest_equity_series(close_df, "risk_parity", "2026-01-01")
        if eq26 is not None and len(eq26) > 1:
            mret = eq26.resample("ME").last().pct_change().dropna()
            logger.info("2026 年逐月收益（风险平价）:")
            for dt, v in mret.items():
                logger.info(f"  {dt.strftime('%Y-%m')}: {v*100:+.2f}%")
    except Exception as e:
        logger.warning(f"2026 月度明细计算失败: {e}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 报表
    logger.info("=" * 60)
    logger.info(f"{'方法':<12}{'期间':<8}{'收益':>8}{'年化':>8}{'回撤':>8}{'夏普':>7}")
    names = {"equal": "等权1/N", "constant": "恒定比例", "risk_parity": "风险平价", "target_vol": "目标波动率"}
    for r in sorted(results, key=lambda x: (x["method"], x["period"])):
        logger.info(
            f"{names.get(r['method'], r['method']):<12}{r['period']:<8}"
            f"{r['total_return']*100:>7.1f}%{r['annual_return']*100:>7.1f}%"
            f"{r['max_drawdown']*100:>7.1f}%{r['sharpe']:>7.2f}"
        )
    logger.info(f"总耗时: {time.time()-t0:.0f}s")
    logger.info("对比: 动量轮动(019) 2024-01起 +84.03%/-25.49%/夏普0.78")


if __name__ == "__main__":
    main()
