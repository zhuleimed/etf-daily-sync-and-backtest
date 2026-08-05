"""风险择时回测：动量轮动 + LightGBM 风险开关 vs 纯动量

信号时序（严格 T+1，与 019 纪律一致）：
  T-1 日收盘 → 动量排名 + 风险分（每月重训的 LightGBM 预测）
  T 日开盘 → 执行（风险分>阈值 → 空仓；否则持有动量第 1 名）

成本：佣金万2 双向 + 滑点万1（ETF 无印花税）。

对比组（并行）：纯动量 / 阈值 0.50 / 0.65 / 0.80
输出：4 期（2021-2026 全周期 / 2024 / 2025 / 2026）收益·回撤·年化·夏普

铁律：py312 / 断点续跑（组合结果落盘）/ 详尽日志。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from strategies.lstm_etf_rotation.config import ETF_SYMBOLS, DB_PATH
from strategies.risk_timing.quick_test import (
    load_pool_data, compute_features, make_label, FEATURE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "risk_backtest.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_bt")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "switch_backtest_results.json"

INITIAL_CAPITAL = 10000.0
COMMISSION = 0.0002   # 佣金万2（双向）
SLIPPAGE = 0.0001     # 滑点万1
MOMENTUM_WINDOW = 20
TRAIN_MONTHS = 12     # 风险模型训练窗口（滚动 12 个月）
MIN_TRAIN_DAYS = 120  # 训练数据不足则用默认风险分 0.5
# ── 019 动量轮动核心规则（final-config 2026-06-23） ──
MIN_HOLD_DAYS = 10           # 最小持仓期
MIN_SWITCH_CONVICTION = 0.03  # 切换置信度：第一名动量须超持仓 3%


# ════════════════════════════════════════════════════════
# 数据与风险分
# ════════════════════════════════════════════════════════

def load_etf_data() -> dict[str, pd.DataFrame]:
    """加载 48 只 ETF 日线（含自算 momentum 列）。"""
    conn = sqlite3.connect(str(PROJECT_ROOT / DB_PATH))
    etf_data = {}
    for sym in ETF_SYMBOLS:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM etf_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,),
        )
        df["date"] = pd.to_datetime(df["date"])
        df["momentum"] = df["close"].pct_change(MOMENTUM_WINDOW)
        df = df.reset_index(drop=True)
        etf_data[sym] = df
    conn.close()
    return etf_data


def compute_risk_scores(
    pool_df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
) -> pd.Series:
    """逐日风险分（P(未来21日回撤>3%)）。

    月度重训：每月首个交易日，用截至该日前 train_months 个月的数据训练
    LightGBM，预测本月所有交易日的风险分。训练数据不足时风险分=0.5（中性）。
    """
    import lightgbm as lgb

    df = compute_features(pool_df)
    df["label"] = make_label(df)
    df = df.dropna(subset=FEATURE_COLS)

    scores = pd.Series(0.5, index=df.index, dtype=float)
    # 每月分组
    months = df.index.to_period("M").unique()
    logger.info(f"风险模型月度重训：{len(months)} 个月")

    for m in months:
        month_end = df.index[df.index.to_period("M") == m].max()
        # 训练窗口：截至上月月末往前 train_months 个月
        train_start = (month_end - pd.DateOffset(months=train_months)).strftime("%Y-%m-%d")
        train_df = df[(df.index >= train_start) & (df.index <= month_end)]
        if len(train_df) < MIN_TRAIN_DAYS:
            continue  # 默认 0.5

        X = train_df[FEATURE_COLS].values
        y = train_df["label"].values
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            num_leaves=16, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model.fit(X, y)

        # 预测本月各交易日
        month_mask = df.index.to_period("M") == m
        month_df = df[month_mask]
        X_pred = month_df[FEATURE_COLS].values
        proba = model.predict_proba(X_pred)[:, 1]
        scores.loc[month_df.index] = proba

    return scores


# ════════════════════════════════════════════════════════
# 回测引擎
# ════════════════════════════════════════════════════════

def backtest(
    etf_data: dict[str, pd.DataFrame],
    risk_scores: pd.Series,
    threshold: float,
    start: str = "2021-01-01",
) -> dict:
    """单组合回测：动量轮动 + 风险开关（threshold 为 None 表示纯动量）。"""
    # 共同交易日
    dates = sorted(set.intersection(*[set(df["date"]) for df in etf_data.values()]))
    dates = pd.DatetimeIndex([d for d in dates if d >= pd.Timestamp(start)])
    dates = dates.intersection(risk_scores.index)
    if len(dates) < 100:
        return {"status": "skipped", "reason": "日期不足"}

    # date → 全局索引映射
    def _locate(sym: str, date: pd.Timestamp) -> int:
        s = etf_data[sym]["date"]
        return s.searchsorted(date)

    cash = INITIAL_CAPITAL
    hold_sym: str | None = None
    hold_shares = 0
    hold_days = 0                 # 已持仓天数（min_hold 用）
    equity = []
    trades = []

    # 5 日动量列（短期动量确认用，对齐 019 SHORT_TERM_MOMENTUM_CHECK）
    mom5 = {}
    for sym, df in etf_data.items():
        s = df["close"]
        mom5[sym] = (s / s.shift(5) - 1).reset_index(drop=True)

    prev_signal_sym: str | None = None   # T-1 日信号（None=空仓/风险关闭）
    prev_risk_off: bool = False          # T-1 日风险开关

    for i, date in enumerate(dates):
        # ── T 日开盘执行 T-1 信号 ──
        if prev_signal_sym is not None:
            if hold_sym is not None and hold_sym != prev_signal_sym:
                # 切换：卖出旧持仓（T 日开盘）
                idx = _locate(hold_sym, date)
                if idx < len(etf_data[hold_sym]):
                    px = etf_data[hold_sym].loc[idx, "open"] * (1 - SLIPPAGE)
                    proceeds = hold_shares * px
                    cash += proceeds * (1 - COMMISSION)
                    trades.append((date, "sell", hold_sym, hold_shares, px))
                    hold_sym, hold_shares, hold_days = None, 0, 0
            if hold_sym is None and not prev_risk_off:
                # 买入信号目标（T 日开盘）
                idx = _locate(prev_signal_sym, date)
                if idx < len(etf_data[prev_signal_sym]):
                    px = etf_data[prev_signal_sym].loc[idx, "open"] * (1 + SLIPPAGE)
                    budget = cash * (1 - COMMISSION)
                    shares = int(budget // px // 100) * 100
                    if shares > 0:
                        cost = shares * px
                        cash -= cost * (1 + COMMISSION)
                        hold_sym, hold_shares, hold_days = prev_signal_sym, shares, 0
                        trades.append((date, "buy", hold_sym, shares, px))
        elif prev_risk_off and hold_sym is not None:
            # 风险触发：T 日开盘清仓避险
            idx = _locate(hold_sym, date)
            if idx < len(etf_data[hold_sym]):
                px = etf_data[hold_sym].loc[idx, "open"] * (1 - SLIPPAGE)
                proceeds = hold_shares * px
                cash += proceeds * (1 - COMMISSION)
                trades.append((date, "sell", hold_sym, hold_shares, px))
                hold_sym, hold_shares, hold_days = None, 0, 0

        # ── 估值 ──
        equity_val = cash
        if hold_sym is not None:
            idx = _locate(hold_sym, date)
            if idx < len(etf_data[hold_sym]):
                equity_val += hold_shares * etf_data[hold_sym].loc[idx, "close"]
        equity.append(equity_val)

        # ── 计算 T+1 日信号（用 T 日收盘） ──
        # 动量排名（对齐 019 引擎规则：min_hold + 短期动量确认 + 切换置信度 + 空仓需动量>0）
        mom = {}
        for sym, df in etf_data.items():
            idx = _locate(sym, date)
            if idx >= MOMENTUM_WINDOW and idx < len(df):
                mom[sym] = df.loc[idx, "momentum"]
        valid = {k: v for k, v in mom.items() if not pd.isna(v)}
        top = max(valid, key=valid.get) if valid else None
        top_mom = valid.get(top) if top else None

        def _short_term_ok(sym: str, idx: int) -> bool:
            """短期动量确认（对齐 019 SHORT_TERM_MOMENTUM_CHECK）：
            目标 5 日动量 > -0.5% 且动能未衰减（5日/5 >= 20日/15）。"""
            if idx < 6:
                return True
            t5 = mom5[sym].iloc[idx - 1]   # idx-1 防 look-ahead
            if t5 <= -0.005:
                return False
            t20 = mom[sym]
            if not pd.isna(t20) and t20 > 0:
                if t5 / 5 < t20 / 15:
                    return False
            return True

        if hold_sym is not None:
            hold_days += 1
            if hold_days < MIN_HOLD_DAYS or top is None or top == hold_sym:
                new_signal = hold_sym
            elif not _short_term_ok(top, i):
                new_signal = hold_sym
            elif top_mom - mom[hold_sym] > MIN_SWITCH_CONVICTION:
                new_signal = top   # 切换
            else:
                new_signal = hold_sym
        else:
            # 空仓：动量 > 0 才买入（019 引擎行为）
            new_signal = top if (top is not None and top_mom > 0) else None

        # 风险分（T 日值 → T+1 日执行）
        risk = float(risk_scores.loc[date]) if date in risk_scores.index else 0.5
        risk_off = risk > threshold if threshold is not None else False

        prev_signal_sym = new_signal if not risk_off else None
        prev_risk_off = risk_off

    # ── 指标 ──
    eq = pd.Series(equity, index=dates)
    total_ret = eq.iloc[-1] / INITIAL_CAPITAL - 1
    rets = eq.pct_change().dropna()
    n_years = len(dates) / 252
    annual = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()

    return {
        "status": "ok",
        "threshold": threshold,
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(annual), 4),
        "max_drawdown": round(float(dd), 4),
        "sharpe": round(float(sharpe), 3),
        "n_trades": len(trades),
        "n_days": len(dates),
    }


# ════════════════════════════════════════════════════════
# 主流程（并行）
# ════════════════════════════════════════════════════════

def run_one(args: tuple) -> dict:
    """单组合（供 Pool 并行）：(etf_data, risk_scores, threshold, period) → 结果"""
    etf_data, risk_scores, threshold, period = args
    start_map = {
        "full": "2021-01-01",
        "2024": "2024-01-01",
        "2025": "2025-01-01",
        "2026": "2026-01-01",
    }
    return backtest(etf_data, risk_scores, threshold, start_map[period])


def main() -> None:
    logger.info("=" * 60)
    logger.info("风险择时回测：动量 + LightGBM 风险开关")
    logger.info(f"Python: {sys.executable} | ETF 池: {len(ETF_SYMBOLS)} 只")

    t0 = time.time()
    etf_data = load_etf_data()
    pool_df = load_pool_data()
    risk_scores = compute_risk_scores(pool_df)
    logger.info(f"风险分序列: {len(risk_scores)} 天, 均值 {risk_scores.mean():.3f}")

    # 断点续跑（铁律四）
    results = []
    if RESULT_FILE.exists():
        results = json.loads(RESULT_FILE.read_text(encoding="utf-8")).get("results", [])
    done = {(r.get("threshold"), r.get("period")) for r in results}

    thresholds = [None, 0.50, 0.65, 0.80]
    periods = ["full", "2024", "2025", "2026"]
    tasks = []
    for thr in thresholds:
        for period in periods:
            if (thr, period) in done:
                continue
            tasks.append((etf_data, risk_scores, thr, period))

    logger.info(f"待跑组合: {len(tasks)} 个（并行 4 进程）")
    if tasks:
        with Pool(4) as pool:
            new_results = pool.map(run_one, tasks)
        for r in new_results:
            if r is not None and r.get("status") == "ok":
                results.append(r)
        RESULT_FILE.write_text(
            json.dumps({"results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 汇总报表 ──
    logger.info("=" * 60)
    header = f"{'组合':<22}{'期间':<8}{'收益':>8}{'年化':>8}{'回撤':>8}{'夏普':>7}{'交易':>6}"
    logger.info(header)
    for r in sorted(results, key=lambda x: (str(x.get("threshold")), str(x.get("period")))):
        thr = "纯动量" if r.get("threshold") is None else f"风险>{r['threshold']:.2f}"
        logger.info(
            f"{thr:<22}{str(r.get('period','')):<8}"
            f"{r.get('total_return', 0)*100:>7.1f}%{r.get('annual_return', 0)*100:>7.1f}%"
            f"{r.get('max_drawdown', 0)*100:>7.1f}%{r.get('sharpe', 0):>7.2f}{r.get('n_trades', 0):>6}"
        )
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
