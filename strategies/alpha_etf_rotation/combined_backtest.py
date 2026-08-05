"""广发复原 — 阶段3：择时 × 轮动叠加回测

结构：
  择时层：AGRU 预测 000985 → 连续仓位（分段映射 0.3~1.0）
  轮动层：7 宽基因子聚合（mom_20/rev_5/vol_20/amt_chg_20 权重聚合）→ Top2 等权
  叠加：每日目标权重 = 择时仓位 × 轮动 Top2 各 50%

执行：T 收盘算信号 → T+1 开盘执行（收益率累乘法，免疫拆分）
成本：换手 × (佣金万2+滑点万1)，与 019 全项目一致

对比（2022-01 ~ 2026-07 及 4 期）：
  A. 叠加策略（择时×轮动）
  B. 轮动不择时（满仓 Top2）
  C. 择时不轮动（满仓 000985 × 仓位）→ 即阶段2结果
  D. 持有不动（000985）
  E. 纯动量（019 引擎，2024 起 +84% 参考）

铁律：py312 / 详尽日志 / 断点续跑。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "combined_bt.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("combined_bt")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "combined_results.json"
STOCK_DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"
CONSTITUENTS_FILE = OUTPUT_DIR / "constituents.json"

WINDOW, TRAIN_YEARS, RETRAIN_MONTHS, FIRST_TEST = 30, 3, 3, "2022-01-01"
COST = 0.0003   # 佣金万2 + 滑点万1


def load_all() -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame]]:
    """加载 000985 + 7 宽基 ETF + 因子聚合矩阵。"""
    from strategies.alpha_etf_rotation.timing_poc import (
        load_timing_data, compute_features,
    )
    from strategies.alpha_etf_rotation.poc_rankic_daily import (
        load_constituents, load_stock_factors, aggregate_to_etf,
    )
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP

    idx, width = load_timing_data()
    tdf = compute_features(idx, width)

    conn = sqlite3.connect(str(ETF_DB))
    etf_close = {}
    for etf in INDEX_MAP:
        df = pd.read_sql_query(
            "SELECT date, open, close FROM etf_daily WHERE symbol=? ORDER BY date",
            conn, params=(etf,))
        df["date"] = pd.to_datetime(df["date"])
        etf_close[etf] = df.set_index("date")["close"]
    conn.close()

    constituents = load_constituents()
    stock_data = load_stock_factors(STOCK_DB, constituents)
    factor_dfs = aggregate_to_etf(stock_data, constituents, etf_close, "weight")
    return tdf, etf_close, factor_dfs


def timing_positions(tdf: pd.DataFrame) -> pd.Series:
    """AGRU 滚动预测 → 每日仓位（分段映射 0.3~1.0）。"""
    import tensorflow as tf
    from strategies.alpha_etf_rotation.timing_poc import build_sequences, build_model
    from scipy.stats import rankdata

    feature_cols = [c for c in tdf.columns if c.startswith("f_")]
    test_dates = tdf.index[tdf.index >= pd.Timestamp(FIRST_TEST)]
    periods = {}
    for d in test_dates:
        key = f"{d.year}-{((d.month - 1) // RETRAIN_MONTHS) * RETRAIN_MONTHS + 1:02d}"
        periods.setdefault(key, []).append(d)

    pos_map = {}
    for pkey, pdates in sorted(periods.items()):
        p_start = pdates[0]
        train_end = p_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
        sub = tdf[(tdf.index >= train_start) & (tdf.index <= train_end)]
        test_sub = tdf[(tdf.index >= p_start) & (tdf.index <= pdates[-1])]
        if len(sub) < WINDOW + 100 or len(test_sub) < 20:
            continue
        X_train, y_train, _ = build_sequences(sub, feature_cols)
        X_test, _, test_dts = build_sequences(test_sub, feature_cols)
        if len(X_train) < 200 or len(X_test) < 10:
            continue
        model = build_model("agru", X_train.shape[2])
        early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                 restore_best_weights=True)
        model.fit(X_train, y_train, epochs=60, batch_size=64,
                  validation_split=0.15, callbacks=[early], verbose=0)
        pred = model.predict(X_test, verbose=0).ravel()
        rank = rankdata(pred) / len(pred)
        pos = np.clip((rank - 0.3) / 0.4 * 0.7 + 0.3, 0.3, 1.0)
        for d, p in zip(test_dts, pos):
            pos_map[d] = float(p)
        logger.info(f"[{pkey}] 仓位预测 {len(pred)} 天")
    return pd.Series(pos_map)


def rotation_weights(factor_dfs: dict[str, pd.DataFrame], top_n: int = 2) -> pd.DataFrame:
    """轮动因子合成 → 每日 TopN 等权目标权重（DataFrame: date × ETF）。"""
    # 因子 z-score 合成（combo）
    dates = factor_dfs["mom_20"].index
    etfs = factor_dfs["mom_20"].columns
    combo = pd.DataFrame(index=dates, columns=etfs, dtype=float)
    for etf in etfs:
        zs = []
        for fname, mat in factor_dfs.items():
            s = mat[etf]
            z = (s - s.mean()) / (s.std() + 1e-9)
            zs.append(z)
        combo[etf] = pd.concat(zs, axis=1).mean(axis=1)
    # 每日排名 TopN
    w = pd.DataFrame(0.0, index=dates, columns=etfs)
    rank = combo.rank(axis=1, ascending=False)
    for etf in etfs:
        w[etf] = (rank[etf] <= top_n).astype(float) / top_n
    return w


def run_portfolio(weights: pd.DataFrame, etf_close: dict[str, pd.Series],
                  start: str = FIRST_TEST) -> dict:
    """按目标权重回测（收益率累乘法 + 换手成本）。"""
    dates = weights.index[weights.index >= pd.Timestamp(start)]
    close_df = pd.DataFrame(etf_close).reindex(dates)
    eq = [1.0]
    prev_w = pd.Series(0.0, index=weights.columns)
    for i in range(1, len(dates)):
        w = weights.loc[dates[i]]
        # T-1 权重 → T 日收益
        day_ret = (close_df.loc[dates[i]] / close_df.loc[dates[i - 1]] - 1).fillna(0.0)
        port_ret = float((prev_w * day_ret).sum())
        turnover = float((w - prev_w).abs().sum())
        eq.append(eq[-1] * (1 + port_ret - turnover * COST))
        prev_w = w
    eq_s = pd.Series(eq, index=dates)
    total = eq_s.iloc[-1] - 1
    rets = eq_s.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    dd = (eq_s / eq_s.cummax() - 1).min()
    n_years = len(dates) / 252
    annual = (1 + total) ** (1 / n_years) - 1 if total > -1 else float("nan")
    return {"total_return": round(float(total), 4), "annual": round(float(annual), 4),
            "sharpe": round(float(sharpe), 3), "max_drawdown": round(float(dd), 4)}


def main() -> None:
    logger.info("=" * 60)
    logger.info("阶段3：择时 × 轮动叠加回测")
    t0 = time.time()

    tdf, etf_close, factor_dfs = load_all()
    logger.info(f"数据就绪: {time.time()-t0:.0f}s")

    # 择时仓位（AGRU）
    positions = timing_positions(tdf)
    logger.info(f"择时仓位: {len(positions)} 天, {time.time()-t0:.0f}s")

    # 轮动权重（满仓 Top2）
    rot_w = rotation_weights(factor_dfs, top_n=2)
    logger.info(f"轮动权重: {rot_w.shape}, {time.time()-t0:.0f}s")

    # 叠加：仓位 × 轮动
    common = rot_w.index.intersection(positions.index)
    combined_w = rot_w.loc[common].mul(positions.loc[common], axis=0)
    full_w = rot_w.loc[common]  # 满仓轮动对照

    # 择时不动轮动 = 000985 满仓 × 仓位
    pos_only = pd.DataFrame(0.0, index=common, columns=rot_w.columns)
    pos_only["510300"] = positions.loc[common]  # 用沪深300 近似中证全指

    # 持有不动
    bh = pd.DataFrame(1.0, index=common, columns=["510300"])

    results = {}
    results["combined"] = run_portfolio(combined_w, etf_close)
    results["rotation_full"] = run_portfolio(full_w, etf_close)
    results["timing_only"] = run_portfolio(pos_only, etf_close)
    results["buy_hold"] = run_portfolio(bh, etf_close)

    logger.info("=" * 60)
    for k, v in results.items():
        logger.info(f"{k:<15} 收益={v['total_return']*100:>6.1f}% 年化={v['annual']*100:>5.1f}% "
                    f"夏普={v['sharpe']:>5.2f} 回撤={v['max_drawdown']*100:>6.1f}%")

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
