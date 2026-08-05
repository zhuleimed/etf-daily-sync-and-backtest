"""Neural Momentum — 阶段A：神经预测评分生成（并行版）

铁律一：每只 ETF 独立训练 → ProcessPoolExecutor 并行（12 workers）
铁律四：每 ETF 完成立即保存（output/scores_{etf}.csv），重跑自动跳过已完成
铁律五：详尽日志（每 ETF 进度 + 总 ETA）

AGRU 预测 7 宽基 ETF 未来 5 日收益（输入 = 个股因子聚合值 + ETF 量价），
滚动 WF（3 年训练 + 3 月重训）→ 每日预测分 → neural_scores.csv。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "neural_scores.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("neural_scores")

OUTPUT_DIR = Path(__file__).parent / "output"
SCORES_FILE = OUTPUT_DIR / "neural_scores.csv"
STOCK_DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"
CONSTITUENTS = Path(__file__).parent.parent / "alpha_etf_rotation" / "output" / "constituents.json"

WINDOW = 30
TRAIN_YEARS = 3
RETRAIN_MONTHS = 3
FIRST_PREDICT = "2021-04-01"
FUTURE_DAYS = 5
N_WORKERS = 12


def build_seq(d: pd.DataFrame, feat_cols: list[str], start: str, end: str):
    X, y, dates = [], [], []
    sub = d[(d.index >= start) & (d.index <= end)]
    vals, yv = sub[feat_cols].values, sub["y"].values
    for i in range(WINDOW, len(sub)):
        feat = vals[i - WINDOW:i]
        if np.isnan(feat).any() or np.isnan(yv[i]):
            continue
        X.append(feat)
        y.append(yv[i])
        dates.append(sub.index[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(dates)


def predict_one_etf(args: tuple) -> tuple[str, str, int]:
    """单只 ETF 的滚动训练+预测（独立进程）。"""
    etf, feat_csv, feat_cols, periods = args
    out_f = OUTPUT_DIR / f"scores_{etf}.csv"
    if out_f.exists():
        return etf, "skipped", 0

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from strategies.alpha_etf_rotation.timing_poc import build_model

    d = pd.read_csv(feat_csv, index_col=0, parse_dates=True)
    etf_scores = {}
    for pkey, pdates in periods:
        p_start, p_end = pdates[0], pdates[-1]
        train_end = p_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
        X_tr, y_tr, _ = build_seq(d, feat_cols, train_start.strftime("%Y-%m-%d"),
                                  train_end.strftime("%Y-%m-%d"))
        X_te, _, te_dts = build_seq(d, feat_cols, p_start.strftime("%Y-%m-%d"),
                                    p_end.strftime("%Y-%m-%d"))
        if len(X_tr) < 200 or len(X_te) < 10:
            continue
        model = build_model("agru", X_tr.shape[2])
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=60, batch_size=64,
                  validation_split=0.15, callbacks=[early], verbose=0)
        pred = model.predict(X_te, verbose=0).ravel()
        for dt, p in zip(te_dts, pred):
            etf_scores[dt] = float(p)

    s = pd.Series(etf_scores).sort_index()
    s.to_csv(out_f, header=["score"])
    return etf, "ok", len(s)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Neural Momentum 阶段A（并行版）：AGRU 预测评分")
    logger.info(f"Python: {sys.executable} | workers: {N_WORKERS}")
    t0 = time.time()

    # ── 构建特征（主进程一次） ──
    from strategies.alpha_etf_rotation.poc_rankic_daily import (
        load_constituents, load_stock_factors, aggregate_to_etf,
    )
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP

    constituents = load_constituents()
    stock_data = load_stock_factors(STOCK_DB, constituents)
    conn = sqlite3.connect(str(ETF_DB))
    etf_close, etf_vol = {}, {}
    for etf in INDEX_MAP:
        df = pd.read_sql_query(
            "SELECT date, close, volume FROM etf_daily WHERE symbol=? ORDER BY date",
            conn, params=(etf,))
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        etf_close[etf] = df["close"]
        etf_vol[etf] = df["volume"]
    conn.close()

    factor_dfs = aggregate_to_etf(stock_data, constituents, etf_close, "weight")

    # 每 ETF 特征写入临时 CSV（worker 读取，避免进程间传大对象）
    feat_files = {}
    feat_cols = None
    for etf in INDEX_MAP:
        close, vol = etf_close[etf], etf_vol[etf]
        d = pd.DataFrame(index=close.index)
        for fname, mat in factor_dfs.items():
            d[f"agg_{fname}"] = mat[etf]
        d["etf_mom20"] = close.pct_change(20)
        d["etf_mom60"] = close.pct_change(60)
        d["etf_vol20"] = close.pct_change().rolling(20).std()
        d["etf_vol_ratio"] = vol.rolling(5).mean() / vol.rolling(20).mean()
        d["y"] = close.shift(-FUTURE_DAYS) / close - 1
        f = OUTPUT_DIR / f"feat_{etf}.csv"
        d.to_csv(f)
        feat_files[etf] = str(f)
        if feat_cols is None:
            feat_cols = [c for c in d.columns if c.startswith(("agg_", "etf_"))]
    logger.info(f"特征: {len(feat_cols)} 维 × {len(INDEX_MAP)} 只, {time.time()-t0:.0f}s")

    # ── 滚动期间 ──
    test_dates = etf_close[list(INDEX_MAP)[0]].index
    test_dates = test_dates[test_dates >= pd.Timestamp(FIRST_PREDICT)]
    periods = {}
    for d in test_dates:
        key = f"{d.year}-{((d.month - 1) // RETRAIN_MONTHS) * RETRAIN_MONTHS + 1:02d}"
        periods.setdefault(key, []).append(d)
    period_list = sorted(periods.items())
    logger.info(f"滚动期数: {len(period_list)}")

    # ── 并行训练（铁律一） ──
    tasks = [(etf, feat_files[etf], feat_cols, period_list) for etf in INDEX_MAP]
    done_cnt = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for etf, status, n in pool.map(predict_one_etf, tasks):
            done_cnt += 1
            logger.info(f"[{done_cnt}/{len(tasks)}] {etf}: {status} ({n} 天) "
                        f"ETA {(len(tasks)-done_cnt)*180/60:.0f} 分钟")

    # ── 合并 ──
    series = {}
    for etf in INDEX_MAP:
        f = OUTPUT_DIR / f"scores_{etf}.csv"
        if f.exists():
            s = pd.read_csv(f, index_col=0, parse_dates=True)["score"]
            series[etf] = s
    df = pd.DataFrame(series).sort_index()
    df = (df - df.mean()) / (df.std() + 1e-9)
    df = df.round(4)
    df.to_csv(SCORES_FILE)
    logger.info(f"评分已保存: {SCORES_FILE} ({df.shape}) 总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
