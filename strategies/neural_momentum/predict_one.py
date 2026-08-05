"""Neural Momentum — 单 ETF 预测（独立进程入口，配合 launch_predict.sh 并发）

用法: python -m strategies.neural_momentum.predict_one --etf 510300
读:   output/feat_{etf}.csv（特征，主进程生成）
写:   output/scores_{etf}.csv（断点续跑：存在则跳过）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"
WINDOW, TRAIN_YEARS, RETRAIN_MONTHS, FIRST_PREDICT, FUTURE_DAYS = 30, 3, 3, "2021-04-01", 5


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etf", required=True)
    args = parser.parse_args()
    etf = args.etf

    out_f = OUTPUT_DIR / f"scores_{etf}.csv"
    if out_f.exists():
        print(f"[{etf}] 已完成，跳过")
        return

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from strategies.alpha_etf_rotation.timing_poc import build_model

    d = pd.read_csv(OUTPUT_DIR / f"feat_{etf}.csv", index_col=0, parse_dates=True)
    feat_cols = [c for c in d.columns if c.startswith(("agg_", "etf_"))]

    # 滚动期间
    test_dates = d.index[d.index >= pd.Timestamp(FIRST_PREDICT)]
    periods = {}
    for dt in test_dates:
        key = f"{dt.year}-{((dt.month - 1) // RETRAIN_MONTHS) * RETRAIN_MONTHS + 1:02d}"
        periods.setdefault(key, []).append(dt)

    scores = {}
    for pkey, pdates in sorted(periods.items()):
        p_start = pdates[0]
        train_end = p_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
        X_tr, y_tr, _ = build_seq(d, feat_cols, train_start.strftime("%Y-%m-%d"),
                                  train_end.strftime("%Y-%m-%d"))
        X_te, _, te_dts = build_seq(d, feat_cols, p_start.strftime("%Y-%m-%d"),
                                    pdates[-1].strftime("%Y-%m-%d"))
        if len(X_tr) < 200 or len(X_te) < 10:
            continue
        model = build_model("agru", X_tr.shape[2])
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=60, batch_size=64,
                  validation_split=0.15, callbacks=[early], verbose=0)
        pred = model.predict(X_te, verbose=0).ravel()
        for dt, p in zip(te_dts, pred):
            scores[dt] = float(p)

    s = pd.Series(scores).sort_index()
    s.to_csv(out_f, header=["score"])
    print(f"[{etf}] 完成 {len(s)} 天")


if __name__ == "__main__":
    main()
