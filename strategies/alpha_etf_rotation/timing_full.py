"""广发复原 — 阶段2 收尾：AGRU 择时串联净值回测

修复 POC 的两个评估局限：
  1. 净值串联：20 期预测合并为一条完整净值曲线（2022-01 ~ 2026-07）
  2. 仓位映射对比：纯分位（0~1）vs 分段映射（<0.3→0.3, >0.7→1.0）

T 收盘信号 → T+1 开盘执行 → T+1 收盘结算；成本万2+万1。
对比基准：持有不动（中证全指 000985）。
"""
from __future__ import annotations

import json
import logging
import os
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("timing_full")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "timing_full_results.json"

from strategies.alpha_etf_rotation.timing_poc import (
    load_timing_data, compute_features, build_sequences, build_model,
    WINDOW, TRAIN_YEARS, RETRAIN_MONTHS, FIRST_TEST,
)
import tensorflow as tf


def segment_position(rank: np.ndarray) -> np.ndarray:
    """分段仓位：分位 <0.3 → 0.3（底仓），>0.7 → 1.0（满仓），中间线性。"""
    return np.clip((rank - 0.3) / 0.4 * 0.7 + 0.3, 0.3, 1.0)


def full_backtest(positions: pd.Series, idx: pd.DataFrame, cost: float = 0.0003) -> dict:
    """串联净值回测：T 信号 → T+1 开盘执行 → T+1 收盘结算。"""
    open_px, close_px = idx["open"], idx["close"]
    dates = positions.index
    eq = [1.0]
    pos_prev = 0.0
    for i in range(1, len(dates)):
        if i + 1 >= len(dates):
            eq.append(eq[-1])
            continue
        nxt = dates[i + 1]
        if nxt not in close_px.index:
            eq.append(eq[-1])
            continue
        day_ret = close_px.loc[nxt] / open_px.loc[nxt] - 1
        pos = positions.iloc[i]
        turnover = abs(pos - pos_prev)
        eq.append(eq[-1] * (1 + pos * day_ret - turnover * cost))
        pos_prev = pos
    eq_s = pd.Series(eq[:len(dates)], index=dates)
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
    logger.info("阶段2 收尾：AGRU 择时串联净值回测")
    t0 = time.time()

    idx, width = load_timing_data()
    df = compute_features(idx, width)
    feature_cols = [c for c in df.columns if c.startswith("f_")]

    # 滚动 WF（与 POC 同结构），保存每期预测
    test_dates = df.index[df.index >= pd.Timestamp(FIRST_TEST)]
    periods = {}
    for d in test_dates:
        key = f"{d.year}-{((d.month - 1) // RETRAIN_MONTHS) * RETRAIN_MONTHS + 1:02d}"
        periods.setdefault(key, []).append(d)

    all_pos = {}   # date -> (纯分位仓位, 分段仓位)
    all_ics = []
    for pkey, pdates in sorted(periods.items()):
        p_start, p_end = pdates[0], pdates[-1]
        train_end = p_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=TRAIN_YEARS)
        sub = df[(df.index >= train_start) & (df.index <= train_end)]
        test_sub = df[(df.index >= p_start) & (df.index <= p_end)]
        if len(sub) < WINDOW + 100 or len(test_sub) < 20:
            continue
        X_train, y_train, _ = build_sequences(sub, feature_cols)
        X_test, y_test, test_dts = build_sequences(test_sub, feature_cols)
        if len(X_train) < 200 or len(X_test) < 10:
            continue

        model = build_model("agru", X_train.shape[2])
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=60, batch_size=64,
                  validation_split=0.15, callbacks=[early], verbose=0)
        pred = model.predict(X_test, verbose=0).ravel()

        from scipy.stats import spearmanr, rankdata
        ic, _ = spearmanr(pred, y_test)
        all_ics.append(ic if not np.isnan(ic) else 0.0)

        rank = rankdata(pred) / len(pred)
        pos_pure = rank
        pos_seg = segment_position(rank)
        for d, p1, p2 in zip(test_dts, pos_pure, pos_seg):
            all_pos[d] = (p1, p2)
        logger.info(f"[{pkey}] IC={ic:+.4f} 预测{len(pred)}天")

    # 串联净值
    pos_s = pd.Series({d: v[0] for d, v in all_pos.items()})
    pos_s2 = pd.Series({d: v[1] for d, v in all_pos.items()})
    logger.info(f"串联预测天数: {len(pos_s)}")

    r1 = full_backtest(pos_s, idx)
    r2 = full_backtest(pos_s2, idx)
    # 持有不动
    bh_dates = pos_s.index
    r3 = full_backtest(pd.Series(1.0, index=bh_dates), idx)

    logger.info("=" * 60)
    logger.info(f"期均IC: {np.mean(all_ics):+.4f} (n={len(all_ics)})")
    logger.info(f"纯分位仓位:  收益={r1['total_return']*100:.1f}% 年化={r1['annual']*100:.1f}% "
                f"夏普={r1['sharpe']} 回撤={r1['max_drawdown']*100:.1f}%")
    logger.info(f"分段仓位:    收益={r2['total_return']*100:.1f}% 年化={r2['annual']*100:.1f}% "
                f"夏普={r2['sharpe']} 回撤={r2['max_drawdown']*100:.1f}%")
    logger.info(f"持有不动:    收益={r3['total_return']*100:.1f}% 年化={r3['annual']*100:.1f}% "
                f"夏普={r3['sharpe']} 回撤={r3['max_drawdown']*100:.1f}%")

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps({
        "mean_ic": round(float(np.mean(all_ics)), 4), "n_periods": len(all_ics),
        "pure": r1, "segmented": r2, "buy_hold": r3,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
