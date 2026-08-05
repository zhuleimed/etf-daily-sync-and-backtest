"""LSTM ETF 轮动 — 月度 Walk-Forward 主流程（POC IC 验证）

铁律二：nohup 解绑运行（启动脚本负责）
铁律四：断点续跑——每月 IC 落盘 output/ic_results.json，重跑自动跳过已完成月份
铁律五：详尽日志——每月打印样本数/耗时/IC + ETA；启动打印环境诊断
铁律六：必须用 py312 运行（禁止裸 python3）

用法:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.lstm_etf_rotation.evaluate
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 铁律一：KMP_AFFINITY 清除（import tensorflow 之前） ──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from strategies.lstm_etf_rotation.config import ETF_SYMBOLS, DB_PATH
from strategies.lstm_etf_rotation.features import compute_features, compute_market_features, FEATURE_COLS
from strategies.lstm_etf_rotation.dataset import build_dataset
from strategies.lstm_etf_rotation.train import train_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "logs" / "lstm_poc.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("lstm_poc")

OUTPUT_DIR = Path(__file__).parent / "output"
IC_FILE = OUTPUT_DIR / "ic_results.json"

# ════════════════════════════════════════════════════════
# 参数（POC）
# ════════════════════════════════════════════════════════
WINDOW = 60
TRAIN_MONTHS = 12          # 12 个月训练窗口
# 数据 2019-03-06 起，f_dd_252（一年回撤）需 252 交易日预热 ≈ 2020-03 才有效。
# 首个训练窗口 2020-04-01~2021-03-31（特征完全有效）→ 首次测试 2021-04。
FIRST_TEST = "2021-04-01"


def load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """从 etf_daily.db 加载 48 只 ETF 日线 + 沪深300 基准。"""
    conn = sqlite3.connect(str(PROJECT_ROOT / DB_PATH))
    etf_data = {}
    for sym in ETF_SYMBOLS:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM etf_daily "
            "WHERE symbol=? ORDER BY date", conn, params=(sym,)
        )
        df["date"] = pd.to_datetime(df["date"])
        etf_data[sym] = df
    bench = pd.read_sql_query(
        "SELECT date, close FROM index_daily WHERE symbol='000300' ORDER BY date",
        conn,
    )
    bench["date"] = pd.to_datetime(bench["date"])
    conn.close()

    # 特征计算
    logger.info(f"特征计算：{len(etf_data)} 只 ETF，{len(FEATURE_COLS)} 维")
    for sym, df in etf_data.items():
        etf_data[sym] = compute_features(df)
    etf_data = compute_market_features(etf_data, bench)
    return etf_data, bench


def month_end_dates(etf_data: dict[str, pd.DataFrame], start: str, end: str) -> list[str]:
    """取每月最后一个交易日的日期列表（训练窗口结束点/测试参考日）。"""
    all_dates = sorted(set().union(*[set(df["date"]) for df in etf_data.values()]))
    dates = [d for d in all_dates if start <= d.strftime("%Y-%m-%d") <= end]
    months = {}
    for d in dates:
        months[d.strftime("%Y-%m")] = d
    return [d.strftime("%Y-%m-%d") for d in months.values()]


def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank IC（Spearman 相关）。"""
    from scipy.stats import spearmanr
    if len(y_true) < 5:
        return float("nan")
    ic, _ = spearmanr(y_pred, y_true)
    return float(ic) if not np.isnan(ic) else float("nan")


def run_month(ref_date: str, train_end: str, etf_data: dict, bench: pd.DataFrame) -> dict:
    """单月 WF：训练 → 预测 → IC。"""
    t0 = time.time()
    train_start = (pd.Timestamp(train_end) - pd.DateOffset(months=TRAIN_MONTHS - 1)).strftime("%Y-%m-01")
    # 训练窗口：train_start 月初 ~ train_end 月末
    train_start = pd.Timestamp(train_start)
    # 留 21 天 purge
    ref_ts = pd.Timestamp(ref_date)

    ds = build_dataset(
        etf_data,
        train_start.strftime("%Y-%m-%d"),
        train_end,
        ref_date,
        window=WINDOW,
        sample_stride=5,
        purge_gap=21,
        feature_cols=FEATURE_COLS,
    )
    if ds is None:
        return {"ref_date": ref_date, "status": "skipped", "reason": "样本不足"}

    X_train, y_train, X_test, y_test, test_symbols = ds
    model, diag = train_lstm(
        X_train, y_train,
        window=WINDOW,
        n_features=len(FEATURE_COLS),
        units=64,
    )
    y_pred = model.predict(X_test, verbose=0).ravel()
    ic = compute_ic(y_test, y_pred)

    elapsed = time.time() - t0
    logger.info(
        f"[{ref_date}] 训练样本={len(X_train)} 测试={len(X_test)}只 "
        f"IC={ic:+.4f} 耗时={elapsed:.0f}s "
        f"(loss={diag['final_loss']:.4f} val={diag['val_loss']:.4f} "
        f"epochs={diag['n_epochs']} pred_std={diag['pred_std']:.2e})"
    )
    return {
        "ref_date": ref_date,
        "status": "ok",
        "ic": ic,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "elapsed_s": round(elapsed, 1),
        "test_symbols": test_symbols,
    }


def load_existing() -> dict:
    if IC_FILE.exists():
        return json.loads(IC_FILE.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IC_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    # 铁律五：启动诊断
    logger.info("=" * 60)
    logger.info("LSTM-ETF 月度 WF POC 启动")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"numpy: {np.__version__} pandas: {pd.__version__}")
    logger.info(f"ETF 池: {len(ETF_SYMBOLS)} 只 | 特征: {len(FEATURE_COLS)} 维 | 窗口: {WINDOW}")
    logger.info(f"训练窗口: {TRAIN_MONTHS} 个月 | 首次测试: {FIRST_TEST}")

    etf_data, bench = load_data()
    ref_dates = month_end_dates(etf_data, FIRST_TEST, "2026-07-31")
    logger.info(f"共 {len(ref_dates)} 个月待评估: {ref_dates[0]} ~ {ref_dates[-1]}")

    results = load_existing()
    done = {r["ref_date"] for r in results.get("months", []) if r.get("status") == "ok"}

    t_all = time.time()
    for i, ref in enumerate(ref_dates):
        if ref in done:
            logger.info(f"[{ref}] 已完成，跳过（断点续跑）")
            continue
        train_end = ref  # 训练窗口截至上月
        # 计算训练窗口结束日 = ref_date 前一个月的月末
        train_end_dt = (pd.Timestamp(ref) - pd.DateOffset(months=1))
        train_end_str = month_end_dates(etf_data, "2019-03-01", train_end_dt.strftime("%Y-%m-%d"))[-1] \
            if (pd.Timestamp(ref) - pd.DateOffset(months=1)) >= pd.Timestamp("2019-03-01") else None
        if train_end_str is None:
            continue
        r = run_month(ref, train_end_str, etf_data, bench)
        results.setdefault("months", []).append(r)
        save_results(results)  # 铁律四：每完成一个月立即落盘

        # ETA
        elapsed = time.time() - t_all
        done_cnt = len([m for m in results["months"] if m.get("status") == "ok"])
        avg = elapsed / max(done_cnt, 1)
        remain = (len(ref_dates) - i - 1) * avg
        logger.info(f"  进度 {i+1}/{len(ref_dates)} | ETA {remain/60:.0f} 分钟")

    # ── 汇总报告 ──
    ics = [m["ic"] for m in results.get("months", []) if m.get("status") == "ok" and not np.isnan(m["ic"])]
    if ics:
        ic_arr = np.array(ics)
        logger.info("=" * 60)
        logger.info(f"POC 完成：{len(ics)} 个月有效")
        logger.info(f"月均 IC:   {ic_arr.mean():+.4f}")
        logger.info(f"ICIR:      {ic_arr.mean() / ic_arr.std():.3f}" if ic_arr.std() > 0 else "ICIR: N/A")
        logger.info(f"IC>0 占比: {(ic_arr > 0).mean() * 100:.1f}%")
        # 按年汇总
        years = {}
        for ref, ic in zip([m["ref_date"] for m in results["months"] if m.get("status") == "ok"], ics):
            years.setdefault(ref[:4], []).append(ic)
        for y, yics in sorted(years.items()):
            logger.info(f"  {y}: {np.mean(yics):+.4f} (n={len(yics)})")
        results["summary"] = {
            "mean_ic": round(float(ic_arr.mean()), 4),
            "icir": round(float(ic_arr.mean() / ic_arr.std()), 3) if ic_arr.std() > 0 else None,
            "ic_positive_ratio": round(float((ic_arr > 0).mean()), 4),
            "n_months": len(ics),
        }
        save_results(results)
    else:
        logger.warning("无有效 IC 结果")


if __name__ == "__main__":
    main()
