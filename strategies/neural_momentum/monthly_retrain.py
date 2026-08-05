"""Neural Momentum 月度重训（月末 23:00 cron 触发）

铁律一：7 只宽基预测并行（subprocess，每 ETF 独立进程）
铁律二：nohup 解绑（cron 天然解绑）
铁律四：断点续跑（scores_{etf}.csv 存在则跳过）
铁律五：详尽日志（阶段+进度+ETA）+ 运行时自检（评分 NaN 比例）
铁律六：py312 运行（cron 用绝对路径）

流程：
  1. 特征构建（个股因子聚合，~35 秒）
  2. 7 只宽基 AGRU 预测（并行，~3-4 分钟）
  3. 合并生成 neural_scores.csv（供每日模拟盘读取）
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "neural_monthly_retrain.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("neural_retrain")

OUTPUT_DIR = Path(__file__).parent / "output"
SCORES_FILE = OUTPUT_DIR / "neural_scores.csv"
STOCK_DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"
CONSTITUENTS = Path(__file__).parent.parent / "alpha_etf_rotation" / "output" / "constituents.json"
PY312 = "/home/zhulei/anaconda3/envs/zhulei_py312/bin/python"

# 引擎池 7 只宽基（与回测/模拟盘一致）
WIDE7 = ["510050", "510300", "510500", "512100", "159915", "588000", "510180"]
N_CONCURRENT = 7


def build_features() -> None:
    """阶段1：特征构建（复用 alpha_etf_rotation 因子聚合）。"""
    t0 = time.time()
    from strategies.alpha_etf_rotation.poc_rankic_daily import (
        load_constituents, load_stock_factors, aggregate_to_etf,
    )
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP
    import sqlite3

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
    for etf in INDEX_MAP:
        close, vol = etf_close[etf], etf_vol[etf]
        d = pd.DataFrame(index=close.index)
        for fname, mat in factor_dfs.items():
            d[f"agg_{fname}"] = mat[etf]
        d["etf_mom20"] = close.pct_change(20)
        d["etf_mom60"] = close.pct_change(60)
        d["etf_vol20"] = close.pct_change().rolling(20).std()
        d["etf_vol_ratio"] = vol.rolling(5).mean() / vol.rolling(20).mean()
        d["y"] = close.shift(-5) / close - 1
        d.to_csv(OUTPUT_DIR / f"feat_{etf}.csv")
    logger.info(f"特征构建完成: {len(INDEX_MAP)} 只, {time.time()-t0:.0f}s")


def run_predictions() -> None:
    """阶段2：7 只宽基并行预测（铁律一：独立进程）。"""
    t0 = time.time()
    procs = []
    for etf in WIDE7:
        out_f = OUTPUT_DIR / f"scores_{etf}.csv"
        if out_f.exists():
            logger.info(f"[{etf}] 已有评分，跳过")
            continue
        p = subprocess.Popen(
            [PY312, "-m", "strategies.neural_momentum.predict_one", "--etf", etf],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "KMP_AFFINITY": "", "OMP_NUM_THREADS": "2"},
            stdout=open(PROJECT_ROOT / "logs" / f"predict_{etf}.log", "w"),
            stderr=subprocess.STDOUT,
        )
        procs.append((etf, p))
        logger.info(f"[{etf}] 启动 PID {p.pid}（{len(procs)}/{N_CONCURRENT}）")

    for etf, p in procs:
        p.wait()
        status = "✅" if p.returncode == 0 else f"❌({p.returncode})"
        logger.info(f"[{etf}] {status} 完成, 累计 {time.time()-t0:.0f}s")


def merge_scores() -> None:
    """阶段3：合并 + z-score → neural_scores.csv（运行时自检）。"""
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP
    series = {}
    for etf in INDEX_MAP:
        f = OUTPUT_DIR / f"scores_{etf}.csv"
        if f.exists():
            series[etf] = pd.read_csv(f, index_col=0, parse_dates=True)["score"]
    df = pd.DataFrame(series).sort_index()
    df = (df - df.mean()) / (df.std() + 1e-9)
    df = df.round(4)
    df.to_csv(SCORES_FILE)

    # 铁律五：运行时自检
    wide_nan = df[WIDE7].isna().mean()
    logger.info(f"评分已保存: {SCORES_FILE} ({df.shape})")
    logger.info(f"7 宽基 NaN 比例: {wide_nan.round(3).to_dict()}")
    if (wide_nan > 0.5).any():
        logger.error("自检失败：宽基评分缺失 >50%，重训结果不可用！")
        raise SystemExit(1)
    if df[WIDE7].std().min() < 1e-6:
        logger.error("自检失败：评分无方差（常数预测）！")
        raise SystemExit(1)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Neural Momentum 月度重训")
    logger.info(f"Python: {sys.executable} | 池: {len(WIDE7)} 只宽基")
    t0 = time.time()

    build_features()
    run_predictions()
    merge_scores()
    logger.info(f"重训完成，总耗时 {time.time()-t0:.0f}s（模型就绪，每日模拟盘将自动读取新评分）")


if __name__ == "__main__":
    main()
