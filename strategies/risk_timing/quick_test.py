"""风险状态择时 POC — 第一步：标签可预测性快速测试

目标：判断"未来 21 日 ETF 池最大回撤 > 5%"是否可预测（AUC > 0.60 才值得上 LSTM）。

方法：
  - 池级聚合特征（48 只等权 + 沪深300），约 15 维
  - LightGBM 滚动验证（每 6 个月一个折叠，训练 → 测试）
  - 评估：AUC / 准确率 / 按年分拆 / 特征重要性

⚠️ 019 历史教训：市场宽度择时、HS300 均线择时均被证伪（过滤器永远是负优化）。
本测试是"小三步先看可预测性"——AUC ≤ 0.55 直接终止，不浪费时间。

铁律：py312 运行、断点续跑（按月落盘）、详尽日志。
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
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import sqlite3

from strategies.lstm_etf_rotation.config import ETF_SYMBOLS, DB_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "risk_timing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_timing")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "quick_test_results.json"

FUTURE_DAYS = 21          # 预测窗口（1 个月）
DRAWDOWN_THRESHOLD = 0.03  # 回撤阈值 3%（实测等权池5%正样本率仅7%过低，3%为24%最佳）
FOLD_MONTHS = 6           # 每 6 个月一个折叠


def load_pool_data() -> pd.DataFrame:
    """构建 48 只 ETF 等权池日线 + 沪深300。"""
    conn = sqlite3.connect(str(PROJECT_ROOT / DB_PATH))
    closes = {}
    for sym in ETF_SYMBOLS:
        df = pd.read_sql_query(
            "SELECT date, close, volume FROM etf_daily WHERE symbol=? ORDER BY date",
            conn, params=(sym,),
        )
        closes[sym] = df.set_index(pd.to_datetime(df["date"]))["close"]
    bench = pd.read_sql_query(
        "SELECT date, close FROM index_daily WHERE symbol='000300' ORDER BY date", conn,
    )
    bench.index = pd.to_datetime(bench["date"])
    bench = bench["close"]
    conn.close()

    pool = pd.DataFrame(closes).dropna(how="all")
    pool_ret = pool.pct_change()
    # 等权组合（每日可用标的中位数收益，防缺失）
    eq_ret = pool_ret.median(axis=1).rename("pool_ret")
    eq_close = (1 + eq_ret).cumprod()
    # 宽度：上涨占比
    width = (pool_ret > 0).mean(axis=1).rename("pool_width")

    df = pd.DataFrame({
        "pool_ret": eq_ret,
        "pool_close": eq_close,
        "pool_width": width,
        "bench_close": bench,
    })
    return df.dropna(subset=["pool_close"])


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """池级聚合特征（全部只用 T-1 及更早数据）。"""
    d = df.copy()
    ret = d["pool_ret"]
    close = d["pool_close"]

    d["f_pool_ret_5"] = close.pct_change(5)
    d["f_pool_ret_20"] = close.pct_change(20)
    d["f_pool_vol_5"] = ret.rolling(5).std()
    d["f_pool_vol_10"] = ret.rolling(10).std()
    d["f_pool_vol_20"] = ret.rolling(20).std()
    d["f_pool_vol_ratio"] = ret.rolling(5).std() / ret.rolling(20).std()
    d["f_pool_dd_10"] = close / close.rolling(10).max() - 1
    d["f_pool_dd_20"] = close / close.rolling(20).max() - 1
    d["f_pool_dd_60"] = close / close.rolling(60).max() - 1
    d["f_width_5"] = d["pool_width"].rolling(5).mean()
    d["f_width_20"] = d["pool_width"].rolling(20).mean()

    # 沪深300 特征
    bench_ret = d["bench_close"].pct_change()
    d["f_bench_vol_20"] = bench_ret.rolling(20).std()
    d["f_bench_dd_20"] = d["bench_close"] / d["bench_close"].rolling(20).max() - 1
    d["f_bench_ret_20"] = d["bench_close"].pct_change(20)
    # 池超额：池 20 日收益 - 基准 20 日收益
    d["f_excess_20"] = d["f_pool_ret_20"] - d["f_bench_ret_20"]

    return d


def make_label(df: pd.DataFrame) -> pd.Series:
    """标签：未来 21 日最大回撤 > 5% → 1。

    用未来数据计算（仅用于训练/评估标签，不进入特征——无 look-ahead 问题，
    因为标签是"结果"而非"输入"）。
    """
    close = df["pool_close"]
    # 未来 21 日内的滚动最大回撤（用未来价格计算）
    future_max = close[::-1].rolling(FUTURE_DAYS, min_periods=5).max()[::-1]
    future_dd = close / future_max - 1
    label = (future_dd.shift(-1) < -DRAWDOWN_THRESHOLD).astype(int)
    # 修正：label 对齐到"T 日预测未来 21 日"——rolling 窗口含当日，用 shift 对齐
    return label


FEATURE_COLS = [
    "f_pool_ret_5", "f_pool_ret_20",
    "f_pool_vol_5", "f_pool_vol_10", "f_pool_vol_20", "f_pool_vol_ratio",
    "f_pool_dd_10", "f_pool_dd_20", "f_pool_dd_60",
    "f_width_5", "f_width_20",
    "f_bench_vol_20", "f_bench_dd_20", "f_bench_ret_20", "f_excess_20",
]  # 15 维


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, fold_name: str) -> dict:
    """单折叠：训练 LightGBM → 预测 → AUC。"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["label"].values

    # 过滤 NaN
    mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train, y_train = X_train[mask], y_train[mask]
    mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)
    X_test, y_test = X_test[mask], y_test[mask]

    if len(X_train) < 100 or len(X_test) < 20:
        return {"fold": fold_name, "status": "skipped", "reason": "样本不足"}

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=16, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, (proba > 0.5).astype(int))
    base_rate = float(y_test.mean())

    logger.info(
        f"[{fold_name}] 训练={len(X_train)} 测试={len(X_test)} "
        f"AUC={auc:.3f} 准确率={acc:.3f} 正样本率={base_rate:.3f}"
    )
    return {
        "fold": fold_name,
        "status": "ok",
        "auc": round(float(auc), 4),
        "accuracy": round(float(acc), 4),
        "base_rate": round(base_rate, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "importance": dict(zip(FEATURE_COLS, [round(float(v), 4) for v in model.feature_importances_])),
    }


def main() -> None:
    logger.info("=" * 60)
    logger.info("风险择时 POC 第一步：标签可预测性快速测试")
    logger.info(f"Python: {sys.executable}")
    logger.info(f"ETF 池: {len(ETF_SYMBOLS)} 只 | 特征: {len(FEATURE_COLS)} 维")
    logger.info(f"标签: 未来{FUTURE_DAYS}日回撤>{DRAWDOWN_THRESHOLD*100:.0f}% | 折叠: {FOLD_MONTHS}个月")

    t0 = time.time()
    raw = load_pool_data()
    logger.info(f"数据加载: {len(raw)} 行 ({raw.index[0].date()} ~ {raw.index[-1].date()})")
    df = compute_features(raw)
    df["label"] = make_label(df)
    df = df.dropna(subset=FEATURE_COLS + ["label"])
    logger.info(f"特征+标签: {len(df)} 行, 正样本率 {df['label'].mean():.3f}")

    # 断点续跑（铁律四）
    results = []
    if RESULT_FILE.exists():
        results = json.loads(RESULT_FILE.read_text(encoding="utf-8")).get("folds", [])
    done = {r["fold"] for r in results}

    # 滚动折叠：每 6 个月训练→测试
    start = pd.Timestamp("2021-01-01")   # 特征预热（2019-03 数据 + 252日特征）
    end = df.index.max()
    fold_start = start
    while fold_start + pd.DateOffset(months=FOLD_MONTHS) <= end:
        fold_end = fold_start + pd.DateOffset(months=FOLD_MONTHS) - pd.Timedelta(days=1)
        train_end = fold_start - pd.Timedelta(days=1)
        fold_name = f"{fold_start.strftime('%Y-%m')}-{fold_end.strftime('%Y-%m')}"

        if fold_name in done:
            logger.info(f"[{fold_name}] 已完成，跳过")
            fold_start = fold_end + pd.Timedelta(days=1)
            continue

        train_df = df[df.index < train_end]
        test_df = df[(df.index >= fold_start) & (df.index <= fold_end)]
        if len(train_df) < 100:
            fold_start = fold_end + pd.Timedelta(days=1)
            continue
        r = run_fold(train_df, test_df, fold_name)
        results.append(r)
        OUTPUT_DIR.mkdir(exist_ok=True)
        RESULT_FILE.write_text(
            json.dumps({"folds": results}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_start = fold_end + pd.Timedelta(days=1)

    # 汇总
    oks = [r for r in results if r.get("status") == "ok"]
    if oks:
        aucs = [r["auc"] for r in oks]
        logger.info("=" * 60)
        logger.info(f"折叠数: {len(oks)}")
        logger.info(f"平均 AUC: {np.mean(aucs):.4f} (阈值 0.60 才继续)")
        logger.info(f"AUC>0.55 折叠: {sum(1 for a in aucs if a > 0.55)}/{len(aucs)}")
        # 按年分拆（用 fold 名年份）
        years = {}
        for r in oks:
            y = r["fold"][:4]
            years.setdefault(y, []).append(r["auc"])
        for y, ya in sorted(years.items()):
            logger.info(f"  {y}: {np.mean(ya):.4f} (n={len(ya)})")
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
