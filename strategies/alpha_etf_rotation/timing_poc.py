"""广发复原策略 — 阶段2 POC：神经网络大盘择时（连续仓位）

目标：预测中证全指（000985）次日收益率 → 连续仓位（0~100%）
区别于广发三分类（空/半/满仓）：连续仓位规避边界跳变；
区别于 019 空仓开关（铁律三教训）：仓位连续化避免"错过反弹"。

设计：
  - 特征：指数量价（收益率/波动率/RSI/MACD/偏离度/布林带）+ 市场宽度（48只ETF上涨占比）
  - 模型：GRU / LSTM / AGRU(Attention+GRU) 三模型对比
  - 训练：滚动 WF（3年训练+1年验证，每 3 个月重训）
  - 标签：y = 次日收益率（回归）
  - 仓位：预测值在训练分布的分位 → 0~100% 连续仓位
  - 回测：T收盘信号 → T+1开盘执行 → T+1收盘结算；成本万2+万1
  - 评估：时序 IC（预测 vs 实际秩相关）+ 连续仓位回测 vs 持有不动

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

# 铁律一：KMP_AFFINITY 清除（import tensorflow 之前）
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "timing_poc.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("timing_poc")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "timing_results.json"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"

WINDOW = 30              # 序列窗口（交易日）
TRAIN_YEARS = 3          # 训练窗口年数
RETRAIN_MONTHS = 3       # 重训周期
FIRST_TEST = "2022-01-01"
MODELS = ["gru", "lstm", "agru"]
FEATURE_N = 12           # 特征维度


def load_timing_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 000985 中证全指 + 48 只池 ETF（算市场宽度）。"""
    conn = sqlite3.connect(str(ETF_DB))
    idx = pd.read_sql_query(
        "SELECT date, open, close FROM index_daily WHERE symbol='000985' ORDER BY date", conn)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")

    # 48 只池日收益（宽度）
    from strategies.lstm_etf_rotation.config import ETF_SYMBOLS
    rets = {}
    for sym in ETF_SYMBOLS:
        df = pd.read_sql_query(
            "SELECT date, close FROM etf_daily WHERE symbol=? ORDER BY date", conn, params=(sym,))
        df["date"] = pd.to_datetime(df["date"])
        rets[sym] = df.set_index("date")["close"].pct_change()
    conn.close()
    pool_ret = pd.DataFrame(rets)
    width = (pool_ret > 0).mean(axis=1).rename("width")
    return idx, width


def compute_features(idx: pd.DataFrame, width: pd.Series) -> pd.DataFrame:
    """指数量价特征（全部滞后，T 日收盘后可算）。"""
    d = idx.copy()
    close = d["close"]
    ret = close.pct_change()

    d["f_ret_1"] = ret
    d["f_ret_5"] = close.pct_change(5)
    d["f_ret_20"] = close.pct_change(20)
    d["f_vol_5"] = ret.rolling(5).std()
    d["f_vol_20"] = ret.rolling(20).std()
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["f_rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["f_macd_hist"] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    # 偏离度
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    d["f_ma20_dev"] = close / ma20 - 1
    d["f_ma60_dev"] = close / ma60 - 1
    # 布林带位置
    std20 = close.rolling(20).std()
    d["f_boll_pos"] = (close - ma20) / (2 * std20 + 1e-9)
    # 市场宽度
    d["f_width"] = width.reindex(d.index)
    d["f_width_5"] = width.reindex(d.index).rolling(5).mean()

    # 标签：次日收益率（y = close[T+1]/close[T] - 1）
    d["y"] = close.shift(-1) / close - 1
    return d


def build_sequences(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    X, y, dates = [], [], []
    vals = df[feature_cols].values
    yv = df["y"].values
    for i in range(WINDOW, len(df) - 1):
        feat = vals[i - WINDOW:i]
        if np.isnan(feat).any() or np.isnan(yv[i]):
            continue
        X.append(feat)
        y.append(yv[i])
        dates.append(df.index[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.DatetimeIndex(dates)


def build_model(model_type: str, n_features: int) -> object:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)

    inp = tf.keras.layers.Input(shape=(WINDOW, n_features))
    if model_type == "lstm":
        x = tf.keras.layers.LSTM(32, dropout=0.2)(inp)
    elif model_type == "gru":
        x = tf.keras.layers.GRU(32, dropout=0.2)(inp)
    elif model_type == "agru":
        # AGRU：GRU + 注意力（Attention 加权池化）
        gru_out = tf.keras.layers.GRU(32, return_sequences=True, dropout=0.2)(inp)
        att = tf.keras.layers.Attention()([gru_out, gru_out])
        x = tf.keras.layers.Flatten()(att)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    else:
        raise ValueError(model_type)
    if model_type != "agru":
        x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def train_predict(model_type: str, X_train, y_train, X_test) -> np.ndarray:
    import tensorflow as tf
    model = build_model(model_type, X_train.shape[2])
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=60, batch_size=64,
              validation_split=0.15, callbacks=[early], verbose=0)
    return model.predict(X_test, verbose=0).ravel()


def backtest_position(positions: pd.Series, idx: pd.DataFrame, cost: float = 0.0003) -> dict:
    """连续仓位回测：T 收盘信号 → T+1 开盘执行 → T+1 收盘结算。"""
    open_px = idx["open"]
    close_px = idx["close"]
    dates = positions.index
    eq = [1.0]
    pos_prev = 0.0
    for i in range(1, len(dates)):
        # T=i 日信号 → T+1 日开盘执行 → T+1 收盘结算
        if i + 1 >= len(dates):
            eq.append(eq[-1])   # 最后一日无后续执行
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
    return {"total_return": round(float(total), 4), "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(dd), 4)}


def main() -> None:
    logger.info("=" * 60)
    logger.info("阶段2 POC：神经网络大盘择时（连续仓位）")
    logger.info(f"Python: {sys.executable}")

    t0 = time.time()
    idx, width = load_timing_data()
    df = compute_features(idx, width)
    logger.info(f"数据: {len(df)} 行 ({df.index[0].date()} ~ {df.index[-1].date()})")

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    logger.info(f"特征: {len(feature_cols)} 维: {feature_cols}")

    results = load_existing()
    done = {r["period"] for r in results.get("months", []) if r.get("status") == "ok"}

    # 滚动 WF：每 RETRAIN_MONTHS 个月重训，训练窗口 TRAIN_YEARS 年
    test_dates = df.index[df.index >= pd.Timestamp(FIRST_TEST)]
    periods = {}
    for d in test_dates:
        key = f"{d.year}-{((d.month - 1) // RETRAIN_MONTHS) * RETRAIN_MONTHS + 1:02d}"
        periods.setdefault(key, []).append(d)
    period_list = sorted(periods.items())

    for pkey, pdates in period_list:
        if pkey in done:
            continue
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

        month_res = {"period": pkey, "status": "ok", "models": {}}
        for m in MODELS:
            try:
                pred = train_predict(m, X_train, y_train, X_test)
                # 时序 IC
                from scipy.stats import spearmanr
                ic, _ = spearmanr(pred, y_test)
                # 连续仓位：预测值分位映射
                from scipy.stats import rankdata
                pos = (rankdata(pred) / len(pred)).astype(float)  # 0~1 分位 = 仓位
                # 仓位压缩到 0.2~1.0（最小 20% 仓位，规避空仓错过反弹）
                pos = 0.2 + 0.8 * pos
                pos_s = pd.Series(pos, index=test_dts)
                bt = backtest_position(pos_s, idx)
                month_res["models"][m] = {
                    "ic": round(float(ic), 4) if not np.isnan(ic) else None,
                    **bt,
                }
                logger.info(f"[{pkey}][{m}] IC={month_res['models'][m]['ic']} "
                            f"仓位回测: 收益={bt['total_return']*100:.1f}% 夏普={bt['sharpe']} 回撤={bt['max_drawdown']*100:.1f}%")
            except Exception as e:
                logger.warning(f"[{pkey}][{m}] 失败: {str(e)[:80]}")

        results.setdefault("months", []).append(month_res)
        save_results(results)  # 铁律四

    # ── 汇总 ──
    oks = [m for m in results.get("months", []) if m.get("status") == "ok"]
    if oks:
        logger.info("=" * 60)
        logger.info(f"有效期数: {len(oks)}")
        for m in MODELS:
            ics = [x["models"].get(m, {}).get("ic") for x in oks]
            ics = [x for x in ics if x is not None]
            rets = [x["models"].get(m, {}).get("total_return") for x in oks]
            rets = [x for x in rets if x is not None]
            if ics:
                logger.info(f"  {m:<5} 期均IC={np.mean(ics):+.4f} IC>0={(np.array(ics)>0).mean()*100:.0f}% "
                            f"平均期收益={np.mean(rets)*100:.1f}%")
    # 持有不动基准
    bh = backtest_position(pd.Series(1.0, index=df.index[df.index >= pd.Timestamp(FIRST_TEST)]), idx)
    logger.info(f"  持有不动基准(000985): 收益={bh['total_return']*100:.1f}% 夏普={bh['sharpe']} 回撤={bh['max_drawdown']*100:.1f}%")
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


def load_existing() -> dict:
    if RESULT_FILE.exists():
        return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
