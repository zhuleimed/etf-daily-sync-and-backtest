"""LSTM ETF 轮动 — 特征工程（量价特征，无估值）

参照 sequoia-x V2 的 features.py 思路，但精简为 ETF 场景核心量价特征：
- 全部特征只用 T-1 及更早数据计算（严格 T+1，禁止 look-ahead）
- 特征统一命名为 f_ 前缀，共 21 维
- 归一化在 dataset.py 中沿时间轴 Z-score（per-stock，T4 方式）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算单只 ETF 的量价特征。

    Args:
        df: 含 date/open/high/low/close/volume 的日线 DataFrame（升序）。

    Returns:
        df 副本 + 特征列。特征在当日收盘后可计算，但信号只用前一日
        （调用方 shift(1) 处理，避免 look-ahead）。
    """
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]

    # ── 收益率类 ──
    d["f_ret_1"] = close.pct_change(1)
    d["f_ret_5"] = close.pct_change(5)
    d["f_ret_10"] = close.pct_change(10)
    d["f_ret_20"] = close.pct_change(20)
    d["f_mom_60"] = close.pct_change(60)          # 60 日动量

    # ── 波动率类 ──
    d["f_vol_5"] = d["f_ret_1"].rolling(5).std()
    d["f_vol_10"] = d["f_ret_1"].rolling(10).std()
    d["f_vol_20"] = d["f_ret_1"].rolling(20).std()

    # ── 量价类 ──
    d["f_vol_ratio"] = volume.rolling(5).mean() / volume.rolling(20).mean()  # 量比
    amt = close * volume
    d["f_amt_ratio"] = amt.rolling(5).mean() / amt.rolling(20).mean()        # 额比

    # ── 技术指标类 ──
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["f_rsi_14"] = 100 - 100 / (1 + rs)

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    d["f_macd_dif"] = dif
    d["f_macd_hist"] = dif - dea

    # 布林带位置（20, 2σ）
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    d["f_boll_pos"] = (close - ma20) / (2 * std20 + 1e-9)

    # ── 趋势/回撤类 ──
    d["f_ma20_dev"] = close / ma20 - 1                 # 偏离 20 日线
    ma60 = close.rolling(60).mean()
    d["f_ma60_dev"] = close / ma60 - 1                 # 偏离 60 日线
    d["f_dd_20"] = close / close.rolling(20).max() - 1  # 距 20 日高点回撤
    d["f_dd_252"] = close / close.rolling(252).max() - 1  # 距 1 年高点回撤

    return d


def compute_market_features(etf_data: dict, benchmark: pd.DataFrame) -> pd.DataFrame:
    """计算相对基准（沪深300）的市场状态特征（每股追加）。

    Args:
        etf_data: {symbol: df}（已含 compute_features 后的特征）。
        benchmark: 基准指数日线（date/close，升序）。

    Returns:
        在原 df 上追加 f_beta_60 / f_excess_20 列。
    """
    bench = benchmark.set_index("date")["close"].pct_change(1).rename("bench_ret")
    bench_ret20 = bench.rolling(20).sum()  # 基准 20 日累计收益

    for sym, df in etf_data.items():
        # 按日期 merge 对齐（concat 会因索引不同而全 NaN）
        tmp = df[["date", "f_ret_1"]].merge(
            bench.rename("bench_ret").reset_index(), on="date", how="left"
        )
        # 60 日滚动 Beta（协方差/方差）
        beta = (
            tmp["f_ret_1"].rolling(60).cov(tmp["bench_ret"])
            / tmp["bench_ret"].rolling(60).var()
        )
        df["f_beta_60"] = beta.values
        # 20 日超额收益 = ETF 20日收益 - 基准 20日收益
        excess = df[["date", "f_ret_20"]].merge(
            bench_ret20.rename("b20").reset_index(), on="date", how="left"
        )
        df["f_excess_20"] = (excess["f_ret_20"] - excess["b20"]).values

    return etf_data


FEATURE_COLS = [
    "f_ret_1", "f_ret_5", "f_ret_10", "f_ret_20", "f_mom_60",
    "f_vol_5", "f_vol_10", "f_vol_20",
    "f_vol_ratio", "f_amt_ratio",
    "f_rsi_14", "f_macd_dif", "f_macd_hist", "f_boll_pos",
    "f_ma20_dev", "f_ma60_dev", "f_dd_20", "f_dd_252",
    "f_beta_60", "f_excess_20",
]  # 20 维
