"""LSTM ETF 轮动 — 数据集构建（月度 Walk-Forward）

采样设计（ETF 样本少的应对）：
  - 周频采样：每 5 个交易日取一个样本（48 只 × 52 周/年 ≈ 2500 样本/年）
  - purge：训练集与测试集之间留 21 个交易日（1 个月），防止标签重叠泄漏
  - 标签 y = 未来 21 个交易日收益（下月收益）
  - 归一化：沿时间轴 per-feature Z-score（T4 方式，与标的数量无关）
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_dataset(
    etf_data: dict[str, pd.DataFrame],
    train_start: str,
    train_end: str,
    test_ref_date: str,
    window: int = 60,
    sample_stride: int = 5,
    purge_gap: int = 21,
    min_samples_per_etf: int = 30,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建月度 WF 的训练/测试序列数据集。

    Args:
        etf_data: {symbol: df}，df 含 date 和特征列（升序）。
        train_start/train_end: 训练窗口（含端点，如 '2023-01-01'/'2023-12-31'）。
        test_ref_date: 测试参考日（该日计算特征 → 预测下月收益）。
        window: LSTM 序列窗口（交易日）。
        sample_stride: 采样步长（天），5 = 周频采样。
        purge_gap: 训练集最后样本与测试参考日的最小间隔（交易日）。
        min_samples_per_etf: 每只 ETF 最少样本数，不足则剔除。

    Returns:
        (X_train, y_train, X_test, y_test, train_dates, test_dates)
        y = 未来 21 交易日收益。测试集为 ref_date 当日各 ETF 的一个样本。
    """
    if feature_cols is None:
        raise ValueError("feature_cols 必传")

    # ── 未来收益标签（每只 ETF） ──
    fwd = {}
    for sym, df in etf_data.items():
        close = df["close"]
        fwd[sym] = close.shift(-21) / close - 1  # 未来 21 日收益
        fwd[sym].index = df["date"]

    # ── 训练样本 ──
    X_train, y_train, t_train = [], [], []
    for sym, df in etf_data.items():
        d = df.set_index("date")
        mask = (d.index >= train_start) & (d.index <= train_end)
        sub = d[mask]
        if len(sub) < window + 21:
            continue
        f = fwd[sym]
        # 周频采样（从窗口后开始）
        for i in range(window, len(sub) - 21, sample_stride):
            feat = sub.iloc[i - window + 1: i + 1][feature_cols].values
            if np.isnan(feat).any():
                continue
            y_val = f.reindex([sub.index[i]])  # 标签：样本日当天的未来收益
            y_val = y_val.iloc[0]
            if np.isnan(y_val):
                continue
            X_train.append(feat)
            y_train.append(y_val)
            t_train.append(sub.index[i])

    # ── purge：训练样本与测试参考日间隔 >= purge_gap ──
    X_test, y_test, t_test = [], [], []
    ref = pd.Timestamp(test_ref_date)
    if t_train:
        t_arr = pd.DatetimeIndex(t_train)
        keep = t_arr <= (ref - pd.Timedelta(days=purge_gap * 1.5 + 5))
        # 用自然日近似（purge_gap 交易日 ≈ 1.5× 自然日）
        X_train = [x for x, k in zip(X_train, keep) if k]
        y_train = [y for y, k in zip(y_train, keep) if k]
        t_train = [t for t, k in zip(t_train, keep) if k]

    # ── 测试样本（ref_date 当日各 ETF） ──
    for sym, df in etf_data.items():
        d = df.set_index("date")
        if ref not in d.index:
            continue
        if d.index.get_loc(ref) < window:
            continue
        feat = d.iloc[d.index.get_loc(ref) - window + 1: d.index.get_loc(ref) + 1][feature_cols].values
        if np.isnan(feat).any():
            continue
        y_val = fwd[sym].reindex([ref]).iloc[0]
        if np.isnan(y_val):
            continue
        X_test.append(feat)
        y_test.append(y_val)
        t_test.append(sym)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)

    # ── 沿时间轴 per-feature Z-score（训练集统计量应用到测试集） ──
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test, t_test

