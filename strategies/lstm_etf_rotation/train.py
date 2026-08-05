"""LSTM ETF 轮动 — 模型训练（小 LSTM，POC 版）

铁律一：模块级清除 KMP_AFFINITY（.bashrc 绑核会锁死多进程）
铁律五：启动打印线程配置 + 训练后自检（预测 std > 1e-7）
"""
from __future__ import annotations

import os

# ── 铁律一：清除 KMP_AFFINITY（必须在 import tensorflow 之前） ──
os.environ.pop("KMP_AFFINITY", None)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)


def build_lstm(window: int, n_features: int, units: int = 64) -> tf.keras.Model:
    """构建小型 LSTM 回归模型（ETF 样本少，64 单元而非 sequoia-x 的 128）。"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, n_features)),
        tf.keras.layers.LSTM(units, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(units // 2, dropout=0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    window: int,
    n_features: int,
    units: int = 64,
    epochs: int = 60,
    batch_size: int = 64,
    patience: int = 8,
    validation_split: float = 0.15,
    verbose: int = 0,
) -> tuple[tf.keras.Model, dict]:
    """训练 LSTM 并返回 (模型, 诊断信息)。

    自检（铁律五）：训练后验证预测 std > 1e-7（防常数预测）。
    """
    model = build_lstm(window, n_features, units=units)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )

    # 自检：预测方差
    pred = model.predict(X_train, verbose=0).ravel()
    std_pred = float(np.std(pred))
    if std_pred < 1e-7:
        raise RuntimeError(f"训练自检失败：预测 std={std_pred:.2e} < 1e-7（常数预测）")

    n_epochs = len(history.history["loss"])
    return model, {
        "final_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "n_epochs": n_epochs,
        "pred_std": std_pred,
        "n_samples": int(len(X_train)),
    }
