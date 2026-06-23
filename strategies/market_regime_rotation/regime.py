"""
市场状态检测模块

提供两层架构：
  1. compute_regime() — 底层"裸信号"，无状态，仅凭当日数据判断 BULL/BEAR
  2. RegimeDetector — 有限状态机，在裸信号基础上叠加：
     - 迟滞带（入口/出口不对称阈值）
     - 确认窗口（连续 N 日同向才切换）

使用方法：

    detector = RegimeDetector(ma_period=60, mom_window=20, ...)
    for idx in range(n):
        regime = detector.update(benchmark_data, bm_idx)
        # regime 是经过迟滞和确认平滑后的稳定输出
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_regime(
    benchmark_data: pd.DataFrame,
    idx: int,
    ma_period: int = 60,
    mom_window: int = 20,
    mom_threshold: float = 0.03,
) -> str:
    """
    裸信号检测（无状态版本）—— 仅用于对比/调试。

    若需工程使用，请用 RegimeDetector 类（带迟滞和确认窗口）。
    """
    if benchmark_data.empty or idx >= len(benchmark_data):
        return "BEAR"

    close = benchmark_data.iloc[idx]["close"]
    start = max(0, idx - ma_period + 1)
    ma_long = benchmark_data.iloc[start:idx + 1]["close"].mean()
    if idx >= mom_window:
        mom = close / benchmark_data.iloc[idx - mom_window]["close"] - 1
    else:
        mom = 0.0

    if close > ma_long and mom > mom_threshold:
        return "BULL"
    return "BEAR"


def _raw_signal(
    benchmark_data: pd.DataFrame,
    idx: int,
    ma_period: int,
    mom_window: int,
    mom_threshold: float,
    current_regime: str,
    bull_entry_buffer: float = 0.0,
    bull_exit_buffer: float = 0.0,
) -> str:
    """
    带迟滞的裸信号：
    - BEAR→BULL：价格需高于 均线×(1+entry_buffer) 且动量达阈值
    - BULL→BEAR：价格需低于 均线×(1-exit_buffer)（动量不作为退出条件）
    """
    close = benchmark_data.iloc[idx]["close"]
    start = max(0, idx - ma_period + 1)
    ma = benchmark_data.iloc[start:idx + 1]["close"].mean()
    if idx >= mom_window:
        mom = close / benchmark_data.iloc[idx - mom_window]["close"] - 1
    else:
        mom = 0.0

    if current_regime == "BEAR":
        # 进入 BULL —— 高门槛：价格超标 + 动量达标
        if close > ma * (1 + bull_entry_buffer) and mom > mom_threshold:
            return "BULL"
        else:
            return "BEAR"
    else:  # BULL
        # 退出 BULL —— 低门槛：仅需价格深度跌破均线
        if close < ma * (1 - bull_exit_buffer):
            return "BEAR"
        else:
            return "BULL"


class RegimeDetector:
    """
    带迟滞和确认窗口的市场状态检测器（有限状态机）。

    特点：
    - 进入 BULL 更难（均线 + 动量 + 确认天数），退出 BULL 更容易维持
    - 连续 N 日相反信号才切换，避免单日噪声
    - 记录当前状态，可供外部查询

    用法：
        detector = RegimeDetector(...)
        for idx in range(...):
            regime = detector.update(benchmark_data, bm_idx)
    """

    def __init__(
        self,
        ma_period: int = 60,
        mom_window: int = 20,
        mom_threshold: float = 0.03,
        confirm_days: int = 5,
        bull_entry_buffer: float = 0.02,
        bull_exit_buffer: float = 0.01,
    ):
        self.ma_period = ma_period
        self.mom_window = mom_window
        self.mom_threshold = mom_threshold
        self.confirm_days = confirm_days
        self.bull_entry_buffer = bull_entry_buffer
        self.bull_exit_buffer = bull_exit_buffer

        # 内部状态
        self.current_regime: str = "BEAR"          # 当前已确认的状态
        self._opposite_count: int = 0              # 连续出现相反信号的天数
        self._total_bull_days: int = 0             # 本段 BULL 累计天数（统计用）
        self._total_bear_days: int = 0             # 本段 BEAR 累计天数

    @property
    def regime(self) -> str:
        """当前已确认的市场状态。"""
        return self.current_regime

    def reset(self, regime: str = "BEAR"):
        """重置检测器状态（用于重新开始或手动指定初始状态）。"""
        self.current_regime = regime
        self._opposite_count = 0
        self._total_bull_days = 0
        self._total_bear_days = 0

    def update(
        self,
        benchmark_data: pd.DataFrame,
        idx: int,
    ) -> str:
        """
        更新检测器并返回当前确认后的市场状态。

        Parameters
        ----------
        benchmark_data : pd.DataFrame
            基准指数数据，需含 close 列
        idx : int
            当前日期在 benchmark_data 中的位置索引

        Returns
        -------
        str : "BULL" 或 "BEAR"
        """
        if benchmark_data.empty or idx >= len(benchmark_data):
            return self.current_regime

        # 1. 提取当日原始信号（含迟滞）
        raw = _raw_signal(
            benchmark_data, idx,
            ma_period=self.ma_period,
            mom_window=self.mom_window,
            mom_threshold=self.mom_threshold,
            current_regime=self.current_regime,
            bull_entry_buffer=self.bull_entry_buffer,
            bull_exit_buffer=self.bull_exit_buffer,
        )

        # 2. 状态更新
        if raw == self.current_regime:
            # 信号与当前状态一致 → 重置计数
            self._opposite_count = 0
        else:
            # 信号相反 → 计数 +1
            self._opposite_count += 1
            if self._opposite_count >= self.confirm_days:
                # 确认切换
                self.current_regime = raw
                self._opposite_count = 0

        # 3. 累计统计
        if self.current_regime == "BULL":
            self._total_bull_days += 1
        else:
            self._total_bear_days += 1

        return self.current_regime
