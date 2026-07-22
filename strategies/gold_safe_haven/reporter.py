"""
报告生成模块

基于 momentum_rotation 的 Reporter，适配黄金避险策略。
"""
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

from .config import ETF_POOL, OUTPUT_DIR
from .engine import DailyRecord, TradeRecord
from .metrics import BacktestMetrics


class Reporter:
    """回测报告生成器（适配黄金避险轮动策略）。"""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self._setup_matplotlib()
        os.makedirs(self.output_dir, exist_ok=True)

    def save_daily_records(self, daily_df: pd.DataFrame,
                           filename: str = "daily_records.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        daily_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  📄 净值日报表 → {path}")
        return path

    def save_trade_records(self, trade_df: pd.DataFrame,
                           filename: str = "trade_records.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        if not trade_df.empty:
            trade_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  📄 交易明细 → {path}")
        return path

    def save_metrics(self, metrics: BacktestMetrics,
                     filename: str = "metrics.csv") -> str:
        path = os.path.join(self.output_dir, filename)
        data = metrics.to_dict()
        rows = [{"指标": k, "数值": v} for k, v in data.items()]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  📄 绩效指标 → {path}")
        return path

    def plot_equity_curve(
        self, daily_df: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None,
        ew_data: Optional[pd.DataFrame] = None,
        filename: str = "equity_curve.png",
    ):
        """绘制净值曲线对比图。"""
        fig, ax = plt.subplots(figsize=(16, 7))
        dates = pd.to_datetime(daily_df["date"])
        strategy_net = daily_df["total_value"] / daily_df["total_value"].iloc[0]

        ax.plot(dates, strategy_net, label="黄金避险策略",
                color="#DAA520", linewidth=2, zorder=5)

        if benchmark_data is not None and not benchmark_data.empty:
            bench = benchmark_data["cumulative_returns"].values
            # 对齐长度
            min_len = min(len(dates), len(bench))
            ax.plot(dates[:min_len], bench[:min_len], label="沪深300",
                    color="#A23B72", linewidth=1.5, linestyle="--", alpha=0.8)

        if ew_data is not None and not ew_data.empty:
            ew_vals = ew_data["cumulative_returns"].values
            min_len = min(len(dates), len(ew_vals))
            ax.plot(dates[:min_len], ew_vals[:min_len], label="等权组合",
                    color="#F18F01", linewidth=1.5, linestyle="-.", alpha=0.8)

        cumulative = strategy_net.values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        dd_min = drawdown.min()
        if dd_min < -0.01:
            dd_min_idx = np.argmin(drawdown)
            peak_idx = np.argmax(cumulative[:dd_min_idx + 1])
            ax.axvspan(dates[peak_idx], dates[dd_min_idx],
                       color="gold", alpha=0.15, label="最大回撤区间")

        ax.set_title("黄金避险轮动策略 - 净值曲线", fontsize=14, fontweight="bold")
        ax.set_xlabel("交易日期")
        ax.set_ylabel("净值")
        ax.axhline(y=1.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=11, loc="upper left")
        ax.set_xlim(dates.iloc[0], dates.iloc[-1])
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 净值曲线图 → {path}")

    def plot_drawdown(self, daily_df: pd.DataFrame,
                      filename: str = "drawdown.png"):
        """绘制回撤曲线图。"""
        fig, ax = plt.subplots(figsize=(16, 4))
        dates = pd.to_datetime(daily_df["date"])
        cumulative = (1 + daily_df["cumulative_return"]).values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100

        ax.fill_between(dates, drawdown, 0, color="#E74C3C", alpha=0.3)
        ax.plot(dates, drawdown, color="#E74C3C", linewidth=1)

        dd_min = drawdown.min()
        dd_min_idx = np.argmin(drawdown)
        ax.annotate(f"最大回撤 {dd_min:.2f}%",
                    xy=(dates[dd_min_idx], dd_min),
                    xytext=(dates[dd_min_idx], dd_min * 1.3),
                    arrowprops=dict(arrowstyle="->", color="darkred"),
                    fontsize=10, color="darkred", fontweight="bold")

        ax.set_title("策略回撤曲线", fontsize=13, fontweight="bold")
        ax.set_xlabel("交易日期")
        ax.set_ylabel("回撤 (%)")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_xlim(dates.iloc[0], dates.iloc[-1])
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 回撤曲线图 → {path}")

    def print_summary(self, metrics: BacktestMetrics):
        """控制台打印绩效摘要。"""
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  📊 黄金避险轮动策略 — 回测绩效摘要")
        print(sep)

        sections = [
            ("收益指标", ["累计收益率", "年化收益率", "沪深300收益",
                          "等权组合收益", "超额(沪深300)", "超额(等权)"]),
            ("风险指标", ["最大回撤", "回撤持续天数", "年化波动率", "下行波动率"]),
            ("风险调整收益", ["夏普比率", "Sortino比率", "Calmar比率"]),
            ("交易统计", ["总交易笔数", "调仓切换次数", "平均持仓天数",
                          "日胜率", "交易胜率", "盈亏比", "交易总成本", "成本占比"]),
            ("资金信息", ["初始资金", "最终资金"]),
        ]

        data = metrics.to_dict()
        for section_name, keys in sections:
            print(f"\n  [{section_name}]")
            for key in keys:
                if key in data:
                    print(f"    {key:14s}: {data[key]}")
        print(f"\n{sep}")

    def _setup_matplotlib(self):
        """配置中文字体。"""
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ]
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    fm.fontManager.addfont(font_path)
                    plt.rcParams["font.family"] = "Noto Sans CJK JP"
                    break
                except Exception:
                    continue
        else:
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei",
                                                 "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
