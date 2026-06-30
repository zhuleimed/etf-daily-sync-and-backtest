"""报告生成模块"""
import os
from typing import Optional
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from .config import OUTPUT_DIR
from .metrics import BacktestMetrics

STRATEGY_DISPLAY_NAME = "均值回归轮动"


class Reporter:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir
        self._setup_matplotlib()
        os.makedirs(self.output_dir, exist_ok=True)

    def save_daily_records(self, daily_df, filename="daily_records.csv"):
        path = os.path.join(self.output_dir, filename)
        daily_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  📄 净值日报表 → {path}")
        return path

    def save_trade_records(self, trade_df, filename="trade_records.csv"):
        path = os.path.join(self.output_dir, filename)
        if not trade_df.empty:
            trade_df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  📄 交易明细 → {path}")
        return path

    def save_metrics(self, metrics, filename="metrics.csv"):
        path = os.path.join(self.output_dir, filename)
        data = metrics.to_dict()
        rows = [{"指标": k, "数值": v} for k, v in data.items()]
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  📄 绩效指标 → {path}")
        return path

    def plot_equity_curve(self, daily_df, benchmark_data=None, ew_data=None, filename="equity_curve.png"):
        fig, ax = plt.subplots(figsize=(16, 7))
        dates = pd.to_datetime(daily_df["date"])
        strategy_net = daily_df["total_value"] / daily_df["total_value"].iloc[0]
        ax.plot(dates, strategy_net, label=STRATEGY_DISPLAY_NAME, color="#27AE60", linewidth=2, zorder=5)
        if benchmark_data is not None and not benchmark_data.empty:
            bench = benchmark_data["cumulative_returns"].values[:len(strategy_net)]
            ax.plot(dates, bench, label="沪深300", color="#A23B72", linewidth=1.5, linestyle="--", alpha=0.8)
        if ew_data is not None and not ew_data.empty:
            ew_vals = ew_data["cumulative_returns"].values[:len(strategy_net)]
            ax.plot(dates, ew_vals, label="等权组合", color="#F18F01", linewidth=1.5, linestyle="-.", alpha=0.8)
        ax.set_title(f"{STRATEGY_DISPLAY_NAME} - 净值曲线", fontsize=14, fontweight="bold")
        ax.axhline(y=1.0, color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=11, loc="upper left")
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 净值曲线图 → {os.path.join(self.output_dir, filename)}")

    def plot_drawdown(self, daily_df, filename="drawdown.png"):
        fig, ax = plt.subplots(figsize=(16, 4))
        dates = pd.to_datetime(daily_df["date"])
        cumulative = (1 + daily_df["cumulative_return"]).values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max * 100
        ax.fill_between(dates, drawdown, 0, color="#E74C3C", alpha=0.3)
        ax.plot(dates, drawdown, color="#E74C3C", linewidth=1)
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 回撤曲线图 → {os.path.join(self.output_dir, filename)}")

    def print_summary(self, metrics: BacktestMetrics):
        sep = "=" * 55
        print(f"\n{sep}")
        print(f"  📊 {STRATEGY_DISPLAY_NAME} — 回测绩效摘要")
        print(sep)
        sections = [
            ("收益指标", ["累计收益率", "年化收益率", "沪深300收益", "超额(沪深300)"]),
            ("风险指标", ["最大回撤", "年化波动率"]),
            ("风险调整收益", ["夏普比率"]),
            ("交易统计", ["调仓切换次数", "日胜率"]),
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
            plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei"]
            plt.rcParams["axes.unicode_minus"] = False
