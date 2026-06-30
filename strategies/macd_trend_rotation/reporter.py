"""报告生成模块"""
import os
from typing import Optional
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from .config import ETF_POOL, OUTPUT_DIR
from .metrics import BacktestMetrics


class Reporter:
    def __init__(self, od=OUTPUT_DIR):
        self.output_dir = od; self._setup(); os.makedirs(od, exist_ok=True)

    def save_csv(self, df, fn): p = os.path.join(self.output_dir, fn); df.to_csv(p, index=False, encoding="utf-8-sig"); print(f"  📄 {p}"); return p
    def save_daily(self, df, fn="daily_records.csv"): return self.save_csv(df, fn)
    def save_trades(self, df, fn="trade_records.csv"): return self.save_csv(df, fn) if not df.empty else None
    def save_metrics(self, m, fn="metrics.csv"):
        rows = [{"指标": k, "数值": v} for k, v in m.to_dict().items()]; return self.save_csv(pd.DataFrame(rows), fn)

    def plot_equity(self, df, bd=None, ed=None, fn="equity_curve.png"):
        fig, ax = plt.subplots(figsize=(16, 7)); d = pd.to_datetime(df["date"])
        sn = df["total_value"] / df["total_value"].iloc[0]
        ax.plot(d, sn, label="MACD趋势策略", color="#2E86AB", lw=2, zorder=5)
        if bd is not None and not bd.empty: ax.plot(d, bd["cumulative_returns"].values[:len(sn)], label="沪深300", color="#A23B72", lw=1.5, ls="--", alpha=0.8)
        if ed is not None and not ed.empty: ax.plot(d, ed["cumulative_returns"].values[:len(sn)], label="等权组合", color="#F18F01", lw=1.5, ls="-.", alpha=0.8)
        cv = sn.values; rm = np.maximum.accumulate(cv); dd = (cv - rm) / rm
        if dd.min() < -0.01:
            mi = np.argmin(dd); pi = np.argmax(cv[:mi + 1]); ax.axvspan(d[pi], d[mi], color="lightblue", alpha=0.15)
        ax.set_title("MACD趋势策略 - 净值曲线", fontsize=14, fontweight="bold")
        ax.set_xlabel("交易日期"); ax.set_ylabel("净值")
        ax.axhline(y=1, color="gray", lw=0.5, alpha=0.3); ax.grid(axis="y", ls="--", alpha=0.4)
        ax.legend(fontsize=11, loc="upper left"); ax.set_xlim(d.iloc[0], d.iloc[-1])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        plt.tight_layout(); fig.savefig(os.path.join(self.output_dir, fn), dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  📊 净值曲线图 → {self.output_dir}/{fn}")

    def plot_drawdown(self, df, fn="drawdown.png"):
        fig, ax = plt.subplots(figsize=(16, 4)); d = pd.to_datetime(df["date"])
        cv = (1 + df["cumulative_return"]).values; rm = np.maximum.accumulate(cv); dd = (cv - rm) / rm * 100
        ax.fill_between(d, dd, 0, color="#E74C3C", alpha=0.3); ax.plot(d, dd, color="#E74C3C", lw=1)
        mi, mii = dd.min(), np.argmin(dd)
        ax.annotate(f"最大回撤 {mi:.2f}%", xy=(d[mii], mi), xytext=(d[mii], mi * 1.3), arrowprops=dict(arrowstyle="->", color="darkred"), fontsize=10, color="darkred", fontweight="bold")
        ax.set_title("策略回撤曲线", fontsize=13, fontweight="bold"); ax.set_xlabel("交易日期"); ax.set_ylabel("回撤(%)")
        ax.axhline(y=0, color="gray", lw=0.5); ax.grid(axis="y", ls="--", alpha=0.4); ax.set_xlim(d.iloc[0], d.iloc[-1])
        plt.tight_layout(); fig.savefig(os.path.join(self.output_dir, fn), dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  📊 回撤曲线图 → {self.output_dir}/{fn}")

    def plot_heatmap(self, df, fn="holding_heatmap.png"):
        if df.empty: return
        df = df.copy(); df["date"] = pd.to_datetime(df["date"]); df["ym"] = df["date"].dt.strftime("%Y-%m")
        syms = list(ETF_POOL.keys()); sc = {s: i + 1 for i, s in enumerate(syms)}; sn = ETF_POOL
        df["hc"] = df["hold_symbol"].map(lambda x: sc.get(x, 0) if x else 0)
        mn = df.groupby("ym").last().reset_index()
        fig, ax = plt.subplots(figsize=(max(14, len(mn) * 0.4), 6))
        cd = mn["hc"].values.reshape(1, -1); ms = mn["ym"].values
        cm = plt.cm.Set2
        if len(syms) + 1 > cm.N:
            import matplotlib.colors as mc
            b = cm(np.arange(cm.N)); r = (len(syms) + 1 + cm.N - 1) // cm.N
            cm = mc.ListedColormap(np.tile(b, (r, 1))[:len(syms) + 1])
        bd = np.arange(-0.5, len(syms) + 1.5, 1); nrm = plt.matplotlib.colors.BoundaryNorm(bd, cm.N)
        im = ax.imshow(cd, aspect="auto", cmap=cm, norm=nrm, interpolation="nearest")
        plt.colorbar(im, ax=ax, ticks=list(range(len(syms) + 1)), shrink=0.6).set_ticklabels(["空仓"] + [sn[s] for s in syms])
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=45, ha="right", fontsize=8); ax.set_yticks([])
        ax.set_title("持仓分布热力图（月度末）", fontsize=13, fontweight="bold"); plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, fn), dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  📊 持仓热力图 → {self.output_dir}/{fn}")

    def plot_monthly(self, df, fn="monthly_returns.png"):
        df = df.copy(); df["date"] = pd.to_datetime(df["date"]); df["y"] = df["date"].dt.year; df["m"] = df["date"].dt.month
        pv = df.pivot_table(index="y", columns="m", values="cumulative_return", aggfunc="last")
        pm = pv.pct_change(axis=1).fillna(0); pm[1] = pv[1]
        fig, ax = plt.subplots(figsize=(12, max(4, len(pm) * 0.8)))
        cm = plt.cm.RdYlGn; im = ax.imshow(pm.values * 100, aspect="auto", cmap=cm, vmin=-5, vmax=5)
        for i in range(len(pm)):
            for j in range(len(pm.columns)):
                v = pm.values[i, j] * 100; ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9, color="white" if abs(v) > 3 else "black")
        ax.set_xticks(range(len(pm.columns))); ax.set_xticklabels([f"{int(m)}月" for m in pm.columns])
        ax.set_yticks(range(len(pm))); ax.set_yticklabels([f"{int(y)}年" for y in pm.index])
        ax.set_title("月度收益率热力图(%)", fontsize=13, fontweight="bold"); plt.colorbar(im, ax=ax, shrink=0.6)
        plt.tight_layout(); fig.savefig(os.path.join(self.output_dir, fn), dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  📊 月度收益热力图 → {self.output_dir}/{fn}")

    def print_summary(self, m):
        s = "=" * 55; print(f"\n{s}"); print(f"  📊 MACD趋势策略 — 回测绩效摘要"); print(s)
        secs = [("收益指标", ["累计收益率","年化收益率","沪深300收益","等权组合收益","超额(沪深300)","超额(等权)"]),
                ("风险指标", ["最大回撤","回撤持续天数","年化波动率","下行波动率"]),
                ("风险调整收益", ["夏普比率","Sortino比率","Calmar比率"]),
                ("交易统计", ["总交易笔数","调仓切换次数","平均持仓天数","日胜率","交易胜率","盈亏比","交易总成本","成本占比"]),
                ("资金信息", ["初始资金","最终资金"])]
        d = m.to_dict()
        for sn, ks in secs:
            print(f"\n  [{sn}]")
            for k in ks:
                if k in d: print(f"    {k:14s}: {d[k]}")
        print(f"\n{s}")

    def _setup(self):
        for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(fp):
                try: fm.fontManager.addfont(fp); plt.rcParams["font.family"] = "Noto Sans CJK JP"; break
                except: continue
        else: plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei"]; plt.rcParams["axes.unicode_minus"] = False
