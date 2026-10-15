#!/usr/bin/env python3
"""
回测"指数位置错位"（B018 类）影响面排查（2026-10-16 新增）

背景
----
`judge_market_regime(index_data, idx)` 内部用 `iloc[idx]` 取指数行，而调用方传的 idx
是 **ETF 数组位置**；指数加载起点比 ETF 早（各自 data.py 里会往前多取 150~200 天）
→ 两边位置不对齐时，regime 实际用的是"几个月前的行情"。
adx_trend_rotation 已于 2026-10-02 修（load_data 里按日期 reindex 到 ETF 日历）；
composite_momentum 于 2026-10-16 实测为**外观问题**（regime 列错 61% 的天，持仓/收益一天不差）。

本脚本：对指定策略跑两遍全周期回测——① 现状（错位）② 按日期对齐指数——逐日比对
持仓 / regime / 累计收益，量化"错位到底改不改变交易结果"，避免凭猜决定修不修。

用法
----
    python -m simulation.analysis.audit_regime_alignment                    # 默认三个待排查策略
    python -m simulation.analysis.audit_regime_alignment --strategies rsi_trend_rotation
    python -m simulation.analysis.audit_regime_alignment --start 2024-01-01 --end 2026-10-15
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.composite_momentum.config import DB_PATH  # 所有策略共用同一个库

# 默认排查对象：同一写法、影响面未评估的策略（adx/composite 已单独处理）
DEFAULT_STRATEGIES = ["rsi_trend_rotation", "macd_trend_rotation", "mean_reversion_rotation"]


def run_once(strategy: str, aligned: bool, start: str, end: str) -> pd.DataFrame:
    """跑一遍全周期回测，返回逐日记录 DataFrame。

    aligned=False：用策略现状（指数按自己的起点加载 → 与 ETF 位置错位）
    aligned=True ：把指数按日期 reindex 到 ETF 交易日历后再喂给引擎（与 adx 的修法一致）
    """
    mod = importlib.import_module(f"strategies.{strategy}.engine")
    cls = mod.BacktestEngine

    if aligned:
        orig_load = cls.load_data

        def patched(self, *args, **kwargs):
            ret = orig_load(self, *args, **kwargs)
            if not self.index_data.empty and len(self.dates):
                self.index_data = (
                    self.index_data.set_index("date")
                    .reindex(self.dates)      # ETF 交易日历
                    .ffill()
                    .rename_axis("date")
                    .reset_index()
                )
            return ret

        cls.load_data = patched
    else:
        cls.load_data = getattr(cls, "_orig_load_data", cls.load_data)

    try:
        engine = cls()
        engine.load_data(start_date=start, end_date=end, db_path=DB_PATH)
        engine.run()
    finally:
        if aligned:  # 复原，避免影响后续策略
            cls.load_data = orig_load

    df = pd.DataFrame([r.__dict__ if hasattr(r, "__dict__") else r for r in engine.daily_records])
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    return df


def compare(strategy: str, start: str, end: str) -> dict:
    print(f"\n═══ {strategy} ═══")
    old = run_once(strategy, aligned=False, start=start, end=end)
    new = run_once(strategy, aligned=True, start=start, end=end)

    cols = [c for c in ["date", "hold_symbol", "regime", "cumulative_return"] if c in old.columns]
    m = (old[cols].fillna({"hold_symbol": ""})
         .merge(new[cols].fillna({"hold_symbol": ""}), on="date", suffixes=("_旧", "_对齐")))
    hold_diff = m[m["hold_symbol_旧"] != m["hold_symbol_对齐"]]
    reg_diff = m[m["regime_旧"] != m["regime_对齐"]]
    ret_diff = float((m["cumulative_return_旧"] - m["cumulative_return_对齐"]).abs().max())

    print(f"  共同交易日 {len(m)} 天 | 持仓不同 {len(hold_diff)} 天 | regime 不同 {len(reg_diff)} 天"
          f" | 累计收益最大差 {ret_diff:.4%}")
    if len(reg_diff):
        print("  regime 差异分布（旧口径 → 对齐后）：")
        for (a, b), n in reg_diff.groupby(["regime_旧", "regime_对齐"]).size().items():
            print(f"    {a} → {b}: {n} 天")
    if len(hold_diff):
        print("  持仓不同的日子（前 15 个）：")
        print(hold_diff[["date", "hold_symbol_旧", "hold_symbol_对齐"]].head(15).to_string(index=False))
    else:
        print("  ✅ 持仓与收益完全一致 → 修 B018 只改 regime 列/基准曲线，不改交易结果")
    return {"strategy": strategy, "days": len(m), "hold_diff": len(hold_diff),
            "regime_diff": len(reg_diff), "ret_diff": ret_diff}


def main():
    p = argparse.ArgumentParser(description="回测指数位置错位(B018)影响面排查")
    p.add_argument("--strategies", type=str, default=",".join(DEFAULT_STRATEGIES),
                   help="逗号分隔的策略名（默认三个待排查策略）")
    p.add_argument("--start", type=str, default="2024-01-01")
    p.add_argument("--end", type=str, default="2026-10-15")
    args = p.parse_args()

    print(f"B018 影响面排查 | 区间 {args.start} ~ {args.end}")
    rows = []
    for s in [x.strip() for x in args.strategies.split(",") if x.strip()]:
        try:
            rows.append(compare(s, args.start, args.end))
        except Exception as e:  # 单个策略失败不影响其他
            print(f"  ❌ {s} 排查失败: {type(e).__name__}: {e}")
    print("\n═══ 汇总 ═══")
    for r in rows:
        verdict = "无影响（外观问题）" if r["hold_diff"] == 0 else f"**会改交易结果**（{r['hold_diff']}天）"
        print(f"  {r['strategy']:<26} 持仓差 {r['hold_diff']:>3}天 / regime差 {r['regime_diff']:>3}天 → {verdict}")


if __name__ == "__main__":
    main()
