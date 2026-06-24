#!/usr/bin/env python
"""
方案A（单对最强+全仓）深度分析

包括：
  1. 持仓时间分布（什么时候持有哪只ETF）
  2. 分年度表现
  3. 最大回撤区间分析
  4. 信号质量（胜率、盈亏比）
  5. 与基准（沪深300）对比
  6. 净值曲线 + 持仓标注可视化
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from .config import (
    PAIRS, INITIAL_CAPITAL, ZSCORE_OPEN, ZSCORE_CLOSE, ZSCORE_PERIOD,
    COMMISSION_RATE, SLIPPAGE, DB_PATH, OUTPUT_DIR,
)
from strategies.momentum_rotation.data import load_all_etf_data, load_benchmark_data
from .engine_switch import run_mode_a


ETF_NAMES = {
    "510050": "上证50",
    "510300": "沪深300",
    "159915": "创业板",
    "588000": "科创50",
}


def extract_hold_symbol(hold_str: str) -> str:
    """从 '510050(5400股)' 中提取代码"""
    if not hold_str or "现金" in hold_str:
        return "现金"
    sym = hold_str.split("(")[0]
    return sym.strip()


def analyze_mode_a(records, trades, etf_data, dates, initial_capital):
    """全面分析方案A。"""
    df = pd.DataFrame([{
        "date": r.date,
        "action": r.action,
        "hold_symbol": extract_hold_symbol(r.hold_symbols),
        "hold_raw": r.hold_symbols,
        "cash": r.cash,
        "stock_value": r.stock_value,
        "total_value": r.total_value,
        "daily_return": r.daily_return,
        "cumulative_return": r.cumulative_return,
        "z_scores": r.z_scores,
    } for r in records])

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["cumulative_return_pct"] = df["cumulative_return"] * 100

    total_days = len(df)
    final_value = df["total_value"].iloc[-1]
    total_return = final_value / initial_capital - 1

    # ── 1. 持仓时间分布 ──
    print("\n" + "=" * 65)
    print("  一、持仓时间分布")
    print("=" * 65)
    hold_counts = df[df["hold_symbol"] != "现金"]["hold_symbol"].value_counts()
    hold_days = hold_counts.sum()
    for sym, cnt in hold_counts.items():
        name = ETF_NAMES.get(sym, sym)
        pct = cnt / hold_days * 100
        print(f"  {name:<6} ({sym}): {cnt:>4}天 ({pct:>5.1f}%)")

    # 不同时期持仓变化
    print(f"\n  年持仓分布:")
    for year in sorted(df["year"].unique()):
        yd = df[df["year"] == year]
        yd_hold = yd[yd["hold_symbol"] != "现金"]
        yt = len(yd)
        yh = len(yd_hold)
        print(f"  {year}: 持仓{yh}/{yt}天 ({yh/yt*100:.0f}%)", end="")
        if len(yd_hold) > 0:
            dist = yd_hold["hold_symbol"].value_counts()
            parts = [f"{ETF_NAMES.get(s,s)} {c/len(yd_hold)*100:.0f}%" for s, c in dist.items()]
            print(f"  → {', '.join(parts)}")
        else:
            print()

    # ── 2. 分年度表现 ──
    print("\n" + "=" * 65)
    print("  二、分年度表现（vs 沪深300）")
    print("=" * 65)

    # 加载基准
    try:
        bm_df = load_benchmark_data(start_date=str(df["date"].min().date()),
                                    end_date=str(df["date"].max().date()))
        bm_df["date"] = pd.to_datetime(bm_df["date"])
    except:
        bm_df = None

    yearly = []
    for year in sorted(df["year"].unique()):
        yd = df[df["year"] == year]
        if yd.empty:
            continue
        year_start_val = yd["total_value"].iloc[0] if yearly else initial_capital
        # 找年初首日净值（若无前一年末数据则用初始资金）
        if yearly:
            prev_year = yearly[-1]
            last_day_prev = df[(df["year"] == year - 1) & (df["date"] == df[df["year"] == year-1]["date"].max())]
            if not last_day_prev.empty:
                year_start_val = last_day_prev["total_value"].iloc[0]

        year_end_val = yd["total_value"].iloc[-1]
        year_ret = year_end_val / year_start_val - 1
        year_dd = _max_drawdown(yd["total_value"].values)
        year_vol = yd["daily_return"].std() * np.sqrt(245) * 100

        # 基准
        bm_ret = None
        if bm_df is not None:
            bm_year = bm_df[bm_df["date"].dt.year == year]
            if not bm_year.empty:
                bm_start = bm_year["close"].iloc[0]
                bm_end = bm_year["close"].iloc[-1]
                bm_ret = bm_end / bm_start - 1

        bm_str = f"{bm_ret*100:+.2f}%" if bm_ret is not None else "—"
        yearly.append({"year": year, "return": year_ret, "dd": year_dd, "vol": year_vol, "bm": bm_ret})

        print(f"  {year}: 收益{year_ret*100:+7.2f}%  | 回撤{year_dd*100:5.2f}%  | "
              f"年化波动{year_vol:4.1f}%  | 基准{bm_str}")

    # ── 3. 最大回撤区间 ──
    print("\n" + "=" * 65)
    print("  三、最大回撤区间分析")
    print("=" * 65)

    values = df["total_value"].values
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak

    # 找回撤区间
    in_dd = False
    dd_periods = []
    dd_start = 0
    for i in range(len(dd)):
        if dd[i] > 0.01 and not in_dd:
            in_dd = True
            dd_start = i
        elif (dd[i] <= 0.01 or i == len(dd)-1) and in_dd:
            in_dd = False
            if i - dd_start >= 3:  # 至少3天
                dd_periods.append({
                    "start": df["date"].iloc[dd_start],
                    "end": df["date"].iloc[i],
                    "max_dd": dd[dd_start:i+1].max(),
                    "days": i - dd_start + 1,
                })

    dd_periods.sort(key=lambda x: -x["max_dd"])
    print(f"  前5大回撤区间:")
    print(f"  {'开始':<12} {'结束':<12} {'最大回撤':<10} {'持续天数':<8} {'持有':<12}")
    for dp in dd_periods[:5]:
        mid_idx = df[df["date"] == dp["start"]].index[0] if dp["start"] in df["date"].values else 0
        if isinstance(mid_idx, pd.Index) and len(mid_idx) > 0:
            mid_idx = mid_idx[0]
        hold_at_start = df.iloc[dp["start"].index if hasattr(dp["start"], 'index') else 0]
        # 直接在时间点附近找
        mask = df["date"] >= dp["start"]
        if mask.any():
            idx_start = df[mask].index[0]
            hold = df.iloc[idx_start]["hold_symbol"]
        else:
            hold = "—"
        print(f"  {dp['start'].strftime('%Y-%m-%d'):<12} "
              f"{dp['end'].strftime('%Y-%m-%d'):<12} "
              f"{dp['max_dd']*100:>8.2f}%  "
              f"{dp['days']:>4d}天    "
              f"{ETF_NAMES.get(hold, hold):<12}")

    # ── 4. 交易分析 ──
    print("\n" + "=" * 65)
    print("  四、交易信号分析")
    print("=" * 65)

    # 分析每笔交易
    trade_records = []
    entry_date = None
    entry_price = None
    entry_shares = None
    entry_symbol = None

    for t in trades:
        action = t.get("action", "")
        date = t.get("date", "")
        symbol = t.get("symbol", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0)

        if "买入" in action or ("买" in action and "切" not in action):
            if entry_symbol and entry_date:
                # 前一笔未平仓（切换场景）
                pass
            entry_date = date
            entry_price = price
            entry_shares = shares
            entry_symbol = symbol
        elif "平仓" in action or "止损" in action or "卖出" in action or "只卖" in action:
            if entry_symbol and entry_date:
                pnl = (price - entry_price) * shares * (-1 if "止损" in action else 1)
                # 实际上需要更精确的计算...
                trade_records.append({
                    "entry_date": entry_date, "exit_date": date,
                    "symbol": entry_symbol,
                    "action": action,
                })
                entry_symbol = None

    # 胜率统计
    holding_days = df[df["hold_symbol"] != "现金"].groupby(
        (df["hold_symbol"] != "现金") != (df["hold_symbol"].shift() != "现金")
    ).size()
    # 交易动作统计
    actions = df[df["action"] != "hold"]["action"].tolist()
    buy_actions = sum(1 for a in actions if "buy" in a or "switch" in a)
    sell_actions = sum(1 for a in actions if "close" in a or "stop" in a or "sell" in a)

    print(f"  总交易动作: {len(actions)} 次")
    print(f"  买入/切换: {buy_actions} 次")
    print(f"  卖出/平仓: {sell_actions} 次")
    print(f"  持仓周期数: {len([h for h in holding_days]) if not holding_days.empty else 0}")

    # 持仓周期长度
    if not holding_days.empty:
        hold_lengths = holding_days.values
        print(f"  平均持仓:  {hold_lengths.mean():.0f} 天")
        print(f"  最短持仓:  {hold_lengths.min():.0f} 天")
        print(f"  最长持仓:  {hold_lengths.max():.0f} 天")

    # ── 5. 信号质量 ──
    print("\n" + "=" * 65)
    print("  五、z-score 信号统计")
    print("=" * 65)

    # 从z_scores列解析
    z_data = {"上证50↔创业板": [], "沪深300↔创业板": [], "上证50↔科创50": []}
    for _, row in df.iterrows():
        zs = row.get("z_scores", "")
        if zs and zs != "无数据":
            parts = zs.split("; ")
            for p in parts:
                p = p.strip()
                for key in z_data:
                    if p.startswith(key):
                        try:
                            val = float(p.split("z=")[1])
                            z_data[key].append(val)
                        except:
                            pass

    for pair_name, vals in z_data.items():
        if vals:
            arr = np.array(vals)
            print(f"  {pair_name:<16} 均值{arr.mean():+.2f}  "
                  f"标准差{arr.std():.2f}  "
                  f"最大值{arr.max():+.2f}  "
                  f"最小值{arr.min():+.2f}  "
                  f"|z|>2占比{np.mean(np.abs(arr)>2)*100:.0f}%")

    # ── 6. 净值数据摘要 ──
    print("\n" + "=" * 65)
    print("  六、总体指标")
    print("=" * 65)

    total_return = final_value / initial_capital - 1
    n_days = len(df)
    annual_ret = (final_value / initial_capital) ** (245 / max(n_days, 1)) - 1 if n_days > 0 else 0
    max_dd = dd.max()
    excess = df["daily_return"].mean() - 0.03 / 245
    std_daily = df["daily_return"].std()
    sharpe = (excess / std_daily) * np.sqrt(245) if std_daily > 1e-8 else 0

    # Calmar
    calmar = annual_ret / max_dd if max_dd > 0 else 0

    # 涨跌比
    win_days = (df["daily_return"] > 0).sum()
    loss_days = (df["daily_return"] < 0).sum()
    win_rate = win_days / (win_days + loss_days) * 100

    print(f"  初始资金:       ¥{initial_capital:>8,.0f}")
    print(f"  最终净值:       ¥{final_value:>8,.0f}")
    print(f"  总收益:         {total_return*100:>+7.2f}%")
    print(f"  年化收益:       {annual_ret*100:>+7.2f}%")
    print(f"  夏普比率:       {sharpe:>7.2f}")
    print(f"  最大回撤:       {max_dd*100:>7.2f}%")
    print(f"  Calmar比:       {calmar:>7.2f}")
    print(f"  年化波动率:     {std_daily*np.sqrt(245)*100:>7.2f}%")
    print(f"  涨跌比:         {win_rate:>7.1f}%")
    print(f"  回测天数:       {n_days}")

    # ── 7. 时间线摘要 ──
    print("\n" + "=" * 65)
    print("  七、关键交易时间线")
    print("=" * 65)

    trade_events = df[df["action"] != "hold"].copy()
    # 显示关键转折点
    key_events = []
    for _, row in trade_events.iterrows():
        a = row["action"]
        h = row["hold_symbol"]
        cr = row["cumulative_return_pct"]
        key_events.append(f"  {row['date'].strftime('%Y-%m-%d'):<12} {a:<18} {ETF_NAMES.get(h, h):<8} 累积{cr:+6.2f}%")

    # 显示前30个关键事件
    for line in key_events[:30]:
        print(line)
    if len(key_events) > 30:
        print(f"  ... 还有 {len(key_events)-30} 笔未展示")

    # ── 输出摘要到文件 ──
    summary = {
        "总收益": f"{total_return*100:+.2f}%",
        "年化收益": f"{annual_ret*100:+.2f}%",
        "夏普比率": f"{sharpe:.2f}",
        "最大回撤": f"{max_dd*100:.2f}%",
        "Calmar比": f"{calmar:.2f}",
        "年化波动": f"{std_daily*np.sqrt(245)*100:.2f}%",
        "涨跌比": f"{win_rate:.1f}%",
        "持仓天数": f"{hold_days}/{total_days} ({hold_days/total_days*100:.0f}%)",
        "交易次数": len(actions),
    }

    return df, summary


def _max_drawdown(values):
    peak = np.maximum.accumulate(values)
    dd = (peak - values) / peak
    return dd.max()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--money", type=float, default=INITIAL_CAPITAL)
    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print(f"  配对交易 → 纯多头轮动 | 方案A 深度分析")
    print(f"  {'=' * 65}")
    print(f"  区间: {args.start} → {args.end or '最新'}")
    print(f"  资金: {args.money:,.0f} | 配对: {len(PAIRS)} 对")
    print(f"  参数: z开>{ZSCORE_OPEN} 平<{ZSCORE_CLOSE} | 窗口{ZSCORE_PERIOD}日")

    # 加载数据
    symbols = list(set(p["a"] for p in PAIRS) | set(p["b"] for p in PAIRS))
    print(f"\n  [1] 加载数据 ({len(symbols)}只ETF)…")
    etf_data, dates = load_all_etf_data(
        symbols=symbols, start_date=args.start, end_date=args.end,
        db_path=DB_PATH, momentum_window=ZSCORE_PERIOD,
    )
    print(f"      {len(dates)} 个交易日")

    # 运行方案A
    print(f"  [2] 运行方案A回测…")
    records, trades = run_mode_a(etf_data, dates, initial_capital=args.money)
    print(f"      {len(records)} 条记录, {len(trades)} 笔交易")

    # 分析
    print(f"  [3] 深度分析…")
    df, summary = analyze_mode_a(records, trades, etf_data, dates, args.money)

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_DIR, f"analysis_mode_a_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "detailed.csv"), index=False)

    # 保存摘要
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"  [4] 输出 → {out_dir}")
    print(f"\n{'=' * 65}")
    print(f"  分析完成 ✓")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
