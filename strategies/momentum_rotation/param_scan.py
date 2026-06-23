"""
简化的参数扫描：窗口 + 最小持仓天数。
每次修改 config.py，运行回测，从 metrics.csv 提取结果。
"""
import subprocess, sys, os, re, csv, time
import pandas as pd

PROJ_ROOT = "/public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest"
CONFIG = f"{PROJ_ROOT}/strategies/momentum_rotation/config.py"

RESULTS = []

def set_param(name: str, value):
    """修改 config.py 中的参数值。"""
    with open(CONFIG) as f:
        text = f.read()
    if isinstance(value, float):
        text = re.sub(rf"^{name}\s*=\s*[\d.]+", f"{name} = {value}", text, flags=re.MULTILINE)
    else:
        text = re.sub(rf"^{name}\s*=\s*\d+", f"{name} = {value}", text, flags=re.MULTILINE)
    with open(CONFIG, "w") as f:
        f.write(text)

def run_and_extract(tag: str) -> dict:
    """运行回测并从 metrics.csv 提取关键指标。"""
    proc = subprocess.run(
        [sys.executable, "-m", "strategies.momentum_rotation.run", "--tag", tag],
        capture_output=True, text=True, timeout=300,
        cwd=PROJ_ROOT,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr[:500]}

    # 从 stdout 找 metrics.csv 路径
    match = re.search(r"绩效指标.*?→\s*(\S+)", proc.stdout)
    if not match:
        return {"error": "找不到 metrics.csv 路径"}

    csv_path = match.group(1).strip()
    if not os.path.exists(csv_path):
        return {"error": f"文件不存在: {csv_path}"}

    # 读取 metrics.csv → dict
    df = pd.read_csv(csv_path)
    metrics = dict(zip(df["指标"], df["数值"]))

    # 同时从 daily_records.csv 算2025年指标
    daily_match = re.search(r"净值日报表.*?→\s*(\S+)", proc.stdout)
    ret_2025 = 0.0
    dd_2025 = 0.0
    if daily_match:
        daily_path = daily_match.group(1).strip()
        if os.path.exists(daily_path):
            dd = pd.read_csv(daily_path)
            dd["date"] = pd.to_datetime(dd["date"])
            yr25 = dd[(dd["date"] >= "2025-01-01") & (dd["date"] < "2026-01-01")]
            if len(yr25) > 1:
                ret_2025 = round((yr25["total_value"].iloc[-1] / yr25["total_value"].iloc[0] - 1) * 100, 2)
                peak = yr25["total_value"].cummax()
                drawdown = yr25["total_value"] / peak - 1
                dd_2025 = round(drawdown.min() * 100, 2)

    return {
        "total_ret": metrics.get("累计收益率", "0%").replace("%", ""),
        "max_dd": metrics.get("最大回撤", "0%").replace("%", ""),
        "sharpe": metrics.get("夏普比率", "0"),
        "switches": metrics.get("调仓切换次数", "0"),
        "cost": metrics.get("交易总成本", "0"),
        "ret_2025": str(ret_2025),
        "dd_2025": str(dd_2025),
        "error": "",
    }


# ========== 测试计划 ==========
# 先设 baseline: window=15, conviction=0.03, hold=0
set_param("MOMENTUM_WINDOW", 15)
set_param("MIN_SWITCH_CONVICTION", 0.03)
set_param("MIN_HOLD_DAYS", 0)
BASELINE = run_and_extract("bl15c3h0")

# 测试1: 动量窗口 [10, 12, 15, 18, 20] @ conviction=0.03, hold=0
print(f"\n{'='*80}")
print(f"  测试1: 动量窗口 (Conviction=3%, Hold=0)")
print(f"{'='*80}")
print(f"  {'窗口':>4} | {'总收益':>7} {'回撤':>6} {'夏普':>6} {'切换':>4} {'费用':>6} | {'2025收益':>7} {'2025回撤':>7}")
print(f"  {'-'*60}")
TEST1 = {}
for w in [10, 12, 15, 18, 20]:
    set_param("MOMENTUM_WINDOW", w)
    r = run_and_extract(f"w{w}")
    TEST1[w] = r
    print(f"  {w:>3}d | {r['total_ret']:>6}% {r['max_dd']:>5}% "
          f"{r['sharpe']:>5} {r['switches']:>3} ¥{r['cost']:>5} | "
          f"{r['ret_2025']:>6}% {r['dd_2025']:>5}%")
    if r.get("error"):
        print(f"       ERROR: {r['error']}")

# 找最佳窗口
best_w = max(TEST1, key=lambda w: float(TEST1[w]["total_ret"]) + float(TEST1[w]["ret_2025"]))
print(f"\n  → 最佳窗口: {best_w}d (conviction=3%, hold=0)")

# 测试2: 最佳窗口 + 最小持仓天数 [0, 3, 5, 10]
print(f"\n{'='*80}")
print(f"  测试2: 最小持仓天数 (Window={best_w}, Conviction=3%)")
print(f"{'='*80}")
print(f"  {'持仓':>4} | {'总收益':>7} {'回撤':>6} {'夏普':>6} {'切换':>4} {'费用':>6} | {'2025收益':>7} {'2025回撤':>7}")
print(f"  {'-'*60}")
TEST2 = {}
for h in [0, 3, 5, 10]:
    set_param("MIN_HOLD_DAYS", h)
    r = run_and_extract(f"w{best_w}h{h}")
    TEST2[h] = r
    print(f"  {h:>3}d | {r['total_ret']:>6}% {r['max_dd']:>5}% "
          f"{r['sharpe']:>5} {r['switches']:>3} ¥{r['cost']:>5} | "
          f"{r['ret_2025']:>6}% {r['dd_2025']:>5}%")
    if r.get("error"):
        print(f"       ERROR: {r['error']}")

# 测试3: 最佳窗口+最佳持仓 + 置信度 [0.02, 0.03, 0.04, 0.05]
best_h = min(TEST2.keys(), key=lambda h: int(TEST2[h]["switches"]))
actual_best_h = 0
best_score = -999
for h, r in TEST2.items():
    score = float(r["total_ret"]) * 0.5 + float(r["ret_2025"]) * 0.3 - float(r["max_dd"]) * 0.2
    if score > best_score:
        best_score = score
        actual_best_h = h

print(f"\n{'='*80}")
print(f"  测试3: 置信度 (Window={best_w}, Hold={actual_best_h})")
print(f"{'='*80}")
print(f"  {'置信':>4} | {'总收益':>7} {'回撤':>6} {'夏普':>6} {'切换':>4} {'费用':>6} | {'2025收益':>7} {'2025回撤':>7}")
print(f"  {'-'*60}")
TEST3 = {}
for c in [0.02, 0.03, 0.04, 0.05]:
    set_param("MIN_SWITCH_CONVICTION", c)
    r = run_and_extract(f"w{best_w}h{actual_best_h}c{int(c*100)}")
    TEST3[c] = r
    print(f"  {c:.0%}  | {r['total_ret']:>6}% {r['max_dd']:>5}% "
          f"{r['sharpe']:>5} {r['switches']:>3} ¥{r['cost']:>5} | "
          f"{r['ret_2025']:>6}% {r['dd_2025']:>5}%")
    if r.get("error"):
        print(f"       ERROR: {r['error']}")


# ========== 最终推荐 ==========
print(f"\n{'='*80}")
print(f"  最终对比")
print(f"{'='*80}")
print(f"  {'配置':>24} | {'总收益':>7} {'回撤':>6} {'夏普':>6} {'切换':>4} | {'2025收益':>7} {'2025回撤':>7}")
print(f"  {'-'*70}")

# 列出 top results
all_results = {}
for w, r in TEST1.items():
    all_results[f"窗{w}d c3% h0"] = r
for h, r in TEST2.items():
    all_results[f"窗{best_w}d c3% h{h}d"] = r
for c, r in TEST3.items():
    all_results[f"窗{best_w}d c{int(c*100)}% h{actual_best_h}d"] = r

# 评分排序
def score(r):
    return (float(r["total_ret"]) * 0.4
            - float(r["max_dd"]) * 0.2
            + float(r["sharpe"]) * 20
            - int(r["switches"]) * 0.2
            + float(r["ret_2025"]) * 0.2
            - float(r["dd_2025"]) * 0.15)

ranked = sorted(all_results.items(), key=lambda x: score(x[1]), reverse=True)
for name, r in ranked[:8]:
    print(f"  {name:>24} | {r['total_ret']:>6}% {r['max_dd']:>5}% "
          f"{r['sharpe']:>5} {r['switches']:>3} | "
          f"{r['ret_2025']:>6}% {r['dd_2025']:>5}%")

print(f"\n  Baseline (15d c3% h0): {BASELINE['total_ret']}% "
      f"回撤{BASELINE['max_dd']}% 夏普{BASELINE['sharpe']}")
