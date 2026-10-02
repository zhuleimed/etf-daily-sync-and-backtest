#!/usr/bin/env python3
"""接线审计：列出每条模拟盘实际使用的信号函数来自哪个模块。

用法：
    python -m simulation.analysis.audit_wiring

什么时候跑：**加新策略后必跑**、每季度核对一次。与 README「信号来源清单」对照。
输出到 stdout 的表格可直接替换 README 对应小节。

判据（2026-09-25 接线审计结论）：
    不是"有没有用动量函数"，而是 **实盘是否与该策略自己的回测一致**。
    - 用自身模块（strategies.<同名>.momentum_signals/signals）= 正常
    - 复用 momentum_rotation = 需要判断：设计如此（跨境轮动换池子、波动率过滤
      以动量为基座）还是漏接（2026-09 曾有 8 条候选策略的模拟盘全部在跑动量轮动，
      曲线逐字节相同，几个月无人发现）
    - 不走引擎（自定义流程）= 需手工维护引擎本该负责的一切（T+1 执行、状态字段、
      快照、日报）——本项目 B014/B015 就是这类漏维护造成的

参考：strategies/REWIRE_SIGNALS_20260925.md、memory signal-wiring-audit。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = ROOT / "simulation" / "strategies"

# 设计如此、允许复用 momentum_rotation 信号的策略（附理由）
SHARED_BY_DESIGN = {
    "cross_border": "策略即「动量引擎换跨境ETF池」，回测同样复用 momentum 引擎",
    "momentum_vol_filter": "以动量为基座+波动率过滤，其回测信号与动量版逐字节相同",
    "hs300_ma_timing": "动量+HS300择时叠加（未纳入 pipeline）",
    "market_breadth": "动量+市场宽度择时叠加（未纳入 pipeline）",
}


def _imports(src: str) -> dict[str, tuple[str, str]]:
    """解析 import，返回 {本地名: (模块, 原名)}（支持 as 别名）。"""
    imp: dict[str, tuple[str, str]] = {}
    pat = re.compile(r"from\s+([\w\.]+)\s+import\s+(\([^)]*\)|[^\n(]+)")
    for m in pat.finditer(src):
        mod, names = m.group(1), m.group(2).strip()
        if names.startswith("("):
            names = names[1:-1]
        for n in names.split(","):
            n = n.strip()
            if not n:
                continue
            if " as " in n:
                orig, alias = (x.strip() for x in n.split(" as "))
                imp[alias] = (mod, orig)
            else:
                imp[n] = (mod, n)
    return imp


def _def_body(src: str, fn: str) -> str:
    m = re.search(rf"def\s+{re.escape(fn)}\(.*?(?=\ndef |\Z)", src, re.S)
    return m.group(0) if m else ""


def resolve_signal(sid: str) -> tuple[str, str, str]:
    """返回 (signal_func 名, 来源模块, 判定)。"""
    src = (SIM_DIR / sid / "daily.py").read_text(encoding="utf-8")
    imp = _imports(src)
    m = re.search(r"signal_func=([\w\.]+)", src)
    if not m:
        return ("(无引擎)", "-", "自定义流程（不走 DailySimEngine）")
    name = m.group(1)

    # 本文件内定义的包装函数 → 看它调用了哪个 import 进来的函数
    if re.search(rf"def\s+{re.escape(name)}\(", src):
        body = _def_body(src, name)
        calls = [c for c in re.findall(r"(\w+)\s*\(", body) if c in imp]
        if calls:
            mod, orig = imp[calls[0]]
            return (f"{name} → {orig}", mod, "用自身模块" if f"strategies.{sid}." in mod else f"复用 {mod}")
        return (name, "本文件内实现", "自定义流程（不走 DailySimEngine）")

    # 直接 import 的函数
    if name in imp:
        mod, orig = imp[name]
        verdict = "用自身模块" if f"strategies.{sid}." in mod else f"复用 {mod}"
        return (f"{orig}", mod, verdict)
    return (name, "?", "未知（需人工核对）")


def main() -> None:
    sims = sorted(p.name for p in SIM_DIR.iterdir() if (p / "daily.py").exists())
    pipe = set(re.findall(r'"id":\s*"(\w+)"', (ROOT / "pipeline.py").read_text(encoding="utf-8")))
    own, shared, custom = [], [], []
    print(f"\n{'策略':<22}{'在管道':<7}{'signal_func → 来源模块':<58}{'判定'}")
    print("-" * 130)
    for sid in sims:
        fn, mod, verdict = resolve_signal(sid)
        inp = "✓" if sid in pipe else "—"
        print(f"{sid:<22}{inp:<7}{(fn + '  ←  ' + mod):<58}  {verdict}")
        if "自定义流程" in verdict:
            custom.append(sid)
        elif "复用" in verdict:
            shared.append((sid, SHARED_BY_DESIGN.get(sid, "⚠ 需核对是否与自身回测一致")))
        else:
            own.append(sid)
    print("\n" + "=" * 70)
    print(f"用自身信号 {len(own)} 条：{', '.join(own)}")
    print(f"\n复用其他策略信号 {len(shared)} 条：")
    for sid, why in shared:
        flag = "" if "⚠" not in why else "   ← 若非设计如此，即为漏接 bug（参见 09-25 审计）"
        print(f"   {sid:<22} {why}{flag}")
    print(f"\n自定义流程 {len(custom)} 条：{', '.join(custom)}")
    print("\n注：自定义流程需手工维护引擎本该负责的状态/执行逻辑（B014/B015 教训）。")


if __name__ == "__main__":
    main()
