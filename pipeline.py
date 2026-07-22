#!/usr/bin/env python
"""ETF 模拟盘全自动管线编排器。

由 cron 在 20:00 启动，依次执行：
  1. 数据同步（etf_sync）— 必需
  2. 动量轮动模拟盘 — 必需

上一步完成后立即启动下一步，不依赖固定时间。

使用方法：
    python pipeline.py

状态文件：pipeline_status.json（本项目根目录）
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

# 强制 stdout/stderr 无缓冲，确保日志按时间顺序写入文件
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── 让 Python 能找到项目包 ──
PROJECT_DIR: Path = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from pipeline_status import PipelineStatus, push_pipeline_summary
from simulation.framework.notify import send_message
from simulation.framework.summary import push_strategy_summary, gather_pipeline_info

# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════

PYTHON: str = sys.executable  # 当前 Python 解释器

# ════════════════════════════════════════════════════════════
#  交易日判断
# ════════════════════════════════════════════════════════════

def is_trade_day(check_date: date | None = None) -> bool:
    """判断是否为 A 股交易日。

    策略：周末过滤 → chinese_calendar 节假判断。
    """
    if check_date is None:
        check_date = date.today()

    if check_date.weekday() >= 5:
        return False

    try:
        from chinese_calendar import is_workday
        return is_workday(check_date)
    except ImportError:
        # fallback：数据库中有数据即视为交易日
        import sqlite3
        day_str = check_date.strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(str(PROJECT_DIR / "data" / "etf_daily.db"))
            cur = conn.execute(
                "SELECT COUNT(*) FROM etf_daily WHERE date = ? LIMIT 1",
                (day_str,),
            )
            cnt = cur.fetchone()[0]
            conn.close()
            return cnt > 0
        except Exception:
            return True  # fail-open


# ════════════════════════════════════════════════════════════
#  步骤配置
# ════════════════════════════════════════════════════════════

STEPS: list[dict] = [
    # ── 1. 数据同步（必需） ──
    {
        "id": "sync",
        "name": "ETF 数据同步",
        "cmd": ["main.py", "--sync-only"],
        "cwd": str(PROJECT_DIR),
        "required": True,
        "timeout": 10800,
    },
    # ═══ 动量类策略（批量推送，BATCH_MODE=1） ═══
    {
        "id": "momentum_rotation",
        "name": "动量轮动",
        "cmd": ["-m", "simulation.strategies.momentum_rotation.daily"],
        "cwd": str(PROJECT_DIR),
        "required": True,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "composite_momentum",
        "name": "复合动量",
        "cmd": ["-m", "simulation.strategies.composite_momentum.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "macd_trend_rotation",
        "name": "MACD趋势轮动",
        "cmd": ["-m", "simulation.strategies.macd_trend_rotation.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "rsi_trend_rotation",
        "name": "RSI趋势确认",
        "cmd": ["-m", "simulation.strategies.rsi_trend_rotation.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "adaptive_rotation",
        "name": "自适应轮动",
        "cmd": ["-m", "simulation.strategies.adaptive_rotation.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "adx_trend_rotation",
        "name": "ADX趋势强度",
        "cmd": ["-m", "simulation.strategies.adx_trend_rotation.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "momentum_vol_filter",
        "name": "波动率过滤",
        "cmd": ["-m", "simulation.strategies.momentum_vol_filter.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "pair_trading",
        "name": "配对交易风格轮动",
        "cmd": ["-m", "simulation.strategies.pair_trading.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
        "batch": True,
    },
    {
        "id": "combined",
        "name": "组合策略(动量80%+配对20%)",
        "cmd": ["-m", "simulation.strategies.combined.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 300,
        "batch": True,
    },
    # ═══ 独立推送策略（不参与批量，各自单独推送） ═══
    
    # ═══ 新纳入: 机制独立于纯动量的候选策略 ═══
    {
        "id": "dual_momentum",
        "name": "双动量",
        "cmd": ["-m", "simulation.strategies.dual_momentum.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "sortino_ranking",
        "name": "Sortino排名",
        "cmd": ["-m", "simulation.strategies.sortino_ranking.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "sharpe_ranking",
        "name": "Sharpe排名",
        "cmd": ["-m", "simulation.strategies.sharpe_ranking.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "median_momentum",
        "name": "中位数#2",
        "cmd": ["-m", "simulation.strategies.median_momentum.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "tail_risk",
        "name": "尾部风险轮动",
        "cmd": ["-m", "simulation.strategies.tail_risk.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "bollinger_reversion",
        "name": "布林带回归",
        "cmd": ["-m", "simulation.strategies.bollinger_reversion.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "spread_reversion",
        "name": "价差回归",
        "cmd": ["-m", "simulation.strategies.spread_reversion.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "volume_price",
        "name": "量价配合",
        "cmd": ["-m", "simulation.strategies.volume_price.daily"],
        "cwd": str(PROJECT_DIR), "required": False, "timeout": 600, "batch": True,
    },
    {
        "id": "gold_safe_haven",
        "name": "黄金避险轮动 🥇",
        "cmd": ["-m", "simulation.strategies.gold_safe_haven.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
    },
    {
        "id": "cross_border",
        "name": "跨境轮动 🌏",
        "cmd": ["-m", "simulation.strategies.cross_border.daily"],
        "cwd": str(PROJECT_DIR),
        "required": False,
        "timeout": 600,
    },
]


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def main():
    today = date.today()
    today_str = today.isoformat()

    print(f"\n{'=' * 55}")
    print(f"  ETF 模拟盘管线 | {today_str}")
    print(f"  {'=' * 55}")

    # 交易日检查
    if not is_trade_day(today):
        print(f"  {today_str} 非交易日，跳过管线")
        # 记录跳过状态
        ps = PipelineStatus()
        ps.reset()
        for s in STEPS:
            ps.add_step(s["id"], s["name"])
        ps.finish("skipped")
        return

    # 初始化状态（新建当日记录）
    ps = PipelineStatus()
    ps.reset()

    for s in STEPS:
        ps.add_step(s["id"], s["name"])

    pipeline_ok = True

    # 逐步骤执行
    for step in STEPS:
        sid = step["id"]
        sname = step["name"]
        required = step.get("required", False)
        timeout = step.get("timeout", 0)

        # ── 批量模式：动量类策略合并推送，跨境/黄金独立推送 ──
        if step.get("batch"):
            os.environ["BATCH_MODE"] = "1"
        else:
            os.environ.pop("BATCH_MODE", None)

        print(f"\n  ▶ {sname}...")
        ps.start_step(sid)

        t0 = time.time()
        try:
            proc = subprocess.Popen(
                [PYTHON] + step["cmd"],
                cwd=step["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # 流式输出子进程日志
            stdout_lines = []
            stderr_lines = []
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip("\n")
                    stdout_lines.append(line)
                    print(f"    {line}")
            if proc.stderr:
                for line in iter(proc.stderr.readline, ""):
                    line = line.rstrip("\n")
                    stderr_lines.append(line)
                    print(f"    {line}", file=sys.stderr)
            proc.wait(timeout=timeout if timeout > 0 else None)

            elapsed = time.time() - t0
            success = proc.returncode == 0

            stdout_text = "\n".join(stdout_lines[-20:])   # 保留最后20行
            stderr_text = "\n".join(stderr_lines[-20:])

            detail = {
                "returncode": proc.returncode,
                "stdout_last": stdout_text[-300:],
                "stderr_last": stderr_text[-300:],
            }

            if success:
                print(f"  ✅ {sname} 完成（{elapsed:.0f}s）")
                ps.complete_step(sid, success=True, detail=detail)
            else:
                print(f"  ❌ {sname} 失败（{elapsed:.0f}s）")
                error_detail = stderr_text[-500:] if stderr_text else "未知错误"
                ps.complete_step(sid, success=False, detail=detail, error=error_detail)
                if required:
                    pipeline_ok = False
                    break
                else:
                    print(f"  ⚠ {sname} 失败但非必需，继续下一项")

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"  ⏰ {sname} 超时（{timeout}s）")
            ps.complete_step(sid, success=False, error=f"超时 {timeout}s")
            if required:
                pipeline_ok = False
                break

        except Exception as e:
            print(f"  💥 {sname} 异常: {e}")
            ps.complete_step(sid, success=False, error=str(e))
            if required:
                pipeline_ok = False
                break

    # 管线结束
    status = "completed" if pipeline_ok else "failed"
    ps.finish(status)

    # ── 推送批量聚合的动量类策略日报 ──
    try:
        from simulation.framework.notify import flush_batch_reports
        flush_batch_reports("动量类策略合集")
    except Exception as e:
        print(f"  ⚠ 推送批量日报异常: {e}")

    # 推送管线汇总
    try:
        push_pipeline_summary(ps.to_dict(), send_message)
    except Exception as e:
        print(f"  ⚠ 推送管线汇总异常: {e}")

    # 推送策略汇总日报
    try:
        output_dir = str(PROJECT_DIR / "simulation" / "output")
        pinfo = gather_pipeline_info(PROJECT_DIR / "pipeline_status.json")
        push_strategy_summary(output_dir, send_message, pipeline_info=pinfo)
    except Exception as e:
        print(f"  ⚠ 推送策略汇总异常: {e}")

    # 清理环境变量
    os.environ.pop("BATCH_MODE", None)

    print(f"\n  {'=' * 55}")
    print(f"  管线状态: {status}")
    print(f"  {'=' * 55}\n")

    sys.exit(0 if pipeline_ok else 1)


if __name__ == "__main__":
    main()
