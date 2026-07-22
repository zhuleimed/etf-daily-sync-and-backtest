"""
策略汇总日报模块

每日 pipeline 运行后调用，读取各策略的 state JSON 和 CSV 日志：
  1. state_{id}.json  → 当前持仓、现金、总资产
  2. sim_log_{id}.csv  → 全部历史收益数据，用于计算年化/夏普/回撤/胜率

输出：一条微信推送，汇总所有模拟盘策略的全维度对比。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ── 策略ID → 中文简称（当CSV不可用时作为后备） ──
_FALLBACK_NAMES: dict[str, str] = {
    "momentum_rotation": "动量轮动",
    "composite_momentum": "复合动量",
    "macd_trend_rotation": "MACD趋势",
    "adx_trend_rotation": "ADX趋势",
    "rsi_trend_rotation": "RSI趋势",
    "momentum_vol_filter": "波动率过滤",
    "pair_trading": "配对交易",
    "adaptive_rotation": "自适应轮动",
    "gold_safe_haven": "黄金避险",
    "cross_border": "跨境轮动",
        "dual_momentum": "双动量",
    "sortino_ranking": "Sortino排名",
    "sharpe_ranking": "Sharpe排名",
    "median_momentum": "中位数#2",
    "tail_risk": "尾部风险轮动",
    "bollinger_reversion": "布林带回归",
    "spread_reversion": "价差回归",
    "volume_price": "量价配合",
    "combined": "组合策略",
}

# ═══════════════════════════════════════════════
#  数据读取
# ═══════════════════════════════════════════════


def _state_path(output_dir: str, strategy_id: str) -> Path:
    return Path(output_dir) / f"state_{strategy_id}.json"


def _csv_path(output_dir: str, strategy_id: str) -> Path:
    return Path(output_dir) / f"sim_log_{strategy_id}.csv"


def _read_state(output_dir: str, strategy_id: str) -> Optional[dict]:
    p = _state_path(output_dir, strategy_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _read_csv(output_dir: str, strategy_id: str) -> Optional[pd.DataFrame]:
    p = _csv_path(output_dir, strategy_id)
    if p.exists():
        try:
            return pd.read_csv(str(p), encoding="utf-8-sig")
        except Exception:
            return None
    return None


def _extract_strategy_id(filename: str) -> Optional[str]:
    """从 state_xxx.json 提取策略ID。"""
    if filename.startswith("state_") and filename.endswith(".json"):
        return filename[6:-5]
    return None


# ═══════════════════════════════════════════════
#  指标计算
# ═══════════════════════════════════════════════


def _compute_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    """从CSV历史数据计算绩效指标。"""
    n = len(df)
    if n < 2:
        return {"n_days": n}

    tv = df["总资产"].values
    ic = initial_capital if initial_capital > 0 else tv[0]

    # 累计收益率
    total_return = tv[-1] / ic - 1

    # 年化（至少20天数据才可信）
    annual_return = (1 + total_return) ** (252 / n) - 1 if n >= 20 else None

    # 日收益率
    tv_series = pd.Series(tv)
    daily_ret = tv_series.pct_change().fillna(0.0)

    # 夏普（需有波动）
    std_r = daily_ret.std()
    if std_r > 1e-10:
        rf_daily = 0.03 / 252
        mean_r = daily_ret.mean()
        sharpe = round((mean_r - rf_daily) / std_r * np.sqrt(252), 2)
    else:
        sharpe = None

    # 最大回撤
    cumulative = tv / tv[0]
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    # 胜率
    win_rate = (daily_ret > 0).sum() / n if n > 0 else None

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_days": n,
    }


# ═══════════════════════════════════════════════
#  数据组装
# ═══════════════════════════════════════════════


def build_strategy_summary(output_dir: str) -> list[dict]:
    """扫描状态文件，读取CSV，组装全量汇总数据。"""
    out = Path(output_dir)
    if not out.exists():
        return []

    strategies: list[dict] = []

    for f in sorted(out.glob("state_*.json")):
        sid = _extract_strategy_id(f.name)
        if not sid:
            continue

        state = _read_state(output_dir, sid)
        csv_df = _read_csv(output_dir, sid)

        entry: dict[str, Any] = {"strategy_id": sid}

        # ── 策略名称 ──
        if csv_df is not None and not csv_df.empty:
            entry["name"] = str(csv_df.iloc[-1].get("策略", _FALLBACK_NAMES.get(sid, sid)))
        else:
            entry["name"] = _FALLBACK_NAMES.get(sid, sid)

        # ── 启动日期 ──
        if csv_df is not None and len(csv_df) > 0:
            raw = str(csv_df.iloc[0].get("日期", ""))
            entry["start_date"] = raw.replace("←历史起点", "").strip()
        else:
            entry["start_date"] = str(state.get("last_update", "")[:10]) if state else ""

        # ── 当前实时状态（来自 state JSON） ──
        if state:
            pos = state.get("position", {})
            entry["hold_symbol"] = str(pos.get("symbol", ""))
            entry["hold_shares"] = int(pos.get("shares", 0))
            entry["cash"] = float(state.get("cash", 0))
            entry["total_value"] = float(state.get("total_value", state.get("cash", 0)))
            entry["initial_capital"] = float(state.get("initial_capital", 10000))
            entry["pending_order"] = state.get("pending_order")
        else:
            entry["hold_symbol"] = ""
            entry["hold_shares"] = 0
            entry["cash"] = 0.0
            entry["total_value"] = 0.0
            entry["initial_capital"] = 10000
            entry["pending_order"] = None

        # ── 今日操作描述（来自 CSV 最后一行） ──
        if csv_df is not None and not csv_df.empty:
            last = csv_df.iloc[-1]
            # 用 get 取列再转 str，处理 pandas NaN
            def _safe_str(v: Any) -> str:
                if isinstance(v, float) and np.isnan(v):
                    return ""
                return str(v)
            entry["last_action"] = _safe_str(last.get("操作", ""))
            entry["last_pending"] = _safe_str(last.get("明日待执行", ""))
            entry["last_date"] = _safe_str(last.get("日期", ""))
        else:
            entry["last_action"] = ""
            entry["last_pending"] = ""
            entry["last_date"] = ""

        # ── 绩效指标（至少需要2个数据点） ──
        if csv_df is not None and len(csv_df) >= 2:
            try:
                entry["metrics"] = _compute_metrics(
                    csv_df, entry.get("initial_capital", 10000)
                )
            except Exception:
                entry["metrics"] = {}
        else:
            entry["metrics"] = {}

        strategies.append(entry)

    return strategies


# ═══════════════════════════════════════════════
#  文本格式化
# ═══════════════════════════════════════════════


def format_summary_text(
    strategies: list[dict],
    pipeline_info: Optional[dict] = None,
) -> str:
    """生成微信推送文本。"""
    today = datetime.now().strftime("%m-%d")
    lines: list[str] = []

    lines.append(f"📊 ETF模拟盘策略汇总 | {today}")
    lines.append("═" * 40)

    if not strategies:
        lines.append("\n暂无策略运行数据")
        lines.append("═" * 40)
        return "\n".join(lines)

    # 按收益率降序
    sorted_s = sorted(strategies,
                      key=lambda s: s.get("metrics", {}).get("total_return", -1),
                      reverse=True)

    emoji_bullets = ["❶", "❷", "❸", "❹", "❺", "❻", "❼", "❽", "❾", "❿"]

    for i, s in enumerate(sorted_s):
        bullet = emoji_bullets[i] if i < len(emoji_bullets) else f"{i+1}."
        name = s.get("name", "?")
        m = s.get("metrics", {})

        # 第一行：策略名 + 核心指标
        tr = m.get("total_return")
        ar = m.get("annual_return")
        sp = m.get("sharpe")
        md = m.get("max_drawdown")
        nd = m.get("n_days", 0)

        if nd < 2:
            # 数据不足
            lines.append(f"\n{bullet} {name}")
            lines.append(f"  📊 数据收集中（不足2天）")
            start = s.get("start_date", "")[:10]
            lines.append(f"  {'启动' + start if start else '启动--'} | 💤 空仓")
            continue

        tr_s = f"{tr:+.0%}" if tr is not None else "--"
        ar_s = f"{ar:+.0%}" if ar is not None else ("--" if nd < 20 else f"{ar:+.0%}")
        ar_s = ar_s if nd >= 20 else "起步"
        sp_s = f"{sp:.2f}" if sp is not None else "--"
        md_s = f"{md:.0%}" if md is not None else "--"

        lines.append(f"\n{bullet} {name}")
        lines.append(f"  累计{tr_s} 年化{ar_s} 夏普{sp_s} 回撤{md_s}")

        # 第二行：启动 + 胜率 + 当前状态
        start = s.get("start_date", "")[:10]
        wr = m.get("win_rate")
        wr_s = f"胜率{wr:.0%}" if wr is not None else ""
        nd_s = f"{nd}天"

        # 持仓描述
        hold_sym = s.get("hold_symbol", "")
        hold_shares = s.get("hold_shares", 0)
        last_action = s.get("last_action", "")
        pending = s.get("last_pending", "")

        # 清理空字符串/NaN
        if not pending or pending == "nan":
            pending = ""
        if not last_action or last_action == "nan":
            last_action = ""

        if hold_sym and hold_shares > 0:
            pos_desc = f"📈 {hold_sym}×{hold_shares}"
            if pending:
                pos_desc += f" ⏩{pending[:20]}"
        elif last_action:
            # 截断操作描述到25字
            pos_desc = last_action[:25]
            if len(last_action) > 25:
                pos_desc += "…"
        else:
            pos_desc = "💤 空仓"

        info_parts = [f"启动{start}" if start else "启动--"]
        if wr_s:
            info_parts.append(wr_s)
        info_parts.append(nd_s)
        lines.append(f"  {' '.join(info_parts)} | {pos_desc}")

    # 底部
    lines.append("\n" + "─" * 40)

    if pipeline_info:
        total = pipeline_info.get("total", 0)
        ok = pipeline_info.get("ok", 0)
        elapsed = pipeline_info.get("elapsed", "")
        status = pipeline_info.get("status", "")
        emoji = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
        lines.append(f"{emoji} 管线: {ok}/{total} 完成 | 耗时 {elapsed}")
    else:
        lines.append(f"🕐 {datetime.now().strftime('%H:%M')}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════
#  推送入口
# ═══════════════════════════════════════════════


def push_strategy_summary(
    output_dir: str,
    send_message_fn: Callable,
    pipeline_info: Optional[dict] = None,
) -> bool:
    """读取状态 → 计算指标 → 格式化 → 推送一条汇总微信。"""
    strategies = build_strategy_summary(output_dir)
    text = format_summary_text(strategies, pipeline_info)
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"📊 ETF模拟盘汇总 | {today}"
    return send_message_fn(title, text)


def gather_pipeline_info(status_path: str | Path) -> Optional[dict]:
    """从 pipeline_status.json 提取管线信息。"""
    p = Path(status_path)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        steps = raw.get("steps", {})
        total = len(steps)
        ok = sum(1 for s in steps.values() if s.get("status") == "completed")
        started = raw.get("started_at", "")
        finished = raw.get("finished_at", "")
        elapsed = ""
        if started and finished:
            fmt = "%H:%M:%S"
            try:
                t1 = datetime.strptime(started, fmt)
                t2 = datetime.strptime(finished, fmt)
                delta = (t2 - t1).total_seconds()
                elapsed = f"{int(delta//60)}分{int(delta%60)}秒"
            except ValueError:
                elapsed = ""
        return {
            "total": total,
            "ok": ok,
            "elapsed": elapsed,
            "status": raw.get("pipeline_status", ""),
        }
    except Exception:
        return None
