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
    "asset_allocation": "资产配置(风险评价)",
    "neural_momentum": "Neural Momentum",
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


def _since_last_reset(df: pd.DataFrame) -> pd.DataFrame:
    """截取最后一个"清零/重构注释行"之后的数据。

    统一口径：绩效指标与克隆巡检都只认当前这段账期。清零重启（2026-08-03、
    2026-10-08）后历史段仍留在 CSV 里，若从历史算，清零后只跑 1-2 天也会
    显示上一段的旧回撤/旧信号。
    """
    if "操作" not in df.columns:
        return df
    # 定位最后一个注释行（操作列以【开头：清零重启/框架重构）
    mask = df["操作"].astype(str).str.startswith("【")
    last_comment = df.index[mask].max() if mask.any() else None
    return df.iloc[last_comment + 1:] if last_comment is not None else df


def _compute_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    """从CSV历史数据计算绩效指标（从最后一次清零之后算起）。"""
    df = _since_last_reset(df)

    n = len(df)
    if n < 2:
        return {"n_days": n}

    # 过滤无效行（清零/重构注释行的总资产列为空 NaN，污染回撤计算）
    tv = pd.to_numeric(df["总资产"], errors="coerce").dropna().values
    if len(tv) < 2:
        return {"n_days": len(tv)}
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

    # ── 资金曲线指纹 ──
    # 用途：识别"多个策略其实是同一份信号跑出来的同一条曲线"——
    # 典型成因是 daily.py 由模板复制、signal_func 还指向别的策略，
    # 结果 7 个策略名字不同但每日净值逐分相同（2026-09 发现）。
    # 取"元"精度（2位小数）拼接：只有逐日净值完全相同才会撞上。
    fingerprint = ",".join(f"{float(v):.2f}" for v in tv)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_days": n,
        "fingerprint": fingerprint,
    }


def _clone_evidence(df: pd.DataFrame) -> dict:
    """从 CSV 提取克隆巡检要用的信息（口径同指标：只算最后一次清零之后）。

    sig_seq —— 当日"选股打分向量指纹"序列，只保留有可辨识打分的那几天
               （全 0 分/空 = 全市场没信号，不参与比对，见 signal_fp.py）。
    traded_days —— 期间有持仓的天数，用作"这条曲线到底动没动过"的门槛。
    """
    d = _since_last_reset(df)

    if "信号指纹" in d.columns:
        sigs = d["信号指纹"].fillna("").astype(str).str.strip()
    else:
        sigs = pd.Series("", index=d.index)

    traded_days = 0
    if "持仓数量" in d.columns:
        shares = pd.to_numeric(d["持仓数量"], errors="coerce").fillna(0)
        traded_days = int((shares > 0).sum())

    return {
        "sig_seq": tuple(s for s in sigs if s),
        "traded_days": traded_days,
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

        # ── 克隆巡检证据（打分指纹序列 + 有持仓天数） ──
        entry["clone_evidence"] = (
            _clone_evidence(csv_df) if csv_df is not None else
            {"sig_seq": (), "traded_days": 0}
        )

        strategies.append(entry)

    return strategies


# ═══════════════════════════════════════════════
#  文本格式化
# ═══════════════════════════════════════════════


# ── 克隆策略巡检 ──
# 判据是"选股打分向量相同"，而不是"净值相同"——净值相同另有成因，
# 打分相同才是同一个大脑（详见 signal_fp.py）。两条序列合起来分三类，
# 只有第一类会推送告警，另两类只写管线日志：
#   打分同 + 净值同 → 克隆嫌疑（真问题，要查 daily.py 的 signal_func）
#   打分同 + 净值不同 → 共用打分实现（如波动率过滤以动量为基座，设计如此）
#   净值同 + 打分不同 → 买到了同一只标的（巧合，非接线问题）
#
# 覆盖范围：全部 21 个在跑的策略。17 个走 DailySimEngine 的由引擎统一记录，
# 4 个自建 main 的（黄金避险/配对交易/资产配置/组合）各自接了指纹：
#   · 黄金避险   → 动量打分 + 宽基第一名
#   · 配对交易   → 各配对 z-score + 最终选中的标的
#   · 资产配置   → 风险平价目标权重（月末更新，平日不变）
#   · 组合策略   → **没有选股动作**（是"动量80%+配对20%"的净值合成器），
#                  指纹记的是它读到的两路子策略收益 = 它做决策的全部输入
MIN_SIGNAL_DAYS = 5

# 有意共用打分实现的策略（设计如此，不告警）。
#   波动率过滤、尾部风险 都以动量打分为基座，分别叠加"高波动覆写"和
#   "沪深300急跌→切最低波动ETF"，它们的回测引擎也是这么写的
#   （见 strategies/REWIRE_SIGNALS_20260925.md 第二节）。
#   中位数#2 同理以动量为基座，但它把第 2 名提到第 1 名、选中的标的与
#   纯动量不同，指纹（含 target）天然分得开，所以不需要登记。
# 判定规则：一组"打分相同"的策略**全部**落在本名单内，才降级为提示；
# 只要混进来一个名单外的策略（例如克隆事故把布林带回归也接成了动量），
# 组里就会出现名单外成员 → 照常告警。
_INTENTIONAL_SHARED_SCORING = {
    "momentum_rotation",
    "momentum_vol_filter",
    "tail_risk",
}


def analyze_clone_groups(strategies: list[dict]) -> dict[str, list[list[str]]]:
    """按"打分指纹序列 / 净值序列"分组，返回三类名单。

    MIN_SIGNAL_DAYS 是证据门槛：至少要攒够这么多天"有可辨识打分"才判定。
    长假清零后头几天所有策略都在空仓观望，任何比对都没有区分力——
    2026-10-09 那次误报正是只攒了 2 天数据就下了结论。
    """
    sig_groups: dict[tuple, list[dict]] = {}
    curve_groups: dict[str, list[dict]] = {}

    for s in strategies:
        ev = s.get("clone_evidence") or {}
        sigs = ev.get("sig_seq", ())
        if len(sigs) >= MIN_SIGNAL_DAYS:
            sig_groups.setdefault(sigs, []).append(s)

        m = s.get("metrics", {})
        fp = m.get("fingerprint")
        # 净值比对额外要求"期间真的持有过"：空仓不动的曲线人人相同，
        # 拿它当证据只会制造噪音（清零后 13 个空仓策略就是这么撞在一起的）。
        if fp and m.get("n_days", 0) >= MIN_SIGNAL_DAYS and ev.get("traded_days", 0) >= 1:
            curve_groups.setdefault(fp, []).append(s)

    clone: list[list[str]] = []
    shared_signal: list[list[str]] = []
    same_curve: list[list[str]] = []

    def _names(members: list[dict]) -> list[str]:
        return [str(m.get("name", m.get("strategy_id", "?"))) for m in members]

    for members in sig_groups.values():
        if len(members) < 2:
            continue
        ids = {str(m.get("strategy_id", "")) for m in members}
        curves = {m.get("metrics", {}).get("fingerprint") for m in members}
        # 净值不同 → 只是共用打分实现，不是克隆；
        # 一圈人全在"有意共用"名单里 → 同理降级为提示。
        if len(curves) > 1 or ids <= _INTENTIONAL_SHARED_SCORING:
            shared_signal.append(_names(members))
        else:
            clone.append(_names(members))

    for members in curve_groups.values():
        if len(members) < 2:
            continue
        seqs = {tuple((m.get("clone_evidence") or {}).get("sig_seq", ())) for m in members}
        if len(seqs) > 1:  # 打分不同 → 只是买到了同一只标的
            same_curve.append(_names(members))

    return {"clone": clone, "shared_signal": shared_signal, "same_curve": same_curve}


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

    # ── 克隆策略告警 ──
    # 只有"选股打分和资金曲线都逐日相同"才推送：这类策略跑的是同一个选股大脑，
    # 会污染横向比较。净值相同但打分不同（买到同一只标的）只写管线日志，
    # 免得天天误报把真正的告警淹掉。
    clones = analyze_clone_groups(strategies)["clone"]
    if clones:
        lines.append(f"⚠️ 疑似克隆策略 {len(clones)} 组（选股打分与资金曲线逐日完全相同）")
        for names in clones:
            lines.append(f"  · {' / '.join(names)}")
        lines.append("  → 打分也一样，说明这些 daily.py 很可能指向了同一个 signal_func")
        lines.append("")

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

    # 同步打到管线日志——推送正文不落盘，出问题时这里才有据可查。
    # 三类都打：推送只出"克隆嫌疑"，另两类留在日志里备查，不占用推送版面。
    analysis = analyze_clone_groups(strategies)
    for key, label in (
        ("clone", "⚠️ 疑似克隆（打分+净值都相同）"),
        ("shared_signal", "ℹ️ 共用打分实现（净值不同）"),
        ("same_curve", "ℹ️ 净值相同但打分不同（买到同一只标的）"),
    ):
        if analysis[key]:
            print(f"  {label} {len(analysis[key])} 组：")
            for names in analysis[key]:
                print(f"     · {' / '.join(names)}")

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
