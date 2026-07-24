"""
模拟盘日报格式化 — 统一日报模板（昨信号执行 + 今日信号 + 排名 + 账户日结）

T+1 时间线：
  T日收盘后 → 计算信号 → 产生"待执行订单"（明日开盘执行）
  T+1日开盘 → 执行昨日待执行订单（用今日开盘价）
  T+1日收盘后 → 计算新信号 → 产生新的"待执行订单"...

日报中：
  - "昨信号执行" = 今日开盘执行了昨天产生的待执行订单
  - "今日新信号" = 今日收盘后计算的新信号，明日开盘执行
"""

from __future__ import annotations

from typing import Any


def build_signal_report(
    report: dict[str, Any],
    strategy_name: str,
    etf_pool: dict[str, str],
    *,
    rank_label: str = "动量排名",
    max_rank_display: int = 3,
) -> list[str]:
    """生成标准格式日结报告。

    报告结构：
      1. 标题行（策略名 | 日期）
      2. 昨信号执行（今日执行了昨天的什么订单）
      3. 今日新信号（今日产生了什么信号，明日开盘执行）+ 排名
      4. 账户日结（持仓、市值、现金、总资产、总收益率）

    Parameters
    ----------
    report : dict
        DailySimEngine.run_daily() 的返回值。
    strategy_name : str
        策略中文名，如 "布林带回归"。
    etf_pool : dict
        {symbol: name} 映射，用于显示 ETF 名称。
    rank_label : str
        排名部分的标题，默认 "动量排名"。
    max_rank_display : int
        最多显示前几名，默认 3。
    """
    state = report.get("state")
    lines: list[str] = []
    action = report.get("action", "unknown")

    def name_of(sym: str) -> str:
        """symbol → 可读名称，如 沪深300ETF(510300)"""
        return f"{etf_pool.get(sym, sym[:4])}({sym})"

    # ── 标题 ──
    lines.append(f"  {strategy_name}  |  {report.get('date', '')}")

    # ═══════════════════════════════════════════════
    #  第一部分：昨信号执行 — 今日开盘执行了昨天的待执行订单
    # ═══════════════════════════════════════════════
    execd = report.get("order_executed")
    blocked = report.get("order_blocked")

    lines.append("")
    lines.append("【昨信号执行】")

    if execd:
        t = execd.get("type", "")
        if t == "buy":
            lines.append(
                f"  >> 开仓: 开盘买入 {name_of(execd['symbol'])} "
                f"{execd['shares']}股 @ {execd['price']:.4f}"
            )
        elif t == "sell":
            lines.append(
                f"  >> 平仓: 开盘卖出 {name_of(execd['symbol'])} "
                f"{execd['shares']}股 @ {execd['price']:.4f}  盈亏{execd.get('pnl', 0):+.2f}"
            )
        elif t == "switch":
            s = execd.get("sell", {})
            b = execd.get("buy", {})
            lines.append(
                f"  >> 切换: 开盘卖出 {name_of(s.get('symbol', ''))}"
                f" → 买入 {name_of(b.get('symbol', ''))}"
            )
    elif blocked:
        lines.append(f"  >> 订单取消: {blocked.get('reason', '')}")
    else:
        lines.append("  >> 无昨日待执行信号")

    # ═══════════════════════════════════════════════
    #  第二部分：今日新信号 — 今日收盘后计算，明日开盘执行
    # ═══════════════════════════════════════════════
    risk = report.get("risk")
    signal = report.get("signal", "")       # 引擎第7步计算的信号
    signal_target = report.get("signal_target", "")
    signal_note = report.get("signal_note", "")

    lines.append("")
    lines.append("【今日新信号（明日开盘执行）】")

    if risk and risk.get("triggered"):
        # 风控触发时优先显示风控信息
        lines.append(f"  >> ⚠ {risk['reason']}")
    elif state and state.pending_order:
        # 已产生待执行订单 → 明日将执行
        po = state.pending_order
        pa = po.get("action", "?")
        if pa == "buy":
            lines.append(f"  >> 买入信号: {name_of(po['symbol'])}（明日开盘执行）")
        elif pa == "sell":
            lines.append(f"  >> 卖出信号: {name_of(po['symbol'])}（明日开盘执行）")
        elif pa == "switch":
            lines.append(
                f"  >> 切换信号: "
                f"{name_of(po['sell_symbol'])} → {name_of(po['buy_symbol'])}（明日开盘执行）"
            )
        lines.append(f"      原因: {po.get('reason', '')}")
    else:
        # 无 pending_order：信号已计算但未生成待执行订单
        # 先判断是否实际有持仓（优先于 signal 字段，signal 可能被重建脚本设错）
        has_position = state and state.position and state.position.shares > 0
        if has_position:
            h = name_of(state.position.symbol)
            if signal_note:
                lines.append(f"  >> 持有 {h}，无切换需求（{signal_note}）")
            else:
                lines.append(f"  >> 持有 {h}，无切换需求")
        elif signal == "hold_cash":
            lines.append("  >> 无持仓，无买入信号")
        elif signal == "open_pending":
            lines.append(f"  >> 买入信号: {name_of(signal_target)}（今日已执行昨信号，此信号仅供参考）")
        elif signal == "switch_pending":
            lines.append(f"  >> 切换信号: → {name_of(signal_target)}（今日已执行昨信号，此信号仅供参考）")
        else:
            lines.append("  >> 暂无新信号")

    # ── 排名信息（今日动量计算结果，新信号的数据基础） ──
    ranking = report.get("ranking", {})
    if ranking:
        rank_parts = []
        for rk in range(1, min(len(ranking) + 1, max_rank_display + 1)):
            sym = ranking.get(str(rk))
            if sym:
                rank_parts.append(f"#{rk} {name_of(sym)}")
        if rank_parts:
            lines.append(f"      {rank_label}: {' > '.join(rank_parts)}")

    # ═══════════════════════════════════════════════
    #  第三部分：账户日结
    # ═══════════════════════════════════════════════
    lines.append("")
    lines.append("【账户日结】")
    if state:
        pos = state.position
        if pos and pos.shares > 0:
            stock_val = report.get("stock_value", 0)
            lines.append(f"    持仓: {name_of(pos.symbol)} {pos.shares}股  均价{pos.avg_cost:.4f}")
            lines.append(f"    市值: {stock_val:>8.2f}")
        else:
            lines.append("    持仓: 空仓")
        lines.append(f"    现金: {state.cash:>8.2f}")
        total_value = report.get("total_value", 0)
        if state.initial_capital > 0:
            total_return = (total_value / state.initial_capital - 1) * 100
            lines.append(f"    总资产: {total_value:>8.2f}  总收益率: {total_return:+8.2f}%")

    return lines
