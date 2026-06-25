"""
模拟盘状态持久化模块

以 JSON 文件存储当日持仓/资金/交易记录，使用原子写入防损坏。
每个策略独立一个状态文件，路径格式: simulation/output/state_{策略名}.json
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass
class PositionState:
    """单一标的持仓状态（TOP_N=1，每次仅持一只）。"""
    symbol: str = ""              # ETF 代码
    shares: int = 0               # 持仓股数
    avg_cost: float = 0.0         # 每股平均成本（含佣金）
    total_cost: float = 0.0       # 持仓总成本
    highest_price: float = 0.0    # 持仓期间最高价（移动止盈用）
    today_opened: bool = False    # 今日新开仓（T+1 保护）


@dataclass
class TradeRecord:
    date: str = ""
    action: str = ""              # "买入" | "卖出" | "调仓买入" | "调仓卖出"
    symbol: str = ""
    shares: int = 0
    price: float = 0.0
    amount: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0              # 本次平仓盈亏（买入时为0）
    reason: str = ""


@dataclass
class SimState:
    """完整模拟盘状态。"""
    version: int = 4               # 版本4 新增 total_value
    last_update: str = ""          # YYYY-MM-DD
    cash: float = 0.0
    initial_capital: float = 0.0
    position: PositionState = field(default_factory=PositionState)
    cumulative_pnl: float = 0.0   # 累计已平仓盈亏
    cumulative_cost: float = 0.0  # 累计交易成本
    trade_log: list[dict] = field(default_factory=list)
    strategy_name: str = ""
    days_since_switch: int = 999   # 距上次切换的天数（持久化，防重启丢失）
    peak_value: float = 0.0       # 历史峰值总资产（极端回撤用）
    pending_order: Optional[dict] = None  # 待执行订单，格式见 engine.py
    total_value: float = 0.0      # 当日总资产（最新估值，供 combined 读取）


class StateManager:
    """JSON 状态管理器 — 原子读写。"""

    def __init__(self, output_dir: str | Path, strategy_name: str):
        self.state_path = Path(output_dir) / f"state_{strategy_name}.json"
        self.strategy_name = strategy_name

    # ── 读写 ──

    def load(self) -> SimState | None:
        """加载状态文件，不存在或损坏时返回 None。"""
        if not self.state_path.exists():
            return None
        try:
            with open(self.state_path) as f:
                raw = json.load(f)
            return self._from_dict(raw)
        except (json.JSONDecodeError, OSError, KeyError):
            return None

    def save(self, state: SimState) -> None:
        """原子写入 JSON。"""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            suffix=".json",
            prefix=f"state_{self.strategy_name}_",
            dir=self.state_path.parent,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._to_dict(state), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ── 初始化 ──

    def init_new(self, initial_capital: float) -> SimState:
        """创建新的模拟盘状态。"""
        state = SimState(
            cash=initial_capital,
            initial_capital=initial_capital,
            strategy_name=self.strategy_name,
        )
        self.save(state)
        return state

    # ── 序列化 ──

    def _to_dict(self, state: SimState) -> dict:
        return {
            "version": state.version,
            "last_update": state.last_update,
            "cash": state.cash,
            "initial_capital": state.initial_capital,
            "position": asdict(state.position),
            "cumulative_pnl": state.cumulative_pnl,
            "cumulative_cost": state.cumulative_cost,
            "trade_log": state.trade_log,
            "strategy_name": state.strategy_name,
            "days_since_switch": state.days_since_switch,
            "peak_value": state.peak_value,
            "pending_order": state.pending_order,
            "total_value": state.total_value,
        }

    def _from_dict(self, raw: dict) -> SimState:
        pos = raw.get("position", {})
        order = raw.get("pending_order")
        return SimState(
            version=raw.get("version", 1),
            last_update=raw.get("last_update", ""),
            cash=raw.get("cash", 0.0),
            initial_capital=raw.get("initial_capital", 0.0),
            position=PositionState(
                symbol=pos.get("symbol", ""),
                shares=pos.get("shares", 0),
                avg_cost=pos.get("avg_cost", 0.0),
                total_cost=pos.get("total_cost", 0.0),
                highest_price=pos.get("highest_price", 0.0),
                today_opened=pos.get("today_opened", False),
            ),
            cumulative_pnl=raw.get("cumulative_pnl", 0.0),
            cumulative_cost=raw.get("cumulative_cost", 0.0),
            trade_log=raw.get("trade_log", []),
            strategy_name=raw.get("strategy_name", ""),
            days_since_switch=raw.get("days_since_switch", 999),
            peak_value=raw.get("peak_value", 0.0),
            pending_order=order,
            total_value=raw.get("total_value", 0.0),
        )

    def append_trade(self, state: SimState, trade: TradeRecord) -> None:
        """追加交易记录到日志（最多保留 100 条）。"""
        state.trade_log.append(asdict(trade))
        if len(state.trade_log) > 100:
            state.trade_log = state.trade_log[-100:]
