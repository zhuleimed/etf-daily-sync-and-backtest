"""
模拟交易执行模块

模拟 A 股 ETF 买卖：
  - 买入：现金 ÷ 价格 → 向下取整到 100 股，扣除佣金
  - 卖出：全仓卖出，计算含佣金净收入
  - ETF 免印花税
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import StateManager, TradeRecord


@dataclass
class TradeResult:
    success: bool = True
    shares: int = 0
    price: float = 0.0
    amount: float = 0.0
    commission: float = 0.0
    net_cost: float = 0.0      # 买入总支出 / 卖出净收入
    pnl: float = 0.0           # 本次平仓盈亏
    reason: str = ""


class SimBroker:
    """模拟交易执行器。"""

    def __init__(
        self,
        state_mgr: StateManager,
        commission_rate: float = 0.0002,
        slippage: float = 0.0001,
    ):
        self.state_mgr = state_mgr
        self.commission_rate = commission_rate
        self.slippage = slippage

    def buy(
        self,
        state,
        symbol: str,
        price: float,
        amount: float | None = None,
        reason: str = "买入",
    ) -> TradeResult:
        """买入 ETF。

        价格按收盘价 ×(1+滑点) 计算，股数向下取整到 100 股。

        Args:
            state: SimState 对象（会被修改）。
            symbol: ETF 代码。
            price: 收盘价。
            amount: 投入金额，None=全部现金。
            reason: 交易原因（日志用）。

        Returns:
            TradeResult。
        """
        if amount is None:
            amount = state.cash

        buy_price = price * (1 + self.slippage)
        max_shares = int(amount // buy_price // 100) * 100
        if max_shares <= 0:
            return TradeResult(
                success=False, reason=f"资金不足买入{symbol}（需1手={buy_price*100:.2f}，仅{amount:.2f}）"
            )

        cost = max_shares * buy_price
        commission = max(cost * self.commission_rate, 0.0)
        total_cost = cost + commission

        if total_cost > state.cash:
            # 重新计算：用可用现金全额买
            max_shares = int(state.cash // buy_price // 100) * 100
            if max_shares <= 0:
                return TradeResult(
                    success=False, reason=f"现金不足（{state.cash:.2f}），不够1手")
            cost = max_shares * buy_price
            commission = max(cost * self.commission_rate, 0.0)
            total_cost = cost + commission

        # 更新持仓（若是切换，先清旧仓再开新仓）
        state.position.symbol = symbol
        state.position.shares = max_shares
        state.position.avg_cost = total_cost / max_shares
        state.position.total_cost = total_cost
        state.position.highest_price = buy_price
        state.position.today_opened = True

        # 扣现金
        state.cash -= total_cost
        state.cumulative_cost += commission

        # 记日志
        self.state_mgr.append_trade(state, TradeRecord(
            date=state.last_update,
            action="买入",
            symbol=symbol,
            shares=max_shares,
            price=round(buy_price, 4),
            amount=round(cost, 2),
            commission=round(commission, 2),
            reason=reason,
        ))

        return TradeResult(
            shares=max_shares,
            price=round(buy_price, 4),
            amount=round(cost, 2),
            commission=round(commission, 2),
            net_cost=total_cost,
        )

    def sell(
        self,
        state,
        price: float,
        reason: str = "卖出",
    ) -> TradeResult:
        """卖出当前持仓（全平）。

        Args:
            state: SimState 对象（会被修改）。
            price: 收盘价。
            reason: 交易原因。

        Returns:
            TradeResult。
        """
        if state.position.shares <= 0:
            return TradeResult(success=True, reason="无持仓，无需卖出")

        sell_price = price * (1 - self.slippage)
        revenue = state.position.shares * sell_price
        commission = max(revenue * self.commission_rate, 0.0)
        net_revenue = revenue - commission

        pnl = net_revenue - state.position.total_cost
        state.cash += net_revenue
        state.cumulative_pnl += pnl
        state.cumulative_cost += commission

        # 记日志
        self.state_mgr.append_trade(state, TradeRecord(
            date=state.last_update,
            action="卖出",
            symbol=state.position.symbol,
            shares=state.position.shares,
            price=round(sell_price, 4),
            amount=round(revenue, 2),
            commission=round(commission, 2),
            pnl=round(pnl, 2),
            reason=reason,
        ))

        # 清空持仓
        result = TradeResult(
            shares=state.position.shares,
            price=round(sell_price, 4),
            amount=round(revenue, 2),
            commission=round(commission, 2),
            net_cost=state.position.total_cost,
            pnl=round(pnl, 2),
        )
        state.position = self._empty_position()
        return result

    def _empty_position(self):
        """返回空持仓状态。"""
        from .state import PositionState
        return PositionState()
