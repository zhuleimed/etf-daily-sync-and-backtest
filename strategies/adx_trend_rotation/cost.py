"""交易成本模块"""
from .config import COMMISSION_RATE, SLIPPAGE, TAX_RATE, MIN_COMMISSION, IMPACT_COST_COEF


def compute_one_way_cost(trade_amount: float, amount_ma20: float) -> tuple[float, float, float, float]:
    commission = max(trade_amount * COMMISSION_RATE, MIN_COMMISSION)
    slippage_cost = trade_amount * SLIPPAGE
    impact_cost = 0.0
    if amount_ma20 > 0 and trade_amount > 0:
        impact_cost = trade_amount * (trade_amount / amount_ma20 * IMPACT_COST_COEF)
    total = commission + slippage_cost + impact_cost
    return commission, slippage_cost, impact_cost, total


def compute_total_friction_cost(from_symbol: str, to_symbol: str, sell_amount: float,
                                 buy_amount: float, etf_data: dict, date_idx: int) -> tuple[float, dict]:
    from_ma = etf_data[from_symbol].loc[date_idx, "amount_ma20"]
    to_ma = etf_data[to_symbol].loc[date_idx, "amount_ma20"]
    _, _, _, sc = compute_one_way_cost(sell_amount, from_ma)
    _, _, _, bc = compute_one_way_cost(buy_amount, to_ma)
    return sc + bc, {"sell_cost": round(sc, 2), "buy_cost": round(bc, 2), "total_friction": round(sc + bc, 2)}
