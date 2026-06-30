"""交易成本模块"""
from .config import COMMISSION_RATE, SLIPPAGE, TAX_RATE, MIN_COMMISSION, IMPACT_COST_COEF


def compute_one_way_cost(ta, ma20):
    c = max(ta * COMMISSION_RATE, MIN_COMMISSION); s = ta * SLIPPAGE; i = 0.0
    if ma20 > 0 and ta > 0: i = ta * (ta / ma20 * IMPACT_COST_COEF)
    return c, s, i, c + s + i

def compute_total_friction_cost(fs, ts, sa, ba, ed, di):
    fma = ed[fs].loc[di, "amount_ma20"]; tma = ed[ts].loc[di, "amount_ma20"]
    _, _, _, sc = compute_one_way_cost(sa, fma); _, _, _, bc = compute_one_way_cost(ba, tma)
    return sc + bc, {"sell_cost": round(sc, 2), "buy_cost": round(bc, 2), "total_friction": round(sc + bc, 2)}
