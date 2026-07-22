"""中位数动量引擎: 买第N名而非第1名"""
import numpy as np
from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.momentum_signals import compute_momentum_signals, rank_etfs_by_momentum
from strategies.momentum_rotation.risk import run_all_risk_checks
from . import config as cfg

class MedianMomentumEngine(BacktestEngine):
    def __init__(self, **kw):
        super().__init__(initial_capital=kw.get('initial_capital',cfg.INITIAL_CAPITAL),
            risk_mode=kw.get('risk_mode',cfg.RISK_MODE), momentum_window=kw.get('momentum_window',cfg.MOMENTUM_WINDOW),
            top_n=1, dynamic_window=False)
        self.rank_pos=kw.get('rank_position',cfg.RANK_POSITION)
    
    def run(self):
        n=len(self.dates); syms=cfg.ETF_SYMBOLS
        for idx in range(n):
            td={sym:self.etf_data[sym].iloc[idx] for sym in syms}
            hp=bool(self.positions); hsym=self._get_hold_symbol()
            if self.risk_mode!="A" and hp and hsym:
                hr=td[hsym]; tv=self._calc_total_value(td)
                self.risk_state.update_peak(hr["high"]); self.risk_state.update_peak_total_value(tv)
                ra,rr=run_all_risk_checks(self.risk_state,tv,hp,hsym,hr["high"],hr["low"],hr["close"],hr["atr"],self.etf_data,idx,mode=self.risk_mode)
                if ra!="none": self._execute_risk_exit(idx,td,ra,rr); self._record_day(idx,td,action_override=ra); continue
            if self.adjustment_days_left>0: self._execute_adjustment_step(idx,td)
            si=max(0,idx-1); mom=compute_momentum_signals(self.etf_data,si,self.momentum_window)
            ranking=rank_etfs_by_momentum(mom)
            target=ranking.get(self.rank_pos) if len(ranking)>=self.rank_pos else (ranking.get(1) if len(ranking)>0 else None)
            if self.adjustment_days_left<=0: self._make_decision(idx,td,hsym,target,mom)
            self._record_day(idx,td,target_etf=target or "")
            self._days_since_last_switch+=1
        self._close_remaining_positions(); return self
