"""价差均值回归: ETF偏离等权均值->反向交易"""
import numpy as np
from strategies.momentum_rotation.engine import BacktestEngine
from . import config as cfg

class SpreadReversionEngine(BacktestEngine):
    def __init__(self, **kw):
        super().__init__(initial_capital=kw.get('initial_capital',cfg.INITIAL_CAPITAL),
            risk_mode="A", momentum_window=60, top_n=1, dynamic_window=False)
        self.lb=kw.get('lookback',cfg.LOOKBACK); self.entry_z=kw.get('entry_z',cfg.ENTRY_Z)
        self.exit_z=kw.get('exit_z',cfg.EXIT_Z); self.rb_days=kw.get('rebalance_days',5); self._ds=999
    
    def run(self):
        n=len(self.dates); syms=cfg.ETF_SYMBOLS; C=0.0002; S=0.0001
        for idx in range(n):
            td={sym:self.etf_data[sym].iloc[idx] for sym in syms}
            si=max(0,idx-1)
            if self._ds>=self.rb_days and si>=self.lb+10:
                cum_rets={s: self.etf_data[s].iloc[si]["close"]/self.etf_data[s].iloc[si-self.lb]["close"]-1 for s in syms if si>=self.lb}
                if len(cum_rets)>1:
                    avg=np.mean(list(cum_rets.values())); std=max(np.std(list(cum_rets.values())),0.001)
                    weights={}; tw=0
                    for sym in syms:
                        if sym in cum_rets:
                            z=(cum_rets[sym]-avg)/std
                            w=1.0/(1+abs(z)) if abs(z)>self.entry_z else (1.0 if abs(z)<self.exit_z else 0.5)
                            weights[sym]=w; tw+=w
                    tv=max(1,self.cash+sum(self.positions.get(s,0)*td[s]["close"] for s in syms))
                    for sym in syms:
                        w=weights.get(sym,0)/tw if tw>0 else 0
                        px=td[sym]["open"]*(1+S)
                        target_sh=max(0,int(tv*w/px/100)*100) if w>0 else 0
                        cur=self.positions.get(sym,0); diff=target_sh-cur
                        if diff>0: cost=diff*px*(1+C)
                        if diff>0 and cost<=self.cash: self.cash-=cost; self.positions[sym]=target_sh
                        elif diff<0: self.cash+=abs(diff)*td[sym]["open"]*(1-S)*(1-C); self.positions[sym]=target_sh
                self._ds=0
            sv=sum(self.positions.get(s,0)*td[s]["close"] for s in syms); tv=self.cash+sv
            prev=self.daily_records[-1].total_value if self.daily_records else 10000
            dr=(tv-prev)/prev if prev>0 else 0
            from strategies.momentum_rotation.engine import DailyRecord
            self.daily_records.append(DailyRecord(date=str(self.dates[idx].date()),cash=round(self.cash,2),stock_value=round(sv,2),total_value=round(tv,2),daily_return=round(dr,6),cumulative_return=round(tv/10000-1,6)))
            self._ds+=1
        for sym in list(self.positions.keys()): self.cash+=self.positions[sym]*self.etf_data[sym].iloc[-1]["close"]; del self.positions[sym]
        return self
