"""布林带均值回归: 买入触及下轨(PctB<阈值)的ETF,等权持有"""
import numpy as np
from strategies.momentum_rotation.engine import BacktestEngine
from . import config as cfg

class BollingerReversionEngine(BacktestEngine):
    def __init__(self, **kw):
        super().__init__(initial_capital=kw.get('initial_capital',cfg.INITIAL_CAPITAL),
            risk_mode="A", momentum_window=60, top_n=1, dynamic_window=False)
        self.bb_period=kw.get('bb_period',cfg.BB_PERIOD)
        self.bb_std=kw.get('bb_std',cfg.BB_STD)
        self.pctb_th=kw.get('pctb_threshold',cfg.PctB_THRESHOLD)
        self.rb_days=kw.get('rebalance_days',5); self._ds=999
    
    def _pct_b(self, df, idx):
        if idx<self.bb_period: return 0.5
        closes=df.iloc[idx-self.bb_period+1:idx+1]["close"]
        ma=closes.mean(); s=closes.std()
        upper=ma+self.bb_std*s; lower=ma-self.bb_std*s
        return (df.iloc[idx]["close"]-lower)/(upper-lower) if upper-lower>0 else 0.5
    
    def run(self):
        n=len(self.dates); syms=cfg.ETF_SYMBOLS; C=0.0002; S=0.0001
        for idx in range(n):
            td={sym:self.etf_data[sym].iloc[idx] for sym in syms}
            si=max(0,idx-1)
            if self._ds>=self.rb_days and si>=self.bb_period:
                buys=[(sym,self._pct_b(self.etf_data[sym],si)) for sym in syms if self._pct_b(self.etf_data[sym],si)<self.pctb_th]
                tv=max(1,self.cash+sum(self.positions.get(s,0)*td[s]["close"] for s in syms))
                if buys:
                    weights={s:max(0,self.pctb_th-pctb) for s,pctb in buys}; tw=sum(weights.values())
                    for sym in syms:
                        w=weights.get(sym,0)/tw if tw>0 else 0
                        px=td[sym]["open"]*(1+S)
                        target_sh=max(0,int(tv*w/px/100)*100) if w>0 else 0
                        cur=self.positions.get(sym,0); diff=target_sh-cur
                        if diff>0: cost=diff*px*(1+C); 
                        if diff>0 and cost<=self.cash: self.cash-=cost; self.positions[sym]=target_sh
                        elif diff<0: self.cash+=abs(diff)*td[sym]["open"]*(1-S)*(1-C); self.positions[sym]=target_sh
                else:
                    for sym in list(self.positions.keys()):
                        sh=self.positions.get(sym,0)
                        if sh>0: self.cash+=sh*td[sym]["open"]*(1-S)*(1-C); del self.positions[sym]
                self._ds=0
            sv=sum(self.positions.get(s,0)*td[s]["close"] for s in syms); tv=self.cash+sv
            prev=self.daily_records[-1].total_value if self.daily_records else 10000
            dr=(tv-prev)/prev if prev>0 else 0
            from strategies.momentum_rotation.engine import DailyRecord
            self.daily_records.append(DailyRecord(date=str(self.dates[idx].date()),cash=round(self.cash,2),stock_value=round(sv,2),total_value=round(tv,2),daily_return=round(dr,6),cumulative_return=round(tv/10000-1,6)))
            self._ds+=1
        for sym in list(self.positions.keys()): self.cash+=self.positions[sym]*self.etf_data[sym].iloc[-1]["close"]; del self.positions[sym]
        return self
