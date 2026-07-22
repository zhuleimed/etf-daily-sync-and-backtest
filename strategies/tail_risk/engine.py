"""尾部风险轮动: 急跌/波动率飙升->切换到最低波动ETF"""
import numpy as np
from strategies.momentum_rotation.engine import BacktestEngine
from strategies.momentum_rotation.momentum_signals import compute_momentum_signals, rank_etfs_by_momentum
from strategies.momentum_rotation.risk import run_all_risk_checks
from . import config as cfg

class TailRiskEngine(BacktestEngine):
    def __init__(self, **kw):
        super().__init__(initial_capital=kw.get('initial_capital',cfg.INITIAL_CAPITAL),
            risk_mode=kw.get('risk_mode',cfg.RISK_MODE), momentum_window=kw.get('momentum_window',cfg.MOMENTUM_WINDOW),
            top_n=1, dynamic_window=False)
        self.tail_threshold=kw.get('tail_threshold',cfg.TAIL_THRESHOLD)
        self.vol_window=kw.get('vol_window',cfg.VOL_WINDOW)
        self.hs300_data=None
    
    def _check_tail(self, idx):
        if self.hs300_data is None or idx<22: return False
        hs=self.hs300_data.set_index("date")["close"]
        ds=str(list(self.etf_data.values())[0].iloc[idx]["date"])[:10]
        if ds not in hs.index: return False
        hi=hs.index.get_loc(ds)
        if isinstance(hi,slice): hi=hi.start
        if hi>=5 and hs.iloc[hi]/hs.iloc[hi-5]-1<self.tail_threshold: return True
        if hi>=20:
            sv=hs.iloc[hi-9:hi+1].pct_change().dropna().std()*np.sqrt(252)
            lv=hs.iloc[hi-19:hi+1].pct_change().dropna().std()*np.sqrt(252)
            if sv>lv*1.5: return True
        return False
    
    def _lowest_vol(self, idx):
        best_sym,best_vol=None,np.inf
        for sym in cfg.ETF_SYMBOLS:
            if sym not in self.etf_data or idx<self.vol_window: continue
            df=self.etf_data[sym]
            rets=df.iloc[idx-self.vol_window+1:idx+1]["pct_chg"]
            vol=rets.std()*np.sqrt(252) if len(rets)>1 else np.inf
            if vol<best_vol: best_vol=vol; best_sym=sym
        return best_sym
    
    def run(self):
        n=len(self.dates); syms=cfg.ETF_SYMBOLS
        for idx in range(n):
            td={sym:self.etf_data[sym].iloc[idx] for sym in syms}
            hp=bool(self.positions); hsym=self._get_hold_symbol(); si=max(0,idx-1)
            if self.risk_mode!="A" and hp and hsym:
                hr=td[hsym]; tv=self._calc_total_value(td)
                self.risk_state.update_peak(hr["high"]); self.risk_state.update_peak_total_value(tv)
                ra,rr=run_all_risk_checks(self.risk_state,tv,hp,hsym,hr["high"],hr["low"],hr["close"],hr["atr"],self.etf_data,idx,mode=self.risk_mode)
                if ra!="none": self._execute_risk_exit(idx,td,ra,rr); self._record_day(idx,td,action_override=ra); continue
            if self.adjustment_days_left>0: self._execute_adjustment_step(idx,td)
            
            if self._check_tail(si):
                safe=self._lowest_vol(si)
                if safe and (not hp or hsym!=safe):
                    if hp and hsym: self._sell_all(idx,td,trade_type="尾部风险切换",symbol=hsym)
                    if safe and self.cash>0: self._buy(safe,self.cash,idx,td,trade_type="买入",reason="尾部风险->低波ETF")
                target=safe
            else:
                mom=compute_momentum_signals(self.etf_data,si,self.momentum_window)
                ranking=rank_etfs_by_momentum(mom)
                target=ranking.get(1) if len(ranking)>0 else None
                if self.adjustment_days_left<=0: self._make_decision(idx,td,hsym,target,mom)
            
            self._record_day(idx,td,target_etf=target or "")
            self._days_since_last_switch+=1
        self._close_remaining_positions(); return self
