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
        # 沪深300指数（尾部风险判据的数据源）。显式传入优先；
        # 不传时由 load_data 自动接基类已加载的 benchmark_data。
        # 【2026-09-25 修复】原实现硬编码为 None，_check_tail 恒返回 False，
        #   导致"尾部风险"分支在回测中从未触发过，策略退化成纯动量。
        self.hs300_data=kw.get('hs300_data')
        # ── 下面两个开关是 2026-09-25 的排查产物，【均已实测且被否定】──
        # 结论：两者都不优于现状，故默认关闭。保留此实现是为了避免以后
        #       有人重复提出同一假设——先看结论，别再重写一遍。
        # 详见 docs/REWIRE_EXPLAINED_20260925.md 第三节。

        # 假设一「警报解除即归位」：怀疑避险仓被 min_hold_days/切换置信度拖太久
        #   （实测 119 个避险持仓日中 91 天(76%)发生在警报解除之后）。
        #   → 打开后大幅变差：2026 从 +9.01% 掉到 −11.48%（回撤 −17.6%→−30.6%），
        #     全周期 70.25%→30.16%。结论：慢归位其实有保护作用——警报解除≠危险过去。
        self.tail_immediate_exit=kw.get('tail_immediate_exit', False)

        # 假设二「入场冷静期」：怀疑反反复换仓（2024-02 曾 3 天换 3 次）是主要成本。
        #   → 3/5/10 天三档结果完全相同且与现状几乎一致（仅 2024 年 −0.2pp），
        #     说明除那一次外根本没发生过"几天内换另一只避险ETF"。结论：churn 不是成本主因。
        self.tail_entry_cooldown=kw.get('tail_entry_cooldown', 0)
        self._tail_defense=False
        self._defense_days=0

    def load_data(self, start_date: str = "2024-01-01",
                  end_date: str = "", db_path: str = cfg.DB_PATH
                  ) -> "TailRiskEngine":
        """加载行情，并把沪深300基准接到 hs300_data 上（尾部风险判据要用）。"""
        super().load_data(start_date, end_date, db_path)
        if self.hs300_data is None:
            bench = getattr(self, "benchmark_data", None)
            self.hs300_data = bench if bench is not None and not bench.empty else None
        return self
    
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
                # 入场冷静期（可选）：已在避险仓且未满冷静期 → 不因"最稳的那只变了"再换
                if (self.tail_entry_cooldown>0 and self._tail_defense and hp and hsym
                        and self._defense_days<self.tail_entry_cooldown):
                    safe=hsym
                if safe and (not hp or hsym!=safe):
                    if hp and hsym: self._sell_all(idx,td,trade_type="尾部风险切换",symbol=hsym)
                    if safe and self.cash>0: self._buy(safe,self.cash,idx,td,trade_type="买入",reason="尾部风险->低波ETF")
                target=safe
                # 标记"当前仓位是避险仓"，供警报解除后立即归位用
                self._tail_defense=bool(safe) and self._get_hold_symbol()==safe
            else:
                mom=compute_momentum_signals(self.etf_data,si,self.momentum_window)
                ranking=rank_etfs_by_momentum(mom)
                target=ranking.get(1) if len(ranking)>0 else None
                if self.tail_immediate_exit and self._tail_defense and hp and hsym:
                    # 警报解除 → 立即归位（跳过 min_hold_days 与切换置信度）
                    self._tail_defense=False
                    if target and target!=hsym:
                        self._sell_all(idx,td,trade_type="避险归位",symbol=hsym)
                        if self.cash>0:
                            self._buy(target,self.cash,idx,td,trade_type="买入",reason="警报解除->归位")
                        self._days_since_last_switch=0
                elif self.adjustment_days_left<=0: self._make_decision(idx,td,hsym,target,mom)

            # 避险持仓天数计时（供入场冷静期使用）
            self._defense_days = self._defense_days+1 if self._tail_defense else 0

            self._record_day(idx,td,target_etf=target or "")
            self._days_since_last_switch+=1
        self._close_remaining_positions(); return self
