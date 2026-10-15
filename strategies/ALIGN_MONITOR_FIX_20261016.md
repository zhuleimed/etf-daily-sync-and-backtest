# 轨迹对齐告警复盘 + 监控口径修复（2026-10-16）

## 0. 一句话结论

2026-10-14 晚的三条"轨迹对齐告警"（composite_momentum / momentum_rotation /
momentum_vol_filter）**不是模拟盘出错，是清零重启留下的路径差**——
模拟盘 10-08 清零后**空仓起步**，对齐监控用的回测却**抱着清零前就持有的旧仓**跑进比对窗口。
修掉监控的这个口径后，6 个被监控策略**全部重回"持仓一致"**。

顺带查出并处置了 composite_momentum 的两处模拟盘接线缺口（W1 已修 / W2 待决策）。

---

## 1. 告警到底是什么（时间线）

| 日期 | 模拟盘 | 回测（对齐监控用的那条） |
|---|---|---|
| 09-30 | （清零前旧历史） | 买入 588000 科创50ETF |
| **10-08** | **【用户决策清零重启】空仓**；收盘后算出 top1=510050 | 继续持有 588000 |
| 10-09 | 开盘买入 510050 3400股 @2.9073 | 当日 `top_etf` **同样是 510050**，但被 `MIN_HOLD_DAYS=10` 挡住没换 |
| 10-12～10-13 | 持有 510050 | 仍持有 588000（同一个 min_hold 挡着） |
| 10-14 | 持有 510050，-0.54% | 588000 触发"极端回撤 15.44%≥15%"强平 → **空仓** |
| 10-15 | 持有 510050，-1.12% | 买入 510050 → **两边一致** |

**关键证据**：回测在 10-09 算出来的第 1 名**也是 510050**（同一信号、同一执行价 2.9072）——
它没买，只是因为手里已经有 588000 且 min_hold 锁着。**同样的规则、不同的起点状态，必然分岔。**

另两条（momentum_rotation / momentum_vol_filter）同根因：模拟盘清零后空仓，
动量得分为负 → "无买入信号"（10-15 实测全部得分为负，最高 510050=-0.048，而空仓开仓要求 >0，
**空仓是正确的**）；回测则一直抱着重启前的 512100。

> 告警原文"回测空仓"只描述了 10-14 那一天——监控只打印**最后一天**的持仓。

---

## 2. 监控侧修复（B020）

`simulation/framework/backtest_align.py`：

1. **清零后"空仓起步"对齐**：识别到 `清零重启/重构` 注释行时，回测改用
   `--trade-from <锚点的下一个交易日>`——预热期只算指标不下单，与模拟盘的空仓起步一一对应。
   为此给 5 个回测引擎（momentum_rotation / momentum_vol_filter / composite_momentum /
   rsi_trend_rotation / adx_trend_rotation，cross_border 复用动量引擎）加了可选参数
   `trade_from`（默认空 = 行为不变）+ 各自的 `run.py --trade-from`。
2. **预热 45 → 90 个交易日**：牛熊判定用 MA60，45 天时 `judge_market_regime` 会因
   `date_idx < ma_period` **恒返回 neutral**（比对窗口内 regime 闸门等于没开）。
3. **"没比对"不再冒充"一致"**：无可对齐记录 / 回测失败 → 单独统计并在结论行点出（原先打印 ✅，是假阴性）。

### 验证（2026-10-16 07:3x 实跑）

```
momentum_rotation     空仓 vs 空仓       ✅      composite_momentum  510050 vs 510050  ✅ 缺口 -0.04pp
momentum_vol_filter   空仓 vs 空仓       ✅      rsi_trend_rotation  空仓 vs 空仓       ✅
adx_trend_rotation    空仓 vs 空仓       ✅      cross_border        513100 vs 513100  ✅ 缺口 -0.00pp
完成：全部持仓一致、无快速漂移 ✅
```

修复前同样 6 个策略里有 3 条在告警（composite 缺口 +1.82pp、momentum 两条 -3~-5.6pp）。

---

## 3. composite_momentum 的两处接线缺口（离线重放实测）

工具：`simulation/analysis/verify_composite_wiring.py`
（真实 DailySimEngine 离线重放；**自检**：现状口径重放 10-08~10-15 与 live CSV 逐日总资产完全一致）。

| | 现状 | 结论 |
|---|---|---|
| **W1** regime 按"数组位置"取 | `judge_market_regime(index_data, today_idx)` 内部用 `iloc[date_idx]`，而 today_idx 是 **ETF 数组位置**；指数窗口 D-140天(93行) 比 ETF 窗口 D-120天(79行) 长 → **判牛熊读的是 14 个交易日前的行情**（2026-10-15 实测读到 09-17；666 天里 279 天判定不同 = 42%） | **已修**：指数按日期对齐到 ETF 日历（同 adx 2026-10-02 的修法）。4 期重放**行为影响 = 0**（regime 只均匀缩放全体得分、不改排序；开仓闸门"得分>0"正数缩放保号；切换阈值 0.03 远离实际 excess 0.1~0.6） |
| **W2** 切换阈值 0.03 | 回测是 `max(截面标准差×0.5, 0.1)` ≈ **0.15~0.25**；模拟盘 `MIN_SWITCH_CONVICTION=0.03`（config 自定义，照抄动量模板）→ **松 5~8 倍** | **待用户决策**（见下表） |

### W2 的 4 期离线重放（纯信号 A 模式，规避风控时序噪声的单变量对照）

| 期间 | 口径 | 与回测持仓一致率 | 末日缺口 |
|---|---|---|---|
| 2024 全年 | 现状 0.03 | 64% | +20.76pp |
| 2024 全年 | **修（σ 口径）** | **66%** | **+10.68pp** |
| 2025 全年 | 现状 0.03 | 55% | **-13.19pp** |
| 2025 全年 | **修（σ 口径）** | **66%** | **+1.02pp** |

风控全开（B，live 配置）下两口径都被"风控时序噪声"淹没（见第 4 节），看不出改善——
这也是**必须用 A 模式做单变量对照**的原因。

---

## 4. 顺带查清：模拟盘 vs 回测的**固有**噪声水平（不是 bug）

同一起点、同一信号下逐日对比，模拟盘与回测的偏离来源：

1. **风控卖出时序**：回测当日卖出、模拟盘 T+1；且风控触发当天引擎**提前 return、跳过信号计算**
   → 回补再晚一天。2024 同口径实测：风控 A 一致率 **64%** / 风控 B 仅 **43%**
   （末日缺口 A +20.8pp vs B -44.5pp）。
2. 回测 3 日渐进调仓 vs 模拟盘一次性切换；回测按 98% 现金买入 vs 模拟盘整手。

⇒ **风控全开(B)的策略天生对不齐**，读对齐告警要按这个噪声水平判断（别把噪声当 bug）。

---

## 5. 回测 B018 同类写法的影响面实测（另外 3 个策略）

工具：`simulation/analysis/audit_regime_alignment.py`（672 天全周期，错位 index vs 按日期对齐 index）。

| 策略 | 持仓不同天数 | regime 列不同天数 | 结论 |
|---|---|---|---|
| composite_momentum | 0 | 414（61%） | 纯外观（本次实测） |
| rsi_trend_rotation | **0** | 478（71%） | 纯外观 |
| macd_trend_rotation | 3 | 351 | 会改交易结果（轻微） |
| **mean_reversion_rotation** | **28** | 270 | **会改交易结果，累计收益最大差 10.22pp** |

（原因同 W1：均匀缩放不改排序、`POSITIVE_SCORE_ONLY` 那道闸是死代码；但 mean_reversion 的市场过滤
`HS300>MA200` 用的是错位后的指数，会真改开平仓时点。）

---

## 6. 复现命令

```bash
# 对齐监控（全部核心策略）
python -m simulation.framework.backtest_align            # 加 --push 推送微信

# 单个策略空仓起步对比
python -m simulation.framework.backtest_align --strategy composite_momentum

# composite 接线缺口：自检 + 4 期对比
python -m strategies.composite_momentum.run --start 2024-01-01 --end 2026-09-30 --tag wirecheck
python -m simulation.analysis.verify_composite_wiring --selfcheck-only
python -m simulation.analysis.verify_composite_wiring

# B018 影响面排查（可指定策略）
python -m simulation.analysis.audit_regime_alignment
python -m simulation.analysis.audit_regime_alignment --strategies mean_reversion_rotation
```

> 附：跨项目通用教训见项目记忆 `align-alert-reset-path-artifact`、
> `composite-sim-regime-threshold-gap`。
