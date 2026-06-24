# 019 ETF 量化策略 — 数据同步 · 回测框架 · 模拟运行

> A 股 ETF 量化交易研究项目。覆盖数据获取、策略回测、模拟盘运行全流程。
>
> **项目入口：** `pipeline.py`（cron 交易日 20:00 触发）
> **策略开发：** `strategies/` 下建子目录 → 回测 → `simulation/` 下建对应模拟盘
> **纪律红线（详见第2节）：** 回测信号用 `close[T-1]`，执行用 `open[T]`；模拟盘用 `close[T]` 信号 → `open[T+1]` 执行

---

## 目录

1. [项目结构](#1-项目结构)
2. [核心交易逻辑（纪律红线）](#2-核心交易逻辑纪律红线)
3. [数据同步模块](#3-数据同步模块)
4. [数据库](#4-数据库)
5. [回测框架](#5-回测框架)
6. [回测引擎核心机制详解](#6-回测引擎核心机制详解)
7. [13 种策略一览](#7-13-种策略一览)
8. [策略对比排行](#8-策略对比排行)
9. [模拟盘框架（T+1 待执行订单）](#9-模拟盘框架t1-待执行订单)
10. [管线编排器 pipeline](#10-管线编排器-pipeline)
11. [策略开发指南](#11-策略开发指南)
12. [配置详解](#12-配置详解)
13. [运行指南](#13-运行指南)
14. [已知 Bug 与修复记录](#14-已知-bug-与修复记录)
15. [常见问题](#15-常见问题)

---

## 1. 项目结构

```
019_etf_daily_sync_and_backtest/
│
├── pipeline.py                  # 管线编排器（cron 入口）
├── pipeline_status.py           # 管线状态追踪与推送
│
├── main.py                      # 数据同步入口（6种运行模式）
├── etf_sync/                    # ══ 数据同步模块 ══
│   ├── config.py                # pydantic-settings 配置管理
│   ├── data_source.py           # 腾讯→新浪 双轨制自动切换
│   ├── engine.py                # SQLite 数据库引擎
│   ├── sync.py                  # 同步管理器（交易日/时间门控）
│   ├── logger.py                # rich 日志
│   └── notify.py                # WxPusher 推送
│
├── strategies/                  # ══ 回测策略（13个）══
│   ├── STRATEGY_COMPARISON.md   # 策略对比分析文档（详细）
│   ├── momentum_rotation/       # ① 纯动量轮动（基准）
│   ├── momentum_vol_filter/     # ② 波动率过滤轮动 ✅ 最优
│   ├── momentum_ma_filter/      # ③ 大盘MA250均线过滤
│   ├── momentum_ma_etf/         # ④ 逐ETF均线过滤
│   ├── momentum_dual/           # ⑤ 双动量（绝对+相对）
│   ├── dual_ma_crossover/       # ⑥ 双均线交叉 MA(60,120)
│   ├── low_vol_rotation/        # ⑦ 低波动率轮动
│   ├── mean_reversion/          # ⑧ 均值回归轮动
│   ├── vol_price_momentum/      # ⑨ 量价配合轮动
│   ├── donchian_breakout/       # ⑩ 唐奇安通道突破
│   ├── bollinger_rotation/      # ⑪ 布林带轮动
│   ├── pair_trading/            # ⑫ 配对交易（市场中性）
│   ├── combined/                # ⑬ 组合策略（动量80%+配对20%）
│   ├── market_regime_rotation/  # （实验）市场状态识别——已放弃
│   └── 各策略目录包含：
│       ├── config.py            # 策略参数（ETF池、动量窗口等）
│       ├── engine.py            # 回测引擎（BacktestEngine 类）
│       ├── data.py              # 数据加载（SQLite → DataFrame）
│       ├── momentum_signals.py  # 信号计算函数（非动量策略也有）
│       ├── cost.py              # 交易成本计算（佣金/滑点/冲击）
│       ├── risk.py              # 风控模块（A/B/C 三模式）
│       ├── metrics.py           # 绩效指标计算（MetricsCalculator）
│       ├── reporter.py          # 报告生成 + matplotlib 图表
│       └── run.py               # 入口脚本（argparse）
│
├── simulation/                  # ══ 模拟盘框架 ══
│   ├── framework/               # 通用模板块（与策略无关）
│   │   ├── state.py             # JSON 原子持久化（StateManager）
│   │   ├── data.py              # 从 etf_daily.db 加载数据
│   │   ├── broker.py            # 模拟交易执行（SimBroker）
│   │   ├── engine.py            # T+1 每日流程编排（DailySimEngine）
│   │   ├── risk.py              # 止损/止盈/极端回撤
│   │   └── notify.py            # WxPusher 推送
│   └── strategies/              # 各策略模拟盘入口
│       └── momentum_rotation/
│           ├── config.py        # 模拟盘配置
│           └── daily.py         # 每日运行入口
│
├── data/etf_daily.db            # SQLite 数据库（gitignore）
├── logs/                        # 日志（gitignore）
├── .env                         # WxPusher Token 等
├── .gitignore
└── README.md                    # ← 本文档
```

---

## 2. 核心交易逻辑（纪律红线）

### 2.1 为什么有这个规则？

这是整个项目的**纪律红线**，源于一次修正：

回测引擎最初使用 `close[T]` 同时计算动量信号和交易执行。这意味着在 bar T 上，
你"看到了 T 日的收盘价，然后用这个收盘价成交"。在真实交易中，收盘后市场已关闭，
你无法以收盘价成交。这是典型的 **look-ahead bias**，会导致回测收益虚高。

修正后，所有引擎遵循**信号与执行严格分离**的原则。

### 2.2 回测引擎时序

```
T-1 日收盘（已知数据）:
  close[T-1] ← 信号数据截止于此
  open[T-1], high[T-1], low[T-1], volume[T-1]

T 日开盘（执行交易）:
  open[T] ← 用此价格买卖（不是 close[T]！）
  检查：涨停（不能买入）/ 跌停（不能卖出）

总结: close[T-1] → 信号 | open[T] → 执行
```

**代码层面：**

```python
# ✅ 回测引擎 _buy() 方法
price = today_data[symbol]["open"] * (1 + SLIPPAGE)

# ✅ 回测引擎 _sell() 方法
sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)

# ✅ 信号计算索引（前移 1 根 bar）
signal_idx = max(1, idx - 1)
momentum = compute_momentum_signals(self.etf_data, signal_idx, ...)
```

**绝对禁止（代码审查红线）：**
- ❌ `price = today_data[symbol]["close"] * (1 + SLIPPAGE)` — 禁止用收盘价执行
- ❌ `compute_momentum_signals(self.etf_data, idx, ...)` — 禁止用当日数据算信号
- ❌ 任何在前一根 bar 未知的数据

### 2.3 模拟盘引擎时序

```
T 日 20:00（数据同步后）:
  close[T] → 计算信号
  如信号触发 → 创建 "待执行订单"（不交易！）
  存入 JSON 状态文件

T+1 日 20:00:
  读取昨日待执行订单
  open[T+1] → 执行（不是 close[T+1]！）
  检查涨停/跌停：涨停无法买入，跌停无法卖出
  用 close[T+1] 计算新信号 → 创建新待执行订单
```

### 2.4 两引擎对比

| 引擎 | 信号时间 | 执行时间 | 执行价格 | 服务端目录 |
|------|---------|---------|---------|-----------|
| **回测** `strategies/` | `close[T-1]` | `open[T]` | 开盘价 ± 滑点 | 各策略目录下 |
| **模拟盘** `simulation/` | `close[T]` | `open[T+1]` | 开盘价 ± 滑点 | `simulation/` 下 |

**数学上等价：** 两个引擎的信号都比执行提前约 1 个 bar。区别在于回测是逐日回放历史，
模拟盘是逐日增量运行。

### 2.5 涨跌停规则

| ETF 类型 | 涨跌停限制 | 代码 |
|---------|-----------|------|
| 普通 ETF（510xxx, 512xxx, 513xxx 等） | ±10% | `limit_pct = 0.10` |
| 创业板 ETF（159915, 159949 等） | ±20% | `limit_pct = 0.20` |
| 科创板 ETF（588000, 588080 等） | ±20% | `limit_pct = 0.20` |

**处理逻辑：** 当开盘价触及涨停价时，买入订单**取消**（无成交）。
当开盘价触及跌停价时，卖出订单**取消**。切换订单（同时卖A买B）需双边均通过检查，
任一方被封锁则全部取消。

---

## 3. 数据同步模块

### 3.1 数据源

| 数据类型 | 主源 | 备选 | 说明 |
|---------|:----:|:----:|------|
| ETF 日线 | **腾讯** `web.ifzq.gtimg.cn` | **新浪** `quotes.sina.cn` | 双轨制自动切换 |
| 指数日线 | **新浪** `quotes.sina.cn` | — | 腾讯不支持指数K线 |
| ETF 列表 | **akshare** `fund_etf_spot_em()` | — | ~1500+只 |

### 3.2 同步标的

**ETF：** 全量场内 ETF（沪深两市所有上市ETF，约1500+只）

**指数：** 上证50(`000016`)、沪深300(`000300`)、中证500(`000905`)、中证1000(`000852`)

### 3.3 运行模式

```bash
python main.py                     # 标准模式（20:00后执行）
python main.py --sync-only         # 仅同步ETF日线+指数（跳过ETF列表更新）
python main.py --force             # 跳过交易日/时间门控
python main.py --backfill          # 全量回填历史
python main.py --list-only         # 仅更新ETF列表
```

### 3.4 时间门控

`sync_after_hour = 20`，`sync_after_minute = 0` — 北京时间 **20:00** 后才允许同步。
这是为了确保当日行情数据已全部发布。

### 3.5 双轨制（Tencent → Sina）

1. 默认使用 **Tencent** 接口（前复权日K线）
2. Tencent 连续失败 → 自动切换到 **Sina**
3. 每 50 次 Sina 请求尝试恢复 Tencent
4. 日志和推送消息中标记当前活跃数据源
5. 接口转换：纯数字代码 → `sh510050` / `sz159915`（腾讯），
   `sh510050` / `sz159915`（新浪）

### 3.6 交易日判断

```python
def is_trade_day(check_date):
    if check_date.weekday() >= 5:  # 周末
        return False
    from chinese_calendar import is_workday
    return is_workday(check_date)   # 法定节假日
```

使用 `chinese_calendar` 库判断春节、国庆、清明等节假日。

---

## 4. 数据库

SQLite 单文件数据库，默认路径 `data/etf_daily.db`。

### 4.1 etf_daily 表（ETF 日线）

```sql
CREATE TABLE etf_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,      -- ETF代码，纯数字 "510050"
    date TEXT NOT NULL,        -- 交易日 "2024-01-02"
    open REAL,                 -- 开盘价
    high REAL,                 -- 最高价
    low REAL,                  -- 最低价
    close REAL,                -- 收盘价
    volume REAL                -- 成交量（份）
);
```

### 4.2 index_daily 表（指数日线）

同上结构，`symbol` 为指数代码 `"000300"`。

### 4.3 etf_list 表（ETF 列表）

```sql
CREATE TABLE etf_list (
    symbol TEXT PRIMARY KEY,   -- ETF代码
    name TEXT,                 -- ETF名称
    delisted_date TEXT         -- 退市日期（NULL=正常交易）
);
```

### 4.4 sync_log 表（同步日志）

```sql
CREATE TABLE sync_log (
    date TEXT PRIMARY KEY,     -- 交易日
    status TEXT,               -- success / failed / skipped
    etf_count INTEGER,         -- 同步ETF只数
    new_listed INTEGER,        -- 新增上市
    delisted INTEGER,          -- 退市
    index_count INTEGER,       -- 同步指数个数
    duration_seconds REAL,     -- 耗时
    tencent_count INTEGER,     -- 腾讯源成功数
    sina_count INTEGER,        -- 新浪源成功数
    error_msg TEXT             -- 错误信息
);
```

### 4.5 数据加载（回测用）

```python
from strategies.momentum_rotation.data import load_all_etf_data

etf_data, common_dates = load_all_etf_data(
    symbols=["510050", "510300", ...],  # ETF代码列表
    start_date="2024-01-01",            # 回测开始日期
    end_date="",                        # 空=不限制
    momentum_window=20,                 # 动量窗口（预计算）
)

# etf_data: {"510050": DataFrame, ...}
# 每个 DataFrame 包含列：
# date, open, high, low, close, volume,
# pct_chg, cumulative_returns, amount, 
# amount_ma20, atr, momentum, momentum_10, momentum_20

# common_dates: DatetimeIndex（所有ETF的共同交易日，过滤后的）
```

### 4.6 数据加载（模拟盘用）

```python
from simulation.framework.data import load_latest_data

etf_data = load_latest_data(
    symbols=["510050", ...],
    lookback_days=40,          # 加载最近N个自然日
    momentum_window=20,        # 动量计算窗口
)
```

---

## 5. 回测框架

### 5.1 每个策略的目录结构

```
strategies/策略名/
  ├── config.py         # 参数配置
  ├── engine.py         # 回测引擎（BacktestEngine 类）
  ├── data.py           # 数据加载函数
  ├── momentum_signals.py  # 信号计算函数（compute/rank）
  ├── cost.py           # 交易成本计算
  ├── risk.py           # 风控模块
  ├── metrics.py        # 绩效指标计算
  ├── reporter.py       # 报告+图表生成
  └── run.py            # CLI入口
```

### 5.2 引擎逐日流程

```python
class BacktestEngine:
    def run(self):
        for idx in range(n):    # 遍历所有交易日
            # 1. 获取当日OHLCV
            today_data = {sym: self.etf_data[sym].iloc[idx] for sym in ETF_SYMBOLS}
            
            # 2. 风控检查（仅B/C模式）
            if self.risk_mode != "A":
                risk_action, risk_reason = run_all_risk_checks(...)
                if risk_action != "none":
                    self._execute_risk_exit(...)
                    continue        # 跳过后续步骤
            
            # 3. 渐进调仓
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)
            
            # 4. 计算信号（用 idx-1 避免 look-ahead！）
            signal_idx = max(0, idx - 1)
            momentum = compute_momentum_signals(self.etf_data, signal_idx, ...)
            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1)
            
            # 5. 决策执行（用 open[T]）
            self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)
            
            # 6. 记录当日状态
            self._record_day(idx, today_data, ...)
```

### 5.3 风控模式（RISK_MODE）

| 模式 | 说明 | 启用检查项 |
|------|------|-----------|
| **A** | 纯信号 — 完全无风控 | 无 |
| **B** | 全开 — 止损+止盈+极端回撤 | 固定比例止损(5%), 移动止盈(10%进入/5%回撤), ATR止损(4×), 极端回撤(15%) |
| **C** | 兜底 — 仅极端回撤 | 极端回撤(15%) |

**历史测试结论：** 对动量策略，风控反而降低收益（打断了趋势持有）。默认用 A。

### 5.4 输出报告

每次回测生成至 `strategies/策略名/output/YYYYMMDD_HHMMSS_标记/`：

| 文件 | 说明 |
|------|------|
| `daily_records.csv` | 逐日账户状态（持仓/资金/收益） |
| `trade_records.csv` | 交易明细（日期/标的/价格/盈亏） |
| `metrics.csv` | 绩效指标汇总 |
| `equity_curve.png` | 净值曲线图（含基准对比） |
| `drawdown.png` | 回撤曲线图 |
| `holding_heatmap.png` | 持仓分布热力图 |
| `monthly_returns.png` | 月度收益热力图 |

---

## 6. 回测引擎核心机制详解

### 6.1 FIFO 成本核算

买入时创建 `BuyLot` 记录（日期/代码/股数/价格/总成本），
卖出时按 **先进先出（FIFO）** 匹配最早买入批次，计算每笔盈亏和持仓天数。

```python
@dataclass
class BuyLot:
    date: str = ""
    symbol: str = ""
    shares: int = 0
    price: float = 0.0
    total_cost: float = 0.0     # 含佣金
```

### 6.2 渐进调仓（ADJUSTMENT_DAYS）

当需要切换 ETF 时（如从 510050 切换到 159915），
不是一天内全部完成，而是分 N 天逐步卖出旧、买入新：

```
第1天：卖出 1/5 旧仓位 → 立即买入新标的（用卖出所得现金）
第2天：卖出 1/5 旧仓位 → 买入新标的
...（直到全部切换完成）
```

调仓期间，动量信号仍然计算，但决策步骤跳过（`adjustment_days_left > 0` 时不触发新切换）。
参数 `ADJUSTMENT_DAYS = 5`，可在 config 中调整。波动率过滤策略建议 `ADJUSTMENT_DAYS = 3`。

### 6.3 切换置信度（MIN_SWITCH_CONVICTION = 3%）

动量差距必须超过此阈值才执行切换。这是叠加在摩擦成本之上的额外条件：

```python
excess = target_mom - current_mom          # 动量差距
friction_ratio = friction / trade_amount   # 摩擦成本占比
switch_threshold = max(friction_ratio, MIN_SWITCH_CONVICTION)
if excess > switch_threshold:
    # 执行切换
```

3% 的置信度意味着目标 ETF 的动量必须比当前持仓高出至少 3 个百分点才切换，
有效过滤掉普涨市中的噪声切换。

### 6.4 最小持仓天数（MIN_HOLD_DAYS = 10）

切换后至少持有 10 个交易日才能再次切换。防止动量排名频繁变动导致反复买卖。
`_days_since_switch` 计数器持久化在模拟盘状态文件中。

### 6.5 短期动量确认（SHORT_TERM_MOMENTUM_CHECK）

切换时检查目标 ETF 的短期趋势是否确认：

```python
# 5日动量不能为负（近期在涨）
tgt_5d = close[T-1] / close[T-6] - 1   # 用 idx-1 避免 look-ahead
if tgt_5d <= -0.005:
    return  # 不切换

# 动量减速检查：短期日均涨幅 < 中期日均涨幅 → 动能衰减
tgt_15d = momentum_series.get(target_etf, nan)
if tgt_15d > 0 and tgt_5d / 5 < tgt_15d / 15:
    return  # 不切换（虽然还在涨，但速度慢了）
```

### 6.6 交易成本模型

| 项目 | 费率 | 说明 |
|------|------|------|
| 佣金 | 0.02%（万分之二） | 双边，买卖都收 |
| 滑点 | 0.01%（万分之一） | 买卖方向各加/减 |
| 印花税 | 0% | ETF 免收 |
| 最低佣金 | 无 | ETF 免最低5元限制 |
| 冲击成本 | 系数 0.1 | = 调仓金额 × 0.1 ÷ 过去20日均成交额 |

### 6.7 绩效指标（MetricsCalculator）

| 指标 | 计算公式 |
|------|---------|
| 累计收益率 | `(最终资产 / 初始资产) - 1` |
| 年化收益率 | `(1 + 累计收益)^(252/交易日数) - 1` |
| 最大回撤 | `min(total_value / cummax(total_value) - 1)` |
| 夏普比率 | `mean(daily_return) / std(daily_return) × √252`（无风险利率3%） |
| Sortino比率 | `mean(daily_return - rf) / std(负收益率) × √252` |
| Calmar比率 | `年化收益率 / |最大回撤|` |
| 日胜率 | `正收益天数 / 总交易天数` |
| 交易胜率 | `盈利交易次数 / 总交易次数` |
| 盈亏比 | `平均盈利 / 平均亏损` |

### 6.8 等权基准收益

```python
def compute_equal_weight_benchmark(etf_data):
    # 每日各ETF日收益率的算术平均 = 组合日收益率
    # 累计 = (1 + 组合日收益率).cumprod()
    # 7只ETF各占 1/7
```

---

## 7. 13 种策略一览

### 7.1 momentum_rotation — 纯动量轮动（基准）

```
动量 = close[T-1] / close[T-21] - 1     # 20日收益率
排名 → 全仓第1名
```

**参数：** MOMENTUM_WINDOW=20, MIN_SWITCH_CONVICTION=3%, MIN_HOLD_DAYS=10  
**结果：** +133%, 夏普 1.19

### 7.2 momentum_vol_filter — 波动率过滤轮动 ✅ 最优

```
年化波动率 = std(日收益率, 20d) × √252
if 年化波动率 > 30%:  空仓
else:                 正常动量轮动
```

**参数：** VOL_THRESHOLD=0.30, VOL_WINDOW=20, ADJUSTMENT_DAYS=3  
**结果：** +117%, 夏普 **1.31**（夏普最优）  
**特点：** 唯一一个在所有维度均超越纯动量的过滤器。高波动时离场，低波动时全力轮动。

### 7.3 momentum_ma_filter — 大盘均线过滤

```
if HS300_close > MA(HS300, 250):  正常动量轮动
else:                              全部空仓
```

**参数：** MA_FILTER_PERIOD=250  
**结果：** +167%, 夏普 1.37  
**特点：** 年线过滤在牛市中几乎从不触发，等同纯动量。熊市中理论上能保护，但数据周期内无法验证。

### 7.4 momentum_ma_etf — 逐ETF均线过滤

```
for each ETF:
  if close[ETF] > MA(ETF, 60):  进入候选池
  if close[ETF] ≤ MA(ETF, 60):  排除/卖出
候选池中按动量排名
```

**参数：** MA_FILTER_PERIOD=60  
**结果：** +191%, 夏普 1.50  
**特点：** 收益最高，但过度依赖于2024-2026牛市周期。在震荡市中可能大幅回撤。

### 7.5 momentum_dual — 双动量（绝对+相对）

```
if 持仓ETF动量 ≤ 0:  卖出空仓
if 目标ETF动量 ≤ 0:  不切换
if 目标动量 > 0 AND 排名第1:  持有
```

**结果：** +86%, 夏普 0.94  
**结论：** 绝对动量>0条件在正常回调中频繁触发清仓，效果不佳。

### 7.6 dual_ma_crossover — 双均线交叉

```
for each ETF:
  if MA_fast(ETF) > MA_slow(ETF):  上升趋势 → 买入持有
  if MA_fast(ETF) ≤ MA_slow(ETF):  下降趋势 → 卖出
所有上升趋势ETF等权持有
```

**最佳组合：** MA(60,120)，+45%, 夏普 0.81  
**结论：** 均线交叉天然滞后，买入时趋势已过半，卖出时已跌了一段。

### 7.7 low_vol_rotation — 低波动率轮动

```
持有过去20日年化波动率最低的ETF（低波动异象）
```

**结果：** +32%, 夏普 0.58  
**结论：** 在牛市中低波动=低收益，错过了大部分涨幅。

### 7.8 mean_reversion — 均值回归轮动

```
乖离率 = (close - MA(close, 20)) / MA(close, 20)
排名 → 持有乖离率最低（跌最狠）的ETF
```

**结果：** +80%, 切换136次, 夏普 0.94  
**结论：** 在趋势市中反复超跌反弹被打脸，切换频率高。

### 7.9 vol_price_momentum — 量价配合轮动

```
信号 = 动量 × (成交量 / 成交量MA)
放量上涨才追，缩量上涨不碰
```

**结果：** +55%, 夏普 0.71  
**结论：** 成交量条件过于严格，减少了有效信号。

### 7.10 donchian_breakout — 唐奇安通道突破

```
upper = max(high[-60:])    # 60日最高价
lower = min(low[-60:])     # 60日最低价
if close > upper:  → 买入
if close < lower:  → 卖出
```

**结果：** +67%, 夏普 0.85  
**结论：** 突破策略在趋势市有效，但震荡市中假突破多。

### 7.11 bollinger_rotation — 布林带轮动

```
position = (close - lower) / (upper - lower)
靠近下轨(0) → 超卖 → 买入信号
靠近上轨(1) → 超买 → 卖出信号
```

**结果：** +108%, 但切换217次(成本12.5%), 夏普 1.10  
**结论：** 夏普不错但换手率过高，需加最小持仓天数限制。

### 7.12 pair_trading — 配对交易（市场中性）

```
for each 配对 (大盘ETF, 成长ETF):
  spread = log(price_a / price_b)
  z-score = (spread - mean) / std
  |z| > 2.0:  价差过大 → 开仓（多空对冲）
  |z| < 0.3:  价差回归 → 平仓获利
  |z| > 3.0:  价差继续发散 → 止损
```

**三对组合：** 上证50↔创业板 + 沪深300↔创业板 + 上证50↔科创50  
**结果：** +25%, 回撤仅 **6%**, 市场中性  
**优化：** 对数比价(`log(price_a/price_b)`) + 成交量过滤 + 自适应z-score阈值  
**年度验证：**

| 年份 | 收益 | 最大回撤 | 市场环境 |
|------|------|---------|---------|
| 2023(3月起) | -0.16% | 2.83% | 熊市（HS300跌-16.7%） |
| 2024 | +6.80% | 3.16% | 牛市 |
| 2025 | +10.25% | 4.80% | 牛市 |
| 2026(半年) | +11.40% | 3.31% | 牛市 |

### 7.13 combined — 组合策略（动量80% + 配对20%）

分别运行两个子引擎后合并每日净值：
- 动量引擎（80%资金）：主攻收益
- 配对引擎（20%资金）：降低回撤

**结果：** +108%, 夏普 1.11, 回撤 24%  
**测试权重：**

| 动量% | 配对% | 收益 | 回撤 | 夏普 |
|-------|-------|------|------|------|
| 100% | 0% | +133% | 26% | 1.19 |
| **80%** | **20%** | **+108%** | **24%** | **1.11** |
| 70% | 30% | +95% | 28% | 0.99 |
| 50% | 50% | +70% | 31% | 0.72 |

### 7.14 market_regime_rotation — [已放弃] 市场状态识别

在 `momentum_rotation` 基础上增加 BULL/BEAR 状态识别：
- BULL → 锁定持有
- BEAR → 正常轮动

**结果：** +39%, 夏普 0.75  
**放弃原因：** BULL 锁仓在牛市中反而阻止了切换到更强的ETF，画蛇添足。
动量轮动本身已隐含趋势跟踪能力。

---

## 8. 策略对比排行

（以 `open[T]` 价格执行为准，`close[T-1]` 信号）

| # | 策略 | 总收益 | 夏普 | 最大回撤 | 切换 | 形态 |
|---|------|--------|------|---------|------|------|
| 1 | ma_etf 逐ETF均线 | **+191%** | **1.50** | -26% | — | 趋势 |
| 2 | ma_filter MA=250 | **+167%** | **1.37** | -26% | — | 趋势 |
| 3 | rotation 纯动量 | **+133%** | **1.19** | -26% | 19 | 趋势 |
| 4 | **vol_filter 波动率过滤** | **+117%** | **1.31** | **-23%** | **19** | **趋势✅** |
| 5 | combined 动量+配对 | +108% | 1.11 | -24% | — | 混合 |
| 6 | bollinger 布林带 | +108% | 1.10 | -20% | 217 | 震荡 |
| 7 | dual 双动量 | +86% | 0.94 | -28% | 33 | 趋势 |
| 8 | mean_reversion 均值回归 | +80% | 0.94 | -17% | 136 | 震荡 |
| 9 | donchian 通道突破 | +67% | 0.85 | -21% | 108 | 趋势 |
| 10 | vol_price 量价 | +55% | 0.71 | -21% | 181 | 趋势 |
| 11 | crossover 双均线 | +45% | 0.81 | -14% | 94 | 趋势 |
| 12 | low_vol 低波动率 | +32% | 0.58 | -18% | 50 | 防御 |
| 13 | **pair_trading 配对交易** | **+25%** | **—** | **-6%** | **42** | **市场中性** |

> 完整分析见 `strategies/STRATEGY_COMPARISON.md`

---

## 9. 模拟盘框架（T+1 待执行订单）

### 9.1 框架定位

`simulation/framework/` 是与策略无关的通用组件，任何 ETF 策略的模拟盘都能复用。
`simulation/strategies/` 下是各策略的适配层，调用回测的信号逻辑 + framework 执行交易。

### 9.2 模块职责

| 模块 | 类/函数 | 职责 |
|------|---------|------|
| **state.py** | `StateManager`, `SimState` | JSON 原子读写（`tempfile.mkstemp` + `os.replace`） |
| **data.py** | `load_latest_data()` | 从 `etf_daily.db` 加载最近 N 日行情 |
| **broker.py** | `SimBroker` | 模拟买卖（100股取整、佣金、滑点） |
| **engine.py** | `DailySimEngine` | T+1 流程编排：执行订单→估值→风控→信号→新订单 |
| **risk.py** | `check_stop_loss()` 等 | 止损/止盈/极端回撤 |
| **notify.py** | `push_daily_report()` | WxPusher 日报推送 |

### 9.3 T+1 待执行订单状态机

```
                    ┌──────────────┐
                    │   无待执行    │
                    └──────┬───────┘
                           │ 信号触发
                           ▼
                    ┌──────────────┐
                    │  待执行订单   │ ← 存入 state.pending_order
                    │  (买/卖/切换) │
                    └──────┬───────┘
                           │ 次日至交易日
                           ▼
              ┌─────────────────────┐
              │   执行订单（开盘价）  │
              │   检查涨/跌停       │
              └──┬──────────┬───────┘
        可成交    │          │  被封锁
                 ▼          ▼
         ┌────────────┐  ┌────────────┐
         │ 订单成交    │  │ 订单取消   │
         │ 更新持仓/资金│  │ 记录原因    │
         └────────────┘  └────────────┘
                 │              │
                 ▼              ▼
          ┌──────────────────────────┐
          │ 用 close 估值 → 算新信号  │
          │ → 产生新待执行订单        │
          └──────────────────────────┘
```

### 9.4 待执行订单格式

```json
{
  "action": "buy",          // buy | sell | switch
  "symbol": "159915",       // 买卖标的
  "buy_symbol": "588000",   // 切换时的买入端
  "sell_symbol": "510050",  // 切换时的卖出端
  "reason": "动量信号开仓",
  "created": "2026-06-22"   // 订单创建日期（T日）
}
```

### 9.5 状态文件格式

```json
{
  "version": 3,
  "last_update": "2026-06-22",
  "cash": 5000.00,
  "initial_capital": 10000.00,
  "position": {
    "symbol": "159915",
    "shares": 2300,
    "avg_cost": 4.3803,
    "total_cost": 10074.69,
    "highest_price": 4.50,
    "today_opened": false
  },
  "cumulative_pnl": 1500.00,
  "cumulative_cost": 50.00,
  "trade_log": [
    {"date": "2026-06-22", "action": "买入", "symbol": "159915",
     "shares": 2300, "price": 4.2914, "amount": 9872.54, "commission": 1.97}
  ],
  "days_since_switch": 10,
  "peak_value": 12000.00,
  "pending_order": null
}
```

### 9.6 每日流程

```python
# simulation/strategies/momentum_rotation/daily.py 的核心逻辑：

# 1. 交易日判断（数据库中有无今日数据）
if not is_trading_day(today_str):
    return  # 跳过

# 2. 数据到位检查（最新交易日 == 今日）
latest_day = get_latest_trading_day(ETF_SYMBOLS)
if latest_day != today_str:
    return  # 跳过（数据尚未同步完成）

# 3. 加载数据 + 引擎初始化
etf_data = load_latest_data(ETF_SYMBOLS, lookback_days=40)
engine = DailySimEngine(state_mgr, broker, signal_func, rank_func, ...)

# 4. 运行（执行订单 → 风控 → 信号 → 新订单）
report = engine.run_daily(etf_data, today_idx, today_str)

# 5. 推送日报
push_daily_report(STRATEGY_NAME, build_report(report))
```

---

## 10. 管线编排器 pipeline

### 10.1 cron 触发

```
0 20 * * 1-5  cd /path/to/project && python pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

### 10.2 执行流程

```
pipeline.py
  │
  ├─ Step 0: 交易日判断
  │   周末 ÷ chinese_calendar 节假日 → 跳过
  │
  ├─ Step 1: ETF 数据同步（main.py --sync-only）
  │   required = True, timeout = 3h
  │   成功 → 下一步    失败 → 管线终止
  │
  ├─ Step 2: 动量轮动模拟盘（-m simulation.strategies.momentum_rotation.daily）
  │   required = True, timeout = 10min
  │   成功 → 推送日报
  │
  ├─ (未来: Step 3, Step 4 ... 其他策略模拟盘)
  │
  └─ 推送管线汇总（WxPusher）
```

### 10.3 子进程日志

使用 `subprocess.Popen` 替代 `subprocess.run`，子进程的输出**实时流式打印**到主进程日志中，
方便排查问题。

### 10.4 自我修复

`PipelineStatus.needs_rerun()` 检测上次运行是否异常中断（状态为 `running` 或 `failed`），
是则重置重新运行。

### 10.5 pipeline_status.json 格式

```json
{
  "date": "2026-06-22",
  "pipeline_status": "completed",
  "started_at": "20:00:05",
  "finished_at": "20:02:30",
  "steps": {
    "sync": {
      "name": "ETF 数据同步",
      "status": "completed",
      "started_at": "20:00:05",
      "finished_at": "20:01:15",
      "duration": 70,
      "detail": {"returncode": 0},
      "error": null
    },
    "momentum_rotation": {
      "name": "动量轮动模拟盘",
      "status": "completed",
      "started_at": "20:01:15",
      "finished_at": "20:01:45",
      "duration": 30,
      "error": null
    }
  }
}
```

---

## 11. 策略开发指南

### 11.1 完整开发流程

```
Phase 1: 回测（strategies/下）
  ├─ 1. 复制已有策略目录（如 momentum_rotation）
  ├─ 2. 修改 config.py（参数、ETF池）
  ├─ 3. 修改 engine.py（信号逻辑）
  ├─ 4. 运行回测验证
  ├─ 5. 迭代调参
  └─ 6. 满意后进入 Phase 2

Phase 2: 模拟盘（simulation/strategies/下）
  ├─ 1. 在 simulation/strategies/下新建目录
  ├─ 2. 创建 config.py（引用回测参数 + 模拟盘特有配置）
  ├─ 3. 创建 daily.py（每日入口，调用 framework + 信号函数）
  ├─ 4. 添加到 pipeline.py 的 STEPS 列表
  └─ 5. cron 自动运行
```

### 11.2 回测引擎检查清单

```
[ ] signal_idx = max(1, idx - 1)  — 信号用前一日数据
[ ] today_data[...]["open"] — 执行用开盘价（非close）
[ ] 无 np.nan 传播到决策逻辑
[ ] 数据不足时返回默认值（非继续执行）
[ ] 费率参数可配置
[ ] run.py 中 args 支持自定义参数
```

### 11.3 模拟盘入口检查清单

```
[ ] is_trading_day() 判断
[ ] 最新交易日 == 运行日（数据到位检查）
[ ] 动量窗口数据足够（idx >= MOMENTUM_WINDOW）
[ ] 涨跌停检查（_check_limit_open）
[ ] 推送日报（可读性强）
```

---

## 12. 配置详解

### 12.1 .env 文件

```
WXPUSHER_TOKEN=AT_xxx
WXPUSHER_TOPIC_IDS=["39277"]
```

### 12.2 策略通用参数（momentum_rotation 等）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ETF_SYMBOLS` | 7只 | ETF 代码列表 |
| `ETF_POOL` | 名称映射 | 代码→中文名 |
| `INITIAL_CAPITAL` | 10000 | 初始资金 |
| `MOMENTUM_WINDOW` | 20 | 动量窗口（交易日） |
| `MIN_SWITCH_CONVICTION` | 0.03 | 切换置信度（3%） |
| `MIN_HOLD_DAYS` | 10 | 最小持仓天数 |
| `SHORT_TERM_MOMENTUM_CHECK` | True | 短期动量确认 |
| `ADJUSTMENT_DAYS` | 5 | 渐进调仓周期 |
| `COMMISSION_RATE` | 0.0002 | 佣金率 |
| `SLIPPAGE` | 0.0001 | 滑点率 |
| `RISK_MODE` | "A" | 风控模式 |
| `DB_PATH` | "data/etf_daily.db" | 数据库路径 |
| `OUTPUT_DIR` | 策略目录下的 output | 输出目录 |

### 12.3 波动率过滤特有参数（momentum_vol_filter）

| 参数 | 值 | 说明 |
|------|----|------|
| `VOL_THRESHOLD` | 0.30 | 年化波动率 > 30% 时空仓 |
| `VOL_WINDOW` | 20 | 波动率计算滚动窗口（交易日） |

### 12.4 配对交易特有参数（pair_trading）

| 参数 | 值 | 说明 |
|------|----|------|
| `PAIRS` | 3对 | 上证50↔创业板 + 沪深300↔创业板 + 上证50↔科创50 |
| `CAPITAL_PER_PAIR` | 3333 | 每对资金 |
| `ZSCORE_PERIOD` | 60 | z-score 统计窗口 |
| `ZSCORE_OPEN` | 2.0 | 开仓阈值 |
| `ZSCORE_CLOSE` | 0.3 | 平仓阈值 |
| `ZSCORE_STOP` | 3.0 | 止损阈值 |

### 12.5 组合策略参数（combined）

| 参数 | 值 | 说明 |
|------|----|------|
| `TOTAL_CAPITAL` | 10000 | 总资金 |
| `MOMENTUM_PCT` | 0.8 | 动量占比 80% |
| `PAIR_PCT` | 0.2 | 配对占比 20% |

### 12.6 模拟盘配置（simulation/strategies/momentum_rotation/config.py）

| 参数 | 值 | 说明 |
|------|----|------|
| `INITIAL_CAPITAL` | 10000 | 初始资金 |
| `RISK_MODE` | "A" | 风控模式（纯信号） |
| `STOP_LOSS_PCT` | 0.05 | 止损比例（RISK_MODE=B 时生效） |
| `DRAWBACK_PCT` | 0.05 | 移动止盈回撤比例 |

---

## 13. 运行指南

### 13.1 环境

```bash
pip install pandas numpy matplotlib pydantic-settings \
            python-dotenv rich chinese_calendar \
            requests akshare wxpusher
```

### 13.2 数据回填（首次使用）

```bash
cd /public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest
python main.py --backfill    # 拉取全部历史数据（约10-20分钟）
```

### 13.3 运行回测

```bash
# 最优策略（波动率过滤）
python -m strategies.momentum_vol_filter.run

# 基准策略（纯动量）
python -m strategies.momentum_rotation.run --tag mytest

# 配对交易
python -m strategies.pair_trading.run

# 组合策略
python -m strategies.combined.run

# 覆盖默认参数
python -m strategies.momentum_rotation.run \
  --start 2024-01-01 --end 2026-06-22 \
  --money 10000 --momentum 20 --tag test1
```

### 13.4 测试模拟盘

```bash
python -m simulation.strategies.momentum_rotation.daily
# 非交易日自动跳过，交易日自动执行
```

### 13.5 手动触发管线

```bash
python pipeline.py    # 同 cron 执行逻辑
```

### 13.6 查看日志

```bash
tail -f logs/pipeline_$(date +%Y%m%d).log
```

---

## 14. 已知 Bug 与修复记录

### 14.1 SQL ORDER BY 位置错误（已修复）

**发现时间：** 2026-06-23  
**影响范围：** 全部回测策略的 `--end` 参数（独立年份回测）  
**症状：** 独立跑某一年份时（`--end 2024-12-31`），数据顺序颠倒，
导致回测结果完全错误  
**原因：** `data.py` 中 `_load_single_etf` 的 SQL 查询拼接：
```python
# ❌ 错误
query = "SELECT ... WHERE symbol=? AND date>=? ORDER BY date"
if end_date:
    query += " AND date<=?"  # ← 拼在 ORDER BY 之后！
# 效果变成了: ORDER BY (date AND date <= ?) → 布尔表达式 → 倒序

# ✅ 正确
query = "SELECT ... WHERE symbol=? AND date>=?"
if end_date:
    query += " AND date<=?"
query += " ORDER BY date"  # ORDER BY 必须在最后
```

### 14.2 信号与交易同日执行（已修复）

**发现时间：** 2026-06-24  
**影响范围：** 全部回测引擎  
**症状：** 用 `close[T]` 计算信号、用 `close[T]` 执行——look-ahead bias  
**修复：** 全部引擎修正为 `close[T-1]` 信号 → `open[T]` 执行  
**影响：** 收益率普遍下降约 10-20%（修正了之前的虚高）  

### 14.3 模拟盘风控卖出状态丢失（已修复）

**影响范围：** `simulation/framework/engine.py`  
**症状：** 风控触发卖出后，`state = self.state_mgr.load()` 重载了旧状态，
覆盖了卖出操作  
**修复：** 移除风控卖出后的 load()，直接 save()  

---

## 15. 常见问题

### Q: 回测和模拟盘的结果为什么不同？

因为交易时序和价格不同：
- 回测：`close[T-1]` 信号 → `open[T]` 执行（使用同一日期的开/收盘价）
- 模拟盘：`close[T]` 信号 → `open[T+1]` 执行（使用不同日期的价格）
时序上等价，但具体执行价格不同（`open[T]` ≠ `open[T+1]`）。

### Q: 为什么 7 只 ETF 的池子不是因为 look-ahead bias 加上 创业板/科创板的？

创业板和科创板是 A 股市场的**独立市场层次**（类似于主板、中小板），
不是"2025年涨得好所以加上的行业板块"。7 只 ETF（5只宽基+2只成长板）
共同构成 A 股的完整市场覆盖。真正有 look-ahead bias 问题的是手动添加行业 ETF
（如半导体ETF、酒ETF），这类做法已被排除。

### Q: 策略开发完回测后，怎么上线模拟盘？

1. 在 `simulation/strategies/` 下建对应目录（config.py + daily.py）
2. daily.py 调用策略信号函数 + framework 引擎执行
3. 在 `pipeline.py` 的 `STEPS` 列表中添加一项
4. 第二天 cron 自动运行

### Q: 为什么不在回调时止损？

动量策略的盈利核心是"持有趋势最强的标的"。历史测试证明（A/B/C 三模式对比），
止损会打断趋势，导致收益大幅下降。**波动率过滤**（高波动时离场）是比价格止损
更有效的风险控制手段——它衡量的是"趋势是否可靠"，而非"价格跌了多少"。

### Q: 数据同步失败了怎么办？

```bash
# 手动强制同步
python main.py --force

# 如果数据损坏，重新回填
python main.py --backfill
```

### Q: pipeline 异常中断了怎么恢复？

下次 cron 触发时，`PipelineStatus.needs_rerun()` 检测到上次状态为 `running`
或 `failed`，会自动重置并从头运行。

### Q: 配对交易在熊市中真的能赚钱吗？

2023 年验证：HS300 跌了 16.7%，配对交易策略 **仅亏 0.16%**（基本持平）。
市场中性策略不依赖大盘方向，收益来源于价差回归。
在牛市中收益低于动量策略（+25% vs +133%），但回撤仅 6%。

### Q: 如何选择最优策略？

看你的目标：
- **追求绝对收益：** `momentum_ma_etf`（+191%），但回撤大
- **追求风险调整收益（推荐）：** `momentum_vol_filter`（夏普 1.31，回撤 23%）
- **追求稳定性：** `pair_trading`（回撤 6%，市场中性）
- **攻守兼备：** `combined`（动量 80% + 配对 20%，+108%）

---

> **最后更新：** 2026-06-24  
> **GitHub：** [github.com/zhuleimed/etf-daily-sync-and-backtest](https://github.com/zhuleimed/etf-daily-sync-and-backtest)  
> **策略对比：** `strategies/STRATEGY_COMPARISON.md`
