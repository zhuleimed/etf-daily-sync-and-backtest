# 019 ETF 量化策略 — 数据同步 · 回测框架 · 模拟运行

> A 股 ETF 量化交易研究项目。覆盖数据获取、策略回测、模拟盘运行全流程。
>
> **项目入口：** `pipeline.py`（cron 20:00 触发）→ 数据同步 → 模拟盘运行
> **策略开发：** `strategies/` 下建子目录 → 回测验证 → `simulation/` 建对应模拟盘

---

## 目录

1. [项目结构](#1-项目结构)
2. [核心交易逻辑（必须遵守！）](#2-核心交易逻辑必须遵守)
3. [数据同步模块](#3-数据同步模块)
4. [回测框架](#4-回测框架)
5. [模拟盘框架](#5-模拟盘框架)
6. [管线编排器 pipeline](#6-管线编排器-pipeline)
7. [策略一览](#7-策略一览)
8. [策略开发指南](#8-策略开发指南)
9. [配置详解](#9-配置详解)
10. [运行指南](#10-运行指南)
11. [常见问题](#11-常见问题)

---

## 1. 项目结构

```
019_etf_daily_sync_and_backtest/
│
├── pipeline.py                  # 管线编排器（cron 入口）
├── pipeline_status.py           # 管线状态追踪与推送
├── main.py                      # 数据同步入口（6种模式）
│
├── .env                         # 环境变量（WxPusher Token 等）
├── .gitignore
├── README.md                    # ← 本文档
│
├── etf_sync/                    # ═══ 数据同步模块 ═══
│   ├── config.py                #   配置管理（pydantic-settings）
│   ├── data_source.py           #   双源数据获取（腾讯→新浪）
│   ├── engine.py                #   SQLite 数据库引擎
│   ├── sync.py                  #   同步管理器
│   ├── logger.py                #   日志
│   └── notify.py                #   WxPusher 推送
│
├── strategies/                  # ═══ 回测策略 ═══
│   ├── STRATEGY_COMPARISON.md   #   策略对比分析文档
│   ├── momentum_rotation/       #   ① 纯动量轮动（基准策略）
│   ├── momentum_vol_filter/     #   ② 波动率过滤轮动 ✅ 最优
│   ├── momentum_ma_filter/      #   ③ 大盘均线过滤
│   ├── momentum_ma_etf/         #   ④ 逐ETF均线过滤
│   ├── momentum_dual/           #   ⑤ 双动量（绝对+相对）
│   ├── dual_ma_crossover/       #   ⑥ 双均线交叉
│   ├── low_vol_rotation/        #   ⑦ 低波动率轮动
│   ├── mean_reversion/          #   ⑧ 均值回归轮动
│   ├── vol_price_momentum/      #   ⑨ 量价配合轮动
│   ├── donchian_breakout/       #   ⑩ 唐奇安通道突破
│   ├── bollinger_rotation/      #   ⑪ 布林带轮动
│   ├── pair_trading/            #   ⑫ 配对交易（市场中性）
│   ├── combined/                #   ⑬ 组合策略（动量+配对）
│   └── 每个策略目录包含：
│       ├── config.py            #   策略参数
│       ├── engine.py            #   回测引擎
│       ├── data.py              #   数据加载
│       ├── run.py               #   运行入口
│       ├── metrics.py           #   绩效指标
│       ├── reporter.py          #   报告+图表
│       ├── cost.py              #   交易成本
│       └── risk.py              #   风控模块
│
├── simulation/                  # ═══ 模拟盘框架 ═══
│   ├── framework/               #   通用模板块
│   │   ├── state.py             #   JSON 原子持久化
│   │   ├── data.py              #   数据加载（从 etf_daily.db）
│   │   ├── broker.py            #   模拟交易执行
│   │   ├── engine.py            #   每日流程编排
│   │   ├── risk.py              #   风控检查
│   │   └── notify.py            #   WxPusher 推送
│   └── strategies/              #   各策略模拟盘入口
│       └── momentum_rotation/
│           ├── config.py        #   模拟盘配置
│           └── daily.py         #   每日运行入口
│
├── data/                        # SQLite 数据库（gitignore）
│   └── etf_daily.db
│
└── logs/                        # 日志（gitignore）
```

---

## 2. 核心交易逻辑（必须遵守！）

### 2.1 回测引擎的交易时序

所有回测策略遵循同一时序规则，**这是整个项目的纪律红线**：

```
T-1 日收盘后:
  信号数据截止于 close[T-1]

T 日开盘:
  用 open[T] 执行买卖（不是 close[T]）
  检查是否涨停/跌停：涨停不能买入，跌停不能卖出

总结: close[T-1] → 信号 | open[T] → 执行
```

**实现方式：**
- `engine.py` 中 `_buy()` 方法：`price = today_data[symbol]["open"] * (1 + SLIPPAGE)`
- `engine.py` 中 `_sell()` 方法：`sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)`
- 信号计算使用 `signal_idx = max(1, idx - 1)`（前一日数据）

**绝对禁止：**
- ❌ 使用 `close[T]` 同时计算信号和执行（look-ahead bias）
- ❌ 使用未来数据计算任何指标
- ❌ 当日买入当日卖出（A 股 T+1 规则）

### 2.2 模拟盘引擎的交易时序

```
T 日 20:00 数据同步完成:
  用 T 日 close 计算信号
  产生 "待执行订单" → 存入状态文件

T+1 日 20:00:
  读取昨日待执行订单
  用 T+1 日 open[T+1] 执行（检查涨跌停）
  用 T+1 日 close 计算新信号 → 产生新待执行订单
```

**关键区别：**
| 引擎 | 信号时间 | 执行时间 | 执行价格 |
|------|---------|---------|---------|
| 回测 `strategies/` | `close[T-1]` | `open[T]` | 开盘价±滑点 |
| 模拟 `simulation/` | `close[T]` | `open[T+1]` | 开盘价±滑点 |

两种引擎的时序在数学上等价——信号都比执行提前约 1 个 bar。

---

## 3. 数据同步模块

### 3.1 数据源

| 类型 | 主源 | 备选 | 说明 |
|------|:----:|:----:|------|
| ETF 日线 | **腾讯** web.ifzq.gtimg.cn | **新浪** quotes.sina.cn | 双轨制自动切换 |
| 指数日线 | **新浪** quotes.sina.cn | — | 腾讯不支持指数K线 |
| ETF 列表 | **akshare** fund_etf_spot_em() | — | 约1500+只 |

### 3.2 同步标的

**ETF：** 全量场内 ETF（沪市 51xxx/56xxx/58xxx/588xxx，深市 15xxx/16xxx/159xxx）

**指数：** 上证50(000016)、沪深300(000300)、中证500(000905)、中证1000(000852)

### 3.3 运行模式

```bash
python main.py                     # 标准模式（20:00 后执行）
python main.py --sync-only         # 仅同步数据
python main.py --force             # 跳过交易日/时间检查
python main.py --backfill          # 全量回填
python main.py --list-only         # 仅更新 ETF 列表
```

### 3.4 双轨制（Tencent → Sina）

ETF 数据获取采用双轨制自动切换：
1. 默认使用 **Tencent** 接口
2. Tencent 连续失败 → 自动切换到 **Sina**
3. 每 50 次 Sina 请求尝试恢复 Tencent
4. 日志中标记当前活跃数据源

---

## 4. 回测框架

### 4.1 每个策略的目录结构

```
strategies/策略名称/
  ├── config.py      # 策略参数
  ├── engine.py      # 回测引擎（BacktestEngine 类）
  ├── data.py        # 数据加载（SQLite → DataFrame）
  ├── momentum_signals.py  # 信号计算（仅动量策略需要）
  ├── cost.py        # 交易成本计算
  ├── risk.py        # 风控模块（A/B/C 三种模式）
  ├── metrics.py     # 绩效指标计算
  ├── reporter.py    # 报告生成 + matplotlib 图表
  └── run.py         # 运行入口（argparse）
```

### 4.2 引擎核心流程

每个回测引擎的 `run()` 方法逐日循环：

```
for idx in range(n):
    1. 加载今日 OHLCV 数据
    2. 计算信号（用 close[T-1]，即 signal_idx = max(0, idx-1)）
    3. 决策：开仓 / 切换 / 持有
    4. 执行（用 open[T]）
    5. 记录当日账户状态
```

### 4.3 风控模式（RISK_MODE）

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| **A** | 纯信号模式，无风控 | 回测基准、动量策略 |
| **B** | 全开（止损+止盈+极端回撤） | 实盘保守策略 |
| **C** | 仅极端回撤（15%） | 适度风控 |

**历史测试结论：** 对动量策略，风控反而降低收益（打断了趋势）。默认用 A。

### 4.4 交易成本模型

| 项目 | 费率 |
|------|------|
| 佣金 | 0.02%（万分之二） |
| 滑点 | 0.01%（万分之一） |
| 印花税 | 0%（ETF 免收） |
| 最小佣金 | 无（ETF 免最低5元限制） |
| 冲击成本 | 调仓金额 × 冲击系数(0.1) ÷ 日均成交额 |

### 4.5 绩效指标

- 累计收益率、年化收益率
- 最大回撤及持续天数
- 夏普比率（无风险利率 3%）
- Sortino 比率、Calmar 比率
- 日均/下行波动率
- 交易胜率、盈亏比
- 持仓分布热力图
- 月度收益热力图
- 超额收益（vs 沪深300 / 等权组合）

---

## 5. 模拟盘框架

### 5.1 通用框架模块

`simulation/framework/` 提供与策略无关的通用组件：

| 模块 | 职责 | 关键文件 |
|------|------|---------|
| **state.py** | JSON 状态持久化（原子写入），记录持仓/资金/交易日志 | `StateManager`, `SimState` |
| **data.py** | 从 `etf_daily.db` 加载最近 N 个交易日数据 | `load_latest_data()` |
| **broker.py** | 模拟 A 股 ETF 买卖（100股取整、佣金滑点） | `SimBroker` |
| **engine.py** | T+1 待执行订单流程编排 | `DailySimEngine` |
| **risk.py** | 止损/止盈/极端回撤检查 | `run_all_risk_checks()` |
| **notify.py** | WxPusher 微信推送 | `push_daily_report()` |

### 5.2 T+1 待执行订单流程

```
每日调用 engine.run_daily():
  1. 加载昨日持仓状态（JSON）
  2. 重置 today_opened = False（昨天买的今天可以卖了）
  3. 执行昨日的 pending_order（如有）：
     a. 用今日开盘价成交
     b. 检查涨/跌停 → 封锁则取消订单
     c. 切换订单需双边都通过检查
  4. 用今日收盘价估值（更新最高价、总资产）
  5. 风控检查 → 触发则生成待执行卖出订单
  6. 计算动量信号 → 生成新的待执行订单
  7. 持久化状态（含 pending_order）
```

### 5.3 涨跌停检查

```python
def _check_limit_open(symbol, open_price, prev_close):
    limit_pct = 0.20 if symbol in 创业板/科创板 else 0.10
    if open_price >= prev_close * (1 + limit_pct):
        return True, "涨停"
    if open_price <= prev_close * (1 - limit_pct):
        return True, "跌停"
    return False, ""
```

### 5.4 状态文件格式

存储在 `simulation/output/state_策略名.json`：

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
  "trade_log": [...],
  "days_since_switch": 10,
  "peak_value": 12000.00,
  "pending_order": null
}
```

---

## 6. 管线编排器 pipeline

### 6.1 触发方式

**cron：** `0 20 * * 1-5`（交易日 20:00）

### 6.2 执行流程

```
pipeline.py
  │
  ├─ 交易日判断（chinese_calendar）
  │   非交易日 → 记录 "skipped" → 结束
  │
  ├─ Step 1: 数据同步（main.py --sync-only）
  │   必需步骤，超时 3h
  │   失败 → 管线终止，模拟盘不运行
  │
  ├─ Step 2: 各策略模拟盘（simulation/strategies/*/daily.py）
  │   逐策略执行，超时 10min
  │   成功后 → 推送微信日报
  │
  └─ 推送管线汇总（WxPusher）
```

### 6.3 状态追踪

`pipeline_status.json` 记录每次运行状态：

```json
{
  "date": "2026-06-22",
  "pipeline_status": "completed",
  "steps": {
    "sync": {
      "name": "ETF 数据同步",
      "status": "completed",
      "duration": 120
    },
    "momentum_rotation": {
      "name": "动量轮动模拟盘",
      "status": "completed",
      "duration": 30
    }
  }
}
```

---

## 7. 策略一览

### 7.1 完整排行

| 排名 | 策略 | 总收益 | 夏普 | 最大回撤 | 切换 | 核心逻辑 |
|------|------|--------|------|---------|------|---------|
| 1 | **ma_etf 逐ETF均线过滤** | **+191%** | **1.50** | -26% | — | 每个ETF价格>自身均线才买入 |
| 2 | **ma_filter MA=250** | **+167%** | **1.37** | -26% | — | 沪深300>250日均线才轮动 |
| 3 | **vol_filter 波动率过滤** | **+117%** | **1.31** | -23% | 19 | 年化波动率>30%时空仓 ✅ 夏普最优 |
| 4 | **rotation 纯动量** | **+133%** | **1.19** | -26% | 19 | 20日动量排名选最强（基准） |
| 5 | combined 动量+配对 | +108% | 1.11 | -24% | — | 80%动量+20%配对 |
| 6 | bollinger 布林带 | +108% | 1.10 | -20% | 217 | 布林带位置打分 |
| 7 | dual 双动量 | +86% | 0.94 | -28% | 33 | 绝对动量+相对动量 |
| 8 | mean_reversion 均值回归 | +80% | 0.94 | -17% | 136 | 超跌反弹 |
| 9 | donchian 通道突破 | +67% | 0.85 | -21% | 108 | 60日通道突破 |
| 10 | vol_price 量价 | +55% | 0.71 | -21% | 181 | 动量×成交量确认 |
| 11 | crossover 双均线 | +45% | 0.81 | -14% | 94 | MA(60,120)交叉 |
| 12 | low_vol 低波动率 | +32% | 0.58 | -18% | 50 | 持有波动最低ETF |
| 13 | **pair_trading 配对交易** | **+25%** | **—** | **-6%** | 42 | z-score价差回归（市场中性） |

### 7.2 各策略简要说明

| 策略 | 原理 | 适用市场 |
|------|------|---------|
| **动量轮动** | 过去N日涨幅最大的ETF → 持有 | 强趋势市 |
| **波动率过滤** | 年化波动率>30%时空仓，否则动量轮动 | 所有市场 ✅ |
| **均线过滤(大盘)** | 沪深300指数>年线才轮动 | 牛市 |
| **均线过滤(逐ETF)** | 每个ETF>自身60日均线才可买入 | 牛市 |
| **双动量** | 动量>0且排名第1才持有 | 弱趋势市 |
| **双均线交叉** | 快线上穿慢线→买入，下穿→卖出 | 大趋势市 |
| **低波动率** | 持有波动率最低的ETF | 低波动环境 |
| **均值回归** | 持有距均线最远的ETF（超跌反弹） | 震荡市 |
| **量价配合** | 动量×成交量确认（放量才追） | 趋势确认 |
| **通道突破** | 突破60日最高价买入，跌破最低价卖出 | 强趋势市 |
| **布林带** | 靠近下轨买，靠近上轨卖 | 震荡市 |
| **配对交易** | 大盘vs成长价差回归，多空对冲 | 所有市场（市场中性） |
| **组合策略** | 80%动量+20%配对 | 攻守兼备 |

> 完整对比分析见 `strategies/STRATEGY_COMPARISON.md`

---

## 8. 策略开发指南

### 8.1 开发流程

```
1. 在 strategies/ 下新建子目录
2. 创建 config.py（策略参数）
3. 创建 engine.py（回测引擎，可复制 momentum_rotation 修改）
4. 创建 run.py（入口脚本）
5. 跑回测验证效果
6. 满意后在 simulation/strategies/ 下建对应模拟盘入口
7. 更新 STRATEGY_COMPARISON.md
```

### 8.2 回测引擎必须遵守的规则

```python
# ✅ 信号用前一日数据
signal_idx = max(1, idx - 1)

# ✅ 执行用今日开盘价
price = today_data[symbol]["open"] * (1 + SLIPPAGE)

# ✅ 卖出也用开盘价
sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)

# ❌ 禁止：用收盘价同时算信号和执行
price = today_data[symbol]["close"] * (1 + SLIPPAGE)  # 错误！
```

### 8.3 数据加载

```python
from .data import load_all_etf_data

# 自动加载 ETF 数据 + 对齐交易日 + 计算辅助列（动量/ATR等）
etf_data, dates = load_all_etf_data(
    symbols=ETF_SYMBOLS,
    start_date="2024-01-01",
    momentum_window=20,  # 控制动量列计算窗口
)
```

---

## 9. 配置详解

### 9.1 全局配置（.env）

```
WXPUSHER_TOKEN=AT_xxx           # WxPusher 应用 Token
WXPUSHER_TOPIC_IDS=["39277"]    # 推送 Topic ID
```

### 9.2 策略通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MOMENTUM_WINDOW | 20 | 动量计算窗口 |
| MIN_SWITCH_CONVICTION | 0.03 | 切换置信度（3%） |
| MIN_HOLD_DAYS | 10 | 最小持仓天数 |
| SHORT_TERM_MOMENTUM_CHECK | True | 短期动量确认 |
| ADJUSTMENT_DAYS | 5 | 渐进调仓周期 |
| COMMISSION_RATE | 0.0002 | 佣金（万分之二） |
| SLIPPAGE | 0.0001 | 滑点（万分之一） |
| RISK_MODE | "A" | 风控模式 |

### 9.3 波动率过滤特有参数

| 参数 | 值 | 说明 |
|------|----|------|
| VOL_THRESHOLD | 0.30 | 年化波动率阈值 |
| VOL_WINDOW | 20 | 波动率计算窗口 |

### 9.4 配对交易特有参数

| 参数 | 值 | 说明 |
|------|----|------|
| ZSCORE_PERIOD | 60 | z-score 统计窗口 |
| ZSCORE_OPEN | 2.0 | 开仓阈值 |
| ZSCORE_CLOSE | 0.3 | 平仓阈值 |
| ZSCORE_STOP | 3.0 | 止损阈值 |

---

## 10. 运行指南

### 10.1 环境要求

```bash
# Python 3.12+
pip install pandas numpy matplotlib pydantic-settings \
            python-dotenv rich chinese_calendar \
            requests akshare wxpusher
```

### 10.2 运行回测

```bash
# 纯动量（基准策略）
python -m strategies.momentum_rotation.run

# 波动率过滤（最优策略）
python -m strategies.momentum_vol_filter.run

# 配对交易
python -m strategies.pair_trading.run

# 组合策略
python -m strategies.combined.run

# 自定义参数
python -m strategies.momentum_rotation.run \
  --start 2024-01-01 --end 2026-06-22 \
  --money 10000 --tag mytest
```

### 10.3 运行模拟盘

```bash
# 测试运行（如当日有数据）
python -m simulation.strategies.momentum_rotation.daily

# 正式运行由 pipeline.py 在 20:00 自动触发
```

### 10.4 cron 配置

```bash
# ETF 数据同步 + 模拟盘管线
0 20 * * 1-5 cd /path/to/project && python pipeline.py \
  >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

---

## 11. 常见问题

### Q: 回测和模拟盘的结果为什么不同？

因为交易时序不同：
- 回测用 `close[T-1]` 信号 → `open[T]` 执行
- 模拟盘用 `close[T]` 信号 → `open[T+1]` 执行
在数学上等价，但具体价格不同（open[T] ≠ open[T+1]）。

### Q: 为什么波动率过滤看起来收益不是最高，但被称为最优？

**风险调整收益（夏普比率）最高。** ma_etf 虽然收益 191%，但依赖于特定参数和牛市环境。波动率过滤在保持 117% 收益的同时，夏普 1.31 是最高的，说明每承担一单位风险获得的回报最大。

### Q: 为什么加了创业板和科创板ETF不是 look-ahead bias？

创业板和科创板是 A 股市场的**独立市场层次**，不是"2025年涨得好所以加上的行业板块"。7 只 ETF（5只宽基+2只成长板）共同构成 A 股完整的市场覆盖。

### Q: 为什么不在回调时止损？

动量策略的盈利核心是"持有趋势最强的标的"。历史测试证明，止损会打断趋势，导致收益大幅下降。波动率过滤（高波动时离场）是比价格止损更有效的风险控制手段。

### Q: 配对交易为什么收益低？

配对交易是市场中性策略（多空对冲），收益来源于价差回归而非市场涨跌。牛市里赚得少，但熊市里也能赚钱。2023 年 HS300 跌了 16.7%，配对交易只亏了 0.16%。

### Q: 如何添加新策略？

1. 复制 `strategies/momentum_rotation/` 到新目录
2. 修改 `config.py`（参数）和 `engine.py`（信号逻辑）
3. 运行回测验证
4. 满意后建 `simulation/strategies/新策略/` 的模拟盘入口
5. 更新 `STRATEGY_COMPARISON.md`

---

> **最后更新：** 2026-06-24
> **GitHub：** [github.com/zhuleimed/etf-daily-sync-and-backtest](https://github.com/zhuleimed/etf-daily-sync-and-backtest)
