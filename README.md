# 019 ETF 量化策略 — 数据同步 · 回测框架 · 模拟运行

> A 股 ETF 量化交易研究项目。覆盖数据获取、策略回测、模拟盘运行全流程。
> Python 3.12+，SQLite 存储，WxPusher 推送，chinese_calendar 交易日判断。
>
> **项目入口：** `pipeline.py`（cron 交易日 20:00 触发）→ 数据同步 → 模拟盘运行
> **策略开发流程：** `strategies/` 下建子目录 → 回测验证 → `simulation/` 下建对应模拟盘入口
> **纪律红线（详见 [2. 核心交易逻辑](#2-核心交易逻辑纪律红线)）：**
>   - 回测：信号用 `close[T-1]`，执行用 `open[T]`
>   - 模拟盘：信号用 `close[T]`，执行用 `open[T+1]`（检查涨跌停）

---

## 目录

1. [快速开始](#1-快速开始)
2. [项目结构](#2-项目结构)
3. [核心交易逻辑（纪律红线）](#3-核心交易逻辑纪律红线)
4. [数据同步模块](#4-数据同步模块)
5. [数据库](#5-数据库)
6. [数据加载 API](#6-数据加载-api)
7. [回测框架](#7-回测框架)
8. [回测引擎核心机制详解](#8-回测引擎核心机制详解)
9. [交易成本与滑点模型](#9-交易成本与滑点模型)
10. [绩效指标计算方法](#10-绩效指标计算方法)
11. [13 种策略一览](#11-13-种策略一览)
12. [策略对比排行](#12-策略对比排行)
13. [模拟盘框架（T+1 待执行订单）](#13-模拟盘框架t1-待执行订单)
14. [管线编排器 pipeline](#14-管线编排器-pipeline)
14.3 [汇总日报推送](#143-汇总日报推送格式)
15. [策略开发指南](#15-策略开发指南)
16. [配置详解](#16-配置详解)
17. [运行指南](#17-运行指南)
18. [日志与监控](#18-日志与监控)
19. [依赖环境与版本兼容](#19-依赖环境与版本兼容)
20. [已知 Bug 与修复记录](#20-已知-bug-与修复记录)
21. [故障排除](#21-故障排除)
22. [术语表](#22-术语表)
23. [常见问题](#23-常见问题)

---

## 1. 快速开始

### 1.1 首次使用（数据回填）

```bash
# 1. 安装依赖
pip install pandas numpy matplotlib pydantic-settings \
            python-dotenv rich chinese_calendar \
            requests akshare wxpusher

# 2. 配置 .env（WxPusher Token）
echo 'WXPUSHER_TOKEN=AT_你的Token' > .env
echo 'WXPUSHER_TOPIC_IDS=["39277"]' >> .env

# 3. 回填历史数据（约 10-20 分钟）
python main.py --backfill

# 4. 运行基准策略回测
python -m strategies.momentum_rotation.run
```

### 1.2 常用命令速查

```bash
python main.py --backfill                          # 回填历史数据
python -m strategies.momentum_vol_filter.run        # 跑最优策略回测
python -m strategies.pair_trading.run                # 跑配对交易回测
python -m strategies.combined.run                    # 跑组合策略回测
python -m strategies.composite_momentum.run          # 跑复合动量回测
python -m strategies.adx_trend_rotation.run           # 跑ADX趋势强度回测
python -m strategies.macd_trend_rotation.run           # 跑MACD趋势轮动回测
python -m strategies.rsi_trend_rotation.run            # 跑RSI趋势确认回测
python -m strategies.gold_safe_haven.run              # 跑黄金避险轮动回测 🆕
python -m strategies.cross_border.run               # 跑跨境轮动回测 🏆
python -m strategies.composite_momentum.run          # 跑复合动量回测
python pipeline.py                                   # 手动触发每日管线
python -m simulation.strategies.momentum_rotation.daily  # 手动跑模拟盘
python -m simulation.strategies.composite_momentum.daily  # 手动跑复合动量
python -m simulation.strategies.adx_trend_rotation.daily      # 手动跑ADX趋势
python -m simulation.strategies.macd_trend_rotation.daily       # 手动跑MACD趋势
python -m simulation.strategies.rsi_trend_rotation.daily        # 手动跑RSI趋势
python -m simulation.strategies.gold_safe_haven.daily    # 手动跑黄金避险 🆕
python -m simulation.strategies.cross_border.daily      # 手动跑跨境轮动 🏆
python -m simulation.strategies.composite_momentum.daily  # 手动跑复合动量模拟盘
```

### 1.3 文件系统布局速览

```
项目根目录 (~1.2MB 代码)
├── 核心脚本              pipeline.py, pipeline_status.py, main.py
├── 数据同步模块(6文件)    etf_sync/
├── 回测策略(14目录)       strategies/*/
├── 模拟盘框架(6+2文件)    simulation/framework/ + simulation/strategies/
├── SQLite 数据库          data/etf_daily.db (~100MB)
├── 日志                  logs/
└── 文档                  README.md, STRATEGY_COMPARISON.md, OPTIMIZATION_HISTORY.md
```

---

## 2. 项目结构

### 2.1 完整目录树

```
019_etf_daily_sync_and_backtest/
│
├── pipeline.py                     # 管线编排器（cron 入口）
│   ├── 交易日判断 → 数据同步 → 模拟盘运行
│   └── 状态追踪 + WxPusher 推送汇总
│
├── pipeline_status.py              # 管线状态追踪类
│   ├── PipelineStatus — JSON 原子持久化
│   ├── needs_rerun() — 自我修复检测
│   └── push_pipeline_summary() — 推送汇总
│
├── main.py                         # 数据同步入口
│   ├── 标准模式（ETF列表+日线+指数）
│   ├── --sync-only（仅同步）
│   ├── --force（跳过检查）
│   ├── --backfill（全量回填）
│   └── --list-only（仅更新列表）
│
├── .env                            # WxPusher Token（不提交）
├── .gitignore
├── README.md                       # ← 本文档
│
├── etf_sync/                       # ══ 数据同步模块 ══
│   ├── __init__.py
│   ├── config.py                   # pydantic-settings 配置
│   │   ├── Settings 类
│   │   ├── db_path|start_date|sync_after_hour
│   │   └── wxpusher_token|topic_ids
│   ├── data_source.py              # 双轨制数据源
│   │   ├── TencentSource — 腾讯接口（主）
│   │   ├── IndexDataSource — 新浪接口（指数）
│   │   └── to_tencent_code() / to_sina_code()
│   ├── engine.py                   # SQLite 数据库引擎
│   │   ├── DataEngine — 查询封装
│   │   └── get_etf_data() / get_index_data()
│   ├── sync.py                     # 同步管理器
│   │   ├── ETFSync — 同步管线控制
│   │   ├── sync_etf_list()
│   │   ├── sync_etf_daily()
│   │   ├── sync_index_daily()
│   │   └── is_trade_day() — 交易日判断
│   ├── logger.py                   # rich 日志
│   └── notify.py                   # WxPusher 推送
│       ├── push_sync_summary()
│       └── push_error_alert()
│
├── strategies/                     # ══ 回测策略（14个）══
│   ├── STRATEGY_COMPARISON.md      # 策略对比分析文档
│   ├── OPTIMIZATION_HISTORY.md     # 优化历程记录（1800行）
│   │
│   ├── momentum_rotation/          # ① 纯动量轮动（基准）
│   ├── momentum_vol_filter/        # ② 波动率过滤轮动 ✅ 夏普最优
│   ├── momentum_ma_filter/         # ③ 大盘MA250均线过滤
│   ├── momentum_ma_etf/            # ④ 逐ETF均线过滤
│   ├── momentum_dual/              # ⑤ 双动量（绝对+相对）
│   ├── dual_ma_crossover/          # ⑥ 双均线交叉 MA(60,120)
│   ├── low_vol_rotation/           # ⑦ 低波动率轮动
│   ├── mean_reversion/             # ⑧ 均值回归轮动
│   ├── vol_price_momentum/         # ⑨ 量价配合轮动
│   ├── donchian_breakout/          # ⑩ 唐奇安通道突破
│   ├── bollinger_rotation/         # ⑪ 布林带轮动
│   ├── pair_trading/               # ⑫ 配对交易（市场中性）
│   ├── combined/                   # ⑬ 组合策略（动量80%+配对20%）
│   ├── composite_momentum/          # ⑭ 复合动量轮动（多因子打分）
│   ├── adx_trend_rotation/           # ⑮ ADX趋势强度轮动
│   ├── macd_trend_rotation/          # ⑯ MACD趋势轮动
│   ├── rsi_trend_rotation/           # ⑰ RSI趋势确认轮动
│   ├── adaptive_rotation/            # ⑱ 自适应轮动（动态切换）
│   └── 每个策略目录结构：
│       ├── __init__.py
│       ├── config.py               # 策略参数
│       ├── engine.py               # 回测引擎（BacktestEngine 类）
│       ├── data.py                 # 数据加载
│       ├── momentum_signals.py     # 信号计算函数
│       ├── cost.py                 # 交易成本
│       ├── risk.py                 # 风控模块
│       ├── metrics.py              # 绩效指标计算
│       ├── reporter.py             # 报告 + matplotlib 图表
│       └── run.py                  # CLI 入口
│
├── simulation/                     # ══ 模拟盘框架 ══
│   ├── __init__.py
│   └── framework/                  # 通用模板块（与策略无关）
│       ├── __init__.py
│       ├── state.py                # JSON 原子持久化
│       │   ├── StateManager — 读写
│       │   └── SimState — 数据类
│       ├── data.py                 # 从 etf_daily.db 加载数据
│       │   └── load_latest_data()
│       ├── broker.py               # 模拟交易执行
│       │   └── SimBroker — 买卖/100股取整/佣金
│       ├── engine.py               # T+1 每日流程编排
│       │   └── DailySimEngine — 执行订单→估值→风控→信号→新订单
│       ├── risk.py                 # 止损/止盈/极端回撤
│       └── notify.py               # WxPusher 推送
│   └── strategies/                 # 各策略模拟盘入口
│       ├── momentum_rotation/
│       │   ├── __init__.py
│       │   ├── config.py           # 模拟盘特有配置
│       │   └── daily.py            # 每日运行入口
│       └── composite_momentum/
│           ├── __init__.py
│           ├── config.py           # 模拟盘特有配置
│           └── daily.py            # 每日运行入口
│
├── market_regime_rotation/         # [已放弃] 市场状态识别实验
│
├── data/                           # SQLite 数据库（gitignore）
│   └── etf_daily.db                # ~100MB，包含所有ETF和指数日线
│
└── logs/                           # 日志（gitignore）
    └── pipeline_YYYYMMDD.log
```

### 2.2 各模块文件计数

| 模块 | 文件数 | 主要功能 |
|------|--------|---------|
| 根目录 | 3 | pipeline + 状态 + 数据同步入口 |
| etf_sync/ | 7 | 数据获取/存储/同步/推送 |
| strategies/*/ | 9×14=126 | 每个策略含完整回测框架 |
| simulation/ | 10 | 模拟盘框架 + 入口 |
| 文档 | 3 | README + 对比 + 优化历程 |
| **总计** | **~150** | |

---

## 3. 核心交易逻辑（纪律红线）

### 3.1 为什么有这个规则？

这是整个项目的**纪律红线**，源于 2026-06-24 的一次修正。

回测引擎最初使用 `close[T]` 同时计算动量信号和交易执行——你"看到了 T 日的收盘价，
然后用这个收盘价成交"。但在真实世界中，收盘后市场已关闭，你无法以收盘价成交。
这是典型的 **look-ahead bias**，会导致回测收益虚高。

**修正前各策略收益虚高幅度：**

| 策略 | 修正前(close) | 修正后(open) | 差异 |
|------|-------------|-------------|------|
| momentum_ma_etf | +102.02% | **+191.0%** | ↑88.98%（修正反而更高） |
| momentum_rotation | +125.12% | **+132.93%** | ↑7.81% |
| momentum_vol_filter | +131.88% | **+116.59%** | ↓15.29%（波动率卖出受影响） |

> 任何新增策略必须在代码审查时检查 `_buy()` 和 `_sell()` 方法是否使用 `open` 而非 `close`。

### 3.2 回测引擎时序

```
T-1 日 15:00 收盘（已知数据）:
  close[T-1] ← 信号数据截止于此
  
T 日 09:30 开盘:
  open[T] ← 用此价格执行（不是 close[T]）
  检查涨停：涨停不能买入
  检查跌停：跌停不能卖出
```

**代码实现——必须遵守的模式：**

```python
# ✅ _buy() 方法：用开盘价 + 滑点
price = today_data[symbol]["open"] * (1 + SLIPPAGE)

# ✅ _sell() 方法：用开盘价 - 滑点
sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)

# ✅ 信号计算：前移 1 根 bar
signal_idx = max(1, idx - 1)  # 第 0 天没有前一日数据
momentum = compute_momentum_signals(self.etf_data, signal_idx, ...)
```

### 3.3 模拟盘引擎时序

```python
# T 日 20:00 数据同步后 → 计算信号 → 创建待执行订单
state.pending_order = {"action": "buy", "symbol": "159915", ...}
state_mgr.save(state)   # 存入 JSON 状态文件

# T+1 日 20:00 → 执行昨日订单
order = state.pending_order
state.pending_order = None  # 原子性取出
if not _check_limit_open(symbol, open[T+1], close[T]):
    execute(order, at=open[T+1])    # 用开盘价执行
```

### 3.4 两引擎对比

| 维度 | 回测 `strategies/` | 模拟盘 `simulation/` |
|------|-------------------|---------------------|
| 信号时间 | `close[T-1]` | `close[T]` |
| 执行时间 | `open[T]` | `open[T+1]` |
| 执行价格 | 开盘价 ± 滑点 | 开盘价 ± 滑点 |
| 涨跌停检查 | 无（回测忽略） | 有（涨/跌停取消订单） |
| 数据来源 | 全量历史数据库 | 增量每日加载 |
| 运行频率 | 一次性处理全部 | 每日增量运行 |
| 状态存储 | 内存 DataFrame | JSON 文件持久化 |
| 信号延迟 | 约 1 个 bar | 约 1 个 bar |

### 3.5 涨跌停规则

| ETF 类型 | 限制 | 包含 |
|---------|------|------|
| 普通 ETF | **±10%** | 510xxx, 512xxx, 513xxx, 563000 等 |
| 创业板 ETF | **±20%** | 159915, 159949 等 |
| 科创板 ETF | **±20%** | 588000, 588080, 588050 等 |

**判断逻辑：**
```python
def _check_limit_open(symbol, open_price, prev_close):
    limit_pct = 0.20 if symbol in LIMIT_20PCT_SYMBOLS else 0.10
    upper = prev_close * (1 + limit_pct)
    lower = prev_close * (1 - limit_pct)
    if open_price >= upper: return True  # 涨停不能买入
    if open_price <= lower: return True  # 跌停不能卖出
    return False
```

### 3.6 禁止事项（代码审查红线）

```python
# ❌ 禁止：用 close[T] 执行
price = today_data[symbol]["close"] * (1 + SLIPPAGE)

# ❌ 禁止：用 idx 计算信号（使用了当日 close）
momentum = compute_momentum_signals(self.etf_data, idx, ...)

# ❌ 禁止：T 日买入、T 日卖出（A 股 T+1 规则）
# ❌ 禁止：使用未来数据（如未来收益率、未来波动率）
```

---

## 4. 数据同步模块

### 4.1 数据源与可靠性

| 类型 | 主源 | 备选 | 可用性 |
|------|:----:|:----:|--------|
| ETF 日线 | **腾讯** web.ifzq.gtimg.cn | **新浪** quotes.sina.cn | 99.9%（双轨制） |
| 指数日线 | **新浪** quotes.sina.cn | — | 99.5% |
| ETF 列表 | **akshare** fund_etf_spot_em() | — | 99% |

**腾讯接口限制：** 单次请求最多返回 800 条日K线（约 3 年数据，已足够）。

### 4.2 同步标的

**ETF：** 全量场内 ETF（沪市 51xxx/56xxx/58xxx/588xxx，深市 15xxx/16xxx/159xxx），
约 1500+ 只，通过 akshare 每日自动获取列表。

**指数：** 上证50(000016)、沪深300(000300)、中证500(000905)、中证1000(000852)

### 4.3 运行模式

```bash
python main.py                     # 标准模式（20:00 后执行）
python main.py --sync-only         # 仅同步ETF日线+指数
python main.py --force             # 跳过交易日/时间门控
python main.py --backfill          # 全量回填历史（首次使用）
python main.py --list-only         # 仅更新ETF列表
```

### 4.4 时间门控

代码检查 `sync_after_hour = 20`（20:00 后才允许同步），
确保当日行情数据已全部发布。cron 在 20:00 触发 pipeline，
pipeline 依次执行同步→模拟盘。

### 4.5 双轨制数据源

ETF 数据获取采用双轨制自动切换：

```python
# 腾讯 → 新浪 切换逻辑
if active_source == "tencent":
    df = tencent_kline(code)       # 默认用腾讯
    if df is None:                 # 腾讯失败
        active_source = "sina"     # 切换新浪
        df = sina_kline(code)

if source_count["sina"] % 50 == 0: # 每50次新浪尝试恢复腾讯
    df = tencent_kline(code)
    if df is not None:
        active_source = "tencent"  # 恢复腾讯
```

### 4.6 交易日判断

```python
def is_trade_day(check_date):
    if check_date.weekday() >= 5:
        return False               # 周末过滤（最快）
    from chinese_calendar import is_workday, is_holiday
    return is_workday(check_date)  # chinese_calendar 法定节假日
```

注意：`chinese_calendar` 的 `is_workday()` 对调休工作日返回 `True`，
但 A 股在这些调休日不交易。`is_trade_day()` 已处理此情况。

### 4.7 WxPusher 推送

同步完成后推送微信通知，包含：
- ✅ 各阶段状态（成功/跳过/失败）
- 📊 双轨制统计（Tencent vs Sina 各成功多少只）
- ⏱ 耗时统计
- ❌ 失败告警

---

## 5. 数据库

### 5.1 数据库概览

- **引擎：** SQLite 3
- **路径：** `data/etf_daily.db`
- **大小：** ~100MB（截至 2026-06）
- **表数量：** 4

### 5.2 表结构

```sql
-- ETF 日线（核心表）
CREATE TABLE etf_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,          -- ETF 代码（纯数字），如 "510050"
    date TEXT NOT NULL,           -- 交易日 YYYY-MM-DD
    open REAL,                    -- 开盘价
    high REAL,                    -- 最高价
    low REAL,                     -- 最低价
    close REAL,                   -- 收盘价
    volume REAL,                  -- 成交量（份）
    UNIQUE(symbol, date)          -- 防止重复
);
CREATE INDEX idx_etf_daily_symbol ON etf_daily(symbol);
CREATE INDEX idx_etf_daily_date ON etf_daily(date);

-- 指数日线
CREATE TABLE index_daily (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,          -- 指数代码，如 "000300"
    date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(symbol, date)
);

-- ETF 列表
CREATE TABLE etf_list (
    symbol TEXT PRIMARY KEY,       -- ETF 代码
    name TEXT,                    -- ETF 名称
    delisted_date TEXT            -- 退市日期，NULL=正常交易
);

-- 同步日志
CREATE TABLE sync_log (
    date TEXT PRIMARY KEY,         -- 交易日
    status TEXT,                  -- success / failed / skipped
    etf_count INTEGER,            -- 同步 ETF 只数
    new_listed INTEGER,           -- 新增上市
    delisted INTEGER,             -- 退市
    index_count INTEGER,          -- 同步指数个数
    duration_seconds REAL,        -- 总耗时
    tencent_count INTEGER,        -- 腾讯源成功数
    sina_count INTEGER,           -- 新浪源成功数
    error_msg TEXT                -- 错误信息
);
```

### 5.3 数据量参考

| 标的类型 | 数量 | 历史数据量 | 每日增量 |
|---------|------|-----------|---------|
| ETF | ~1500+ | ~120万行 | ~1500行 |
| 指数 | 4 | ~5000行 | ~4行 |

### 5.4 查询示例

```sql
-- 查询某ETF近20日收盘价
SELECT date, close FROM etf_daily
WHERE symbol = '159915' ORDER BY date DESC LIMIT 20;

-- 查询最新交易日
SELECT MAX(date) FROM etf_daily WHERE symbol = '510050';

-- 查询某日所有ETF数据
SELECT symbol, close FROM etf_daily WHERE date = '2026-06-22';
```

---

## 6. 数据加载 API

### 6.1 回测数据加载

```python
from strategies.momentum_rotation.data import load_all_etf_data

etf_data, common_dates = load_all_etf_data(
    symbols=["510050", "510300", "510500", "512100",
             "563000", "159915", "588000"],
    start_date="2024-01-01",
    end_date="",                    # 空 = 不限制
    db_path="data/etf_daily.db",
    momentum_window=20,             # 预计算动量窗口
)

# 返回值:
# etf_data: {symbol: DataFrame, ...}
#   每个 DataFrame 包含列：
#   - date (datetime), open, high, low, close (float)
#   - volume (float), pct_chg (float)
#   - cumulative_returns, amount, amount_ma20
#   - atr, momentum, momentum_10, momentum_20
#   - symbol (str)
#
# common_dates: DatetimeIndex（所有ETF共同的交易日）
```

### 6.2 模拟盘数据加载

```python
from simulation.framework.data import load_latest_data

etf_data = load_latest_data(
    symbols=["510050", "510300", "510500", "512100",
             "563000", "159915", "588000"],
    db_path="data/etf_daily.db",
    lookback_days=60,               # 加载最近60个自然日
    momentum_window=20,             # 动量窗口
)
```

### 6.3 基准指数加载

```python
from strategies.momentum_rotation.data import load_benchmark_data

benchmark = load_benchmark_data(
    symbol="000300",               # 沪深300
    start_date="2024-01-01",
    end_date="",
    db_path="data/etf_daily.db",
    momentum_window=20,
)

# 返回值: DataFrame
# 包含列: date, close, pct_chg, cumulative_returns, momentum
```

---

## 7. 回测框架

### 7.1 每个策略的目录结构

```
strategies/策略名/
  ├── __init__.py          # 空文件
  ├── config.py            # 参数（ETF池、动量窗口、费率...）
  ├── engine.py            # 回测引擎 BacktestEngine
  ├── data.py              # 数据加载函数
  ├── momentum_signals.py  # 信号计算（非动量策略也可能需要）
  ├── cost.py              # 交易成本计算
  ├── risk.py              # 风控模块
  ├── metrics.py           # 绩效指标计算
  ├── reporter.py          # 报告 + matplotlib 图表
  └── run.py               # CLI 入口（argparse）
```

### 7.2 引擎逐日循环

```python
class BacktestEngine:
    def run(self):
        for idx in range(n):    # 遍历交易日（常见 500-600 天）
            # Step 1: 获取当日 OHLCV
            today_data = {sym: df.iloc[idx] for sym in ETF_SYMBOLS}

            # Step 2: 风控检查（仅 B/C 模式，A 模式跳过）
            if self.risk_mode != "A":
                risk_action = run_all_risk_checks(...)
                if risk_action != "none":
                    self._execute_risk_exit(...)
                    self._record_day(...)
                    continue

            # Step 3: 渐进调仓
            if self.adjustment_days_left > 0:
                self._execute_adjustment_step(idx, today_data)

            # Step 4: 计算信号（用 idx-1！）
            signal_idx = max(0, idx - 1)
            momentum = compute_momentum_signals(etf_data, signal_idx, ...)
            ranking = rank_etfs_by_momentum(momentum)
            target_etf = ranking.get(1)  # 排名第1

            # Step 5: 决策（用 open[T] 执行）
            self._make_decision(idx, today_data, hold_symbol, target_etf, momentum)

            # Step 6: 记录
            self._record_day(idx, today_data, ...)
```

### 7.3 风控模式

| 模式 | 说明 | 检查项 | 适用场景 |
|------|------|--------|---------|
| **A** | 纯信号 | 无 | 动量策略（默认） |
| **B** | 全开 | 止损5% + 移动止盈10%/5% + ATR4× + 极端回撤15% | 保守 |
| **C** | 兜底 | 仅极端回撤15% | 适度风控 |

**历史测试结论：** 模式 B 对动量策略有害——止损打断趋势，止盈提前下车。
模式 A（纯信号）配合波动率过滤的效果优于任何传统风控。

### 7.4 BacktestEngine 构造函数

```python
engine = BacktestEngine(
    initial_capital=10000,    # 初始资金
    risk_mode="A",            # 风控模式
    momentum_window=20,       # 动量窗口
    top_n=1,                  # 持有前 N 只
    dynamic_window=False,     # 动态动量窗口
)
```

---

## 8. 回测引擎核心机制详解

### 8.1 买入执行（_buy）

```python
def _buy(self, symbol, amount, idx, today_data, trade_type="买入", reason=""):
    price = today_data[symbol]["open"] * (1 + SLIPPAGE)   # 开盘价 + 滑点
    max_shares = int(amount // price // 100) * 100         # 向下取整到 100 股
    if max_shares <= 0:
        return 0                                          # 不够 1 手

    cost = max_shares * price
    commission = max(cost * COMMISSION_RATE, 0.0)
    total_cost = cost + commission

    if total_cost > self.cash:                             # 现金不够
        max_shares = int(self.cash // price // 100) * 100  # 用全部剩余现金
        if max_shares <= 0:
            return 0
        ...

    self.positions[symbol] += max_shares                  # 更新持仓
    self.cash -= total_cost                               # 扣钱
    self.open_buys.append(BuyLot(...))                    # 记录买入批次（FIFO）
```

### 8.2 卖出执行（_sell）与 FIFO 核算

```python
def _sell(self, symbol, shares, idx, today_data, ...):
    sell_price = today_data[symbol]["open"] * (1 - SLIPPAGE)  # 开盘价 - 滑点
    revenue = actual * sell_price
    commission = max(revenue * COMMISSION_RATE, 0.0)
    net_revenue = revenue - commission

    # FIFO 匹配：先买的最先卖
    for lot in self.open_buys:
        if lot.symbol != symbol or remaining <= 0:
            continue
        batch = min(lot.shares, remaining)
        batch_cost = lot.total_cost * (batch / lot.shares)
        total_buy_cost += batch_cost
        days_held = (sell_date - buy_date).days
        lot.shares -= batch
        remaining -= batch

    profit = revenue - total_buy_cost - commission
```

### 8.3 渐进调仓

当需要切换 ETF（如从 510050 → 159915）时，分 N 天逐步完成：

```python
# ADJUSTMENT_DAYS=5 时：
# 第1天：卖出 1/5 的 510050 → 用所得现金买入 159915
# 第2天：卖出 1/5 的 510050 → 买入 159915
# ...（5天后全部切换完成）

def _start_adjustment(self, from_symbol, to_symbol, idx, today_data):
    self.adjustment_from = from_symbol
    self.adjustment_to = to_symbol
    self.adjustment_days_left = ADJUSTMENT_DAYS
    self._execute_adjustment_step(idx, today_data)

def _execute_adjustment_step(self, idx, today_data):
    from_shares = self.positions.get(self.adjustment_from, 0)
    sell_shares = from_shares // self.adjustment_days_left  # 卖 1/N
    if sell_shares > 0:
        net = self._sell(self.adjustment_from, sell_shares, idx, today_data, ...)
        if net > 0:
            self._buy(self.adjustment_to, self.cash, idx, today_data, ...)
    self.adjustment_days_left -= 1
```

调仓期间动量信号仍正常计算，但 `adjustment_days_left > 0` 时不会触发新切换。

### 8.4 FIFO 批次记录

```python
@dataclass
class BuyLot:
    date: str = ""          # 买入日期
    symbol: str = ""        # ETF 代码
    shares: int = 0         # 股数
    price: float = 0.0      # 买入价（含滑点）
    total_cost: float = 0.0 # 总成本（含佣金）

# 买入时追加批次
self.open_buys.append(BuyLot(date, symbol, shares, price, total_cost))

# 卖出时从最早批次匹配
for lot in sorted(self.open_buys, key=lambda x: x.date):
    if lot.symbol == sell_symbol:
        # 先买的最先卖
```

### 8.5 切换置信度

```python
excess = target_momentum - current_momentum        # 动量差距
friction_ratio = friction_cost / trade_amount      # 摩擦成本占比
switch_threshold = max(friction_ratio, MIN_SWITCH_CONVICTION)

if excess > switch_threshold:
    self._start_adjustment(hold_symbol, target_etf, idx, today_data)
```

3% 的置信度意味着目标 ETF 动量必须比当前高出至少 3 个百分点才触发切换。

### 8.6 短期动量确认

```python
if SHORT_TERM_MOMENTUM_CHECK and idx >= 6:
    check_idx = idx - 1  # 避免 look-ahead
    # 5日动量不能为负
    tgt_5d = close[T-1] / close[T-6] - 1
    if tgt_5d <= -0.005:
        return  # 不切换（短期下跌）

    # 动量减速检查
    tgt_15d = momentum_series.get(target_etf)
    if tgt_15d > 0 and tgt_5d / 5 < tgt_15d / 15:
        return  # 不切换（虽然还在涨，但涨速变慢了）
```

### 8.7 最小持仓天数

```python
if MIN_HOLD_DAYS > 0 and self._days_since_last_switch < MIN_HOLD_DAYS:
    return  # 刚切换不久，不再次切换

# _days_since_last_switch 在切换时重置
self._days_since_last_switch = 0
# 每日递增
self._days_since_last_switch += 1
```

---

## 9. 交易成本与滑点模型

### 9.1 费率表

| 成本项 | 费率 | 说明 |
|-------|------|------|
| 佣金（买入） | 0.02% | `commission = max(amount × rate, 0.0)` |
| 佣金（卖出） | 0.02% | 同上，双边收取 |
| 滑点（买入） | 0.01% | `price = open × (1 + SLIPPAGE)` |
| 滑点（卖出） | 0.01% | `price = open × (1 - SLIPPAGE)` |
| 印花税 | 0% | ETF 免收 |
| 最低佣金 | 0 元 | ETF 免最低 5 元限制 |
| 冲击成本 | 系数 0.1 | 见下 |

### 9.2 冲击成本计算

```python
def compute_total_friction_cost(...):
    # 基于日均成交额的冲击成本
    impact_coef = IMPACT_COST_COEF  # 0.1
    amount_ma20 = df["amount_ma20"]  # 20日平均成交额
    impact = impact_coef * abs(trade_amount) / max(amount_ma20, 1)
    friction = commission + slippage + impact
    return friction
```

大额交易时冲击成本显著，小额定投时以佣金为主。

### 9.3 各策略最终成本对比

| 策略 | 总成本 | 占比 |
|------|--------|------|
| momentum_rotation | ~97 元 | ~0.97% |
| momentum_vol_filter | ~92 元 | ~0.92% |
| pair_trading | 忽略 | 忽略（合成空头） |
| mean_reversion | ~726 元 | ❌ 7.26%（切换 136 次） |

---

## 10. 绩效指标计算方法

### 10.1 MetricsCalculator 核心指标

```python
class MetricsCalculator:
    def compute(self, daily_records, trade_records, initial_capital,
                benchmark_return=None, ew_benchmark_return=None):
        # ... 返回 BacktestMetrics 对象
```

| 指标 | 公式 |
|------|------|
| 累计收益率 | `T_n / T_0 - 1` (T=总资产) |
| 年化收益率 | `(1 + 总收益)^(252/天数) - 1` |
| 最大回撤 | `min(T_t / max(T_0..t) - 1)` |
| 年化波动率 | `σ(日收益率) × √252` |
| 下行波动率 | `σ(负收益率) × √252` |
| 夏普比率 | `(E(R) - R_f) / σ(R) × √252`（R_f=3%） |
| Sortino | `(E(R) - R_f) / σ_d × √252` |
| Calmar | `年化收益率 / \|最大回撤\|` |
| 日胜率 | `正收益天数 / 总天数` |
| 交易胜率 | `盈利交易次数 / 总交易次数` |
| 盈亏比 | `平均盈利 / \|平均亏损\|` |

### 10.2 回撤计算（峰值→谷底）

```python
peak = daily_df["total_value"].cummax()
drawdown = daily_df["total_value"] / peak - 1
max_drawdown = drawdown.min()

# 回撤持续天数 = 谷底 - 峰值的时间（交易日）
dd_start = drawdown.idxmin()  # 实际是谷底位置
# 从谷底向前找到最近的峰值
```

### 10.3 等权基准

```python
def compute_equal_weight_benchmark(etf_data):
    # 每日各 ETF 日收益率平均 → 组合日收益
    # 累计 = (1 + 组合日收益).cumprod()
    # 7 只 ETF 各占 1/7
```

---

## 11. 14 种策略一览

### 11.1 策略总览

| # | 名称 | 目录 | 核心逻辑 | 类型 |
|---|------|------|---------|------|
| 1 | **纯动量轮动** | `momentum_rotation/` | 20日动量排名选最强 | 趋势 |
| 2 | **波动率过滤** ⭐ | `momentum_vol_filter/` | 年化波动率>30%时空仓 | 趋势+风控 |
| 3 | 大盘均线过滤 | `momentum_ma_filter/` | 沪深300>250日线才轮动 | 趋势 |
| 4 | 逐ETF均线过滤 | `momentum_ma_etf/` | 各ETF>自身60日线才买入 | 趋势 |
| 5 | 双动量 | `momentum_dual/` | 动量>0且排名第1才持有 | 趋势 |
| 6 | 双均线交叉 | `dual_ma_crossover/` | MA(60,120)交叉判断趋势 | 趋势 |
| 7 | 低波动率 | `low_vol_rotation/` | 持有波动率最低的ETF | 防御 |
| 8 | 均值回归 | `mean_reversion/` | 持有距均线最远的ETF | 震荡 |
| 9 | 量价配合 | `vol_price_momentum/` | 动量×成交量放大倍数 | 趋势 |
| 10 | 通道突破 | `donchian_breakout/` | 突破60日最高/最低价 | 趋势 |
| 11 | 布林带 | `bollinger_rotation/` | 布林带位置打分 | 震荡 |
| 12 | **配对交易** | `pair_trading/` | z-score价差回归，多空对冲 | 市场中性 |
| 13 | **组合策略** | `combined/` | 80%动量+20%配对 | 混合 |
| 14 | **复合动量** | `composite_momentum/` | 多因子复合打分(四因子) | 趋势 |
| 15 | **ADX趋势强度** | `adx_trend_rotation/` | ADX趋势过滤+评分 | 趋势 |
| 16 | **MACD趋势轮动** | `macd_trend_rotation/` | MACD EMA交叉+柱状图 | 趋势 |
| 17 | **RSI趋势确认** ⭐ | `rsi_trend_rotation/` | RSI(21)多头过滤+持续评分 | 趋势/保守 |
| 18 | **自适应轮动** 🆕 | `adaptive_rotation/` | 牛市动量+震荡均值回归 | **动态切换** |
| **19** | **黄金避险轮动** 🆕🔥 | `gold_safe_haven/` | 正常动量+恐慌黄金避险 | **趋势+避险** |
| **20** | **跨境轮动** 🆕🏆 | `cross_border/` | A股+美股+港股三市场动量 | **跨境轮动** |

### 11.2 各策略详细说明

**① momentum_rotation — 纯动量轮动（+133%, 夏普1.19）**
```
动量 = close[T-1] / close[T-21] - 1（20日收益率）
排名 → 全仓第1名
持有直到排名变化 + 置信度检查通过
```
参数: MOMENTUM_WINDOW=20, MIN_SWITCH_CONVICTION=3%, MIN_HOLD_DAYS=10
特点: 最纯粹的动量实现，一切其他策略的基准。

**② momentum_vol_filter — 波动率过滤轮动 ✅ 夏普最优（+117%, 夏普1.31）**
```
年化波动率 = std(日收益率, 20d) × √252
if 年化波动率 > 30%: 全部空仓
else:                 正常动量轮动
```
参数: VOL_THRESHOLD=0.30, VOL_WINDOW=20, ADJUSTMENT_DAYS=3
特点: 唯一全面超越纯动力的过滤器。高波动时离场，低波动时全力轮动。
最大回撤-23%，恢复时间仅155天（纯动量365天）。

**③ momentum_ma_filter — 大盘MA250过滤（+167%, 夏普1.37）**
```
if HS300_close > MA(HS300, 250):  正常动量轮动
else:                              全部空仓（熊市保护）
```
参数: MA_FILTER_PERIOD=250
特点: 牛市中≈纯动量（年线几乎不跌破），熊市中理论上能保护。

**④ momentum_ma_etf — 逐ETF均线过滤（+191%, 夏普1.50）**
```
for each ETF:
  if close[ETF] > MA(ETF, 60):  进入候选池
  if close[ETF] ≤ MA(ETF, 60):  排除/卖出
  候选池中按动量排名
```
特点: 收益最高，但29%时间空仓。依赖牛市环境，震荡市风险大。

**⑤ momentum_dual — 双动量（+86%, 夏普0.94）**
```
if 持仓ETF动量 ≤ 0:  卖出空仓
if 目标ETF动量 ≤ 0:  不切换
```
结论: 绝对动量条件太敏感，正常回调也触发清仓。

**⑥-⑪ 详见 STRATEGY_COMPARISON.md**

**⑫ pair_trading — 配对交易 → 纯多头风格轮动（+129%, 回撤仅10%）**
```
原版（多空对冲，需融券）：
  spread = log(price_a / price_b)
  z-score = (spread - mean) / std
  |z| > 2.0:  开仓（多空对冲）
  |z| < 0.3:  平仓获利

优化版（纯多头切换，2026-06更新）：
  发现 A 股无融券数据 → 改为纯多头风格轮动
  3对中取 |z| 最大信号，全仓切换至"便宜方"
  |z| > 3.0:  开仓买入便宜方（对称阈值，自适应）
  |z| < 0.3:  平仓到现金
  |z| > 3.0:  止损清仓（价差发散）
  结果: +128.75% (2023.03→2026.06), 夏普1.19, 回撤10.14%
```
配对: 上证50↔创业板 + 沪深300↔创业板 + 上证50↔科创50
特点: 纯多头，无需融券，真实可执行。2023-2026 每年正收益。
特点: 市场中性，不依赖大盘方向。2023年熊市仅-0.16%。

**⑭ composite_momentum — 复合动量轮动（+237%, 夏普1.77）🔥 全策略榜首**
```
四因子复合打分 = 趋势25% + 夏普25% + 质量25% + 成交量25% → Z-Score合成 → 排名选第1
```
参数: **等权四因子（2026-06-30优化）**, MIN_HOLD_DAYS=10, MARKET_MA_PERIOD=60, RISK_MODE=B
特点: 调优后四因子等权25%，全周期+237%夏普1.77，跃居所有策略第一。
2024震荡市+42%、2025牛市+57%、2026快牛+31%，每年均大幅跑赢沪深300。
夏普2.14（2025年）、1.94（2026年）显示风险调整后收益极为优秀。
因子截面Z-Score标准化确保量纲一致。市场状态过滤器(沪深300 MA60)熊市自动降仓。
**优化历程：** 原40/25/20/15权重→等权25%后，收益从+69%跃升至+237%。
等权分散优于最优单因子（与"赌马悖论"一致）。

**⑯ macd_trend_rotation — MACD趋势轮动（+123%, 夏普1.20）**
```
MACD = EMA(5) - EMA(35)  → Histogram = MACD - Signal
综合分 = MACD_norm×0.30 + Hist_norm×0.20 + (close/EMA35-1)×0.50
MACD≤0 → 空头回避；风控B模式
```
参数: EMA周期5/35/5, RISK_MODE=B（风控全开）
特点: 使用EMA指数加权（不同于SMA系动量），天然双时间框架。
2024震荡市超额+20%，全周期+123%为所有策略第二高。
纯信号模式噪音大，必须配合风控B使用。

**⑮ adx_trend_rotation — ADX趋势强度轮动（+111%, 夏普1.25）**
```
ADX≥25趋势过滤：趋势不够强→空仓（50%时间）
综合评分 = ADX×0.70 + DI优势×0.20 + 动量×0.10
```
参数: ADX_PERIOD=14, ADX_MIN_STRENGTH=25, RISK_MODE=A（纯信号）
特点: 使用ADX衡量趋势可信任度，夏普1.25。
2024震荡市超额+18.52%、低频切换仅25次/全周期。
ADX自身即过滤器，加风控会打断趋势，故用纯信号A模式。

**⑱ adaptive_rotation — 自适应轮动（+126%, 夏普1.76）🆕**
```
市场状态检测 → 动态切换交易逻辑：
  牛市（HS300>MA60+3%）：  → 动量模式（30日动量排名）
  震荡（MA60±3%之间）：    → 均值回归模式（%B超卖+RSI低位）
  熊市（HS300<MA60-3%）：  → 空仓保护
```
参数: MOM_WINDOW=30, REGIME_MA_PERIOD=60, BULL=+3%, BEAR=-3%
特点: 全市场周期自适应。2024震荡+6%（其他趋势策略亏损），2026快牛+47%。
夏普1.76（全策略第2），回撤仅-15%，25次切换。唯一能应对所有市场状态的策略。

**⑳ cross_border — 跨境轮动（+139%, 夏普1.73）🆕🏆 全策略夏普第一**
```
A股+美股+港股三市场动量轮动：
  ETF池: 沪深300 + 中证500 + 纳指 + 标普500 + 恒生
  10日动量排名 → 全仓第1名
  A股↔美股相关系数仅0.20（A股内部0.86），轮动空间大4倍
```
参数: MOMENTUM_WINDOW=10, MIN_HOLD_DAYS=10, CONVICTION=2%, RISK_MODE=B
特点: 跨市场低相关性是核心竞争力。2024+41%、2025+35%、2026YTD+9%（A股仅+0.5%时仍赚）。
每一期都跑赢基准，没有一个弱年。交易315笔，适合模拟盘运行。

### 日报批量推送 & 月度报告
- 9个原动量策略 → 1条「动量类策略合集」
- 8个新候选策略 → 1条「候选策略合集」
- 黄金避险/跨境轮动 → 独立推送
- 每月末 21:00 → 全策略月度报告
- 每晚共5条推送（3合1 + 2独立 + 1汇总）

**⑰ rsi_trend_rotation — RSI趋势确认轮动（+82%, 夏普1.15）🆕**
```
RSI(21) 非截面评分（v2）：
  Regime分(0-2): RSI>60=2, 50-60=1, <50=0
  持续分(0-2): 过去10天RSI>50占比×2
  斜率分(0-0.5): RSI 5日变化量/10(上限0.5)
  总分≥2.0才交易
```
参数: RSI_PERIOD=21, RSI_MIN_SCORE=2.0, SWITCH_CONVICTION_STD=2.0, RISK_MODE=A
特点: 保守型趋势确认系统。2024震荡市+12%、2026快牛+36%，仅44次切换。
回撤-12.99%（所有策略中最低），适合作为组合"防守配置"。
采用非截面评分（每只ETF独立计分）解决7只ETF高相关性问题。

**⑭ composite_momentum — 复合动量轮动（+237%, 夏普1.77）🔥 全策略榜首**
(2026-06-30调优：原40/25/20/15权重→等权25%，MIN_HOLD_DAYS=5→10后收益+69%→+237%。
B模式保留，等权化+长持仓是质变关键。)

**⑬ combined — 组合策略（+108%, 夏普1.11）**
```
总资金: 80% → 动量轮动（进攻）
         20% → 配对交易（防守）
每日净值 = 动量部分 + 配对部分
```
特点: 80/20 权重下回撤改善最明显。

---

## 12. 策略对比排行

### 12.1 全排行（open[T] 价格执行）

| # | 策略 | 总收益 | 夏普 | 最大回撤 | 年化 | 切换 |
|---|------|--------|------|---------|------|------|
| **1** | **composite_momentum 复合动量** 🔥 | **+237%** | **1.77** | -17% | 67% | 53 |
| **2** | **cross_border 跨境轮动** 🏆 | **+139%** | **1.73** | -20% | 43% | 315 |
| **3** | **gold_safe_haven 黄金避险** 🥇 | **+137%** | **1.12** | -17% | 43% | 87 |
| 4 | rotation 纯动量 | +133% | 1.19 | -26% | 44% | 19 |
| 5 | **adaptive_rotation 自适应轮动** | **+126%** | **1.76** | -15% | 43% | 25 |
| 6 | **vol_filter 波动率过滤** | **+117%** | **1.31** | -23% | 41% | 19 |
| 8 | combined 动量+配对 | +108% | 1.11 | -24% | 38% | — |
| 9 | bollinger 布林带 | +108% | 1.10 | -20% | 38% | 217 |
| 10 | **RSI趋势确认** | **+82%** | **1.15** | **-13%** | **32%** | **44** |
| 11 | dual 双动量 | +86% | 0.94 | -28% | 32% | 33 |
| 12 | mean_reversion 均值回归 | +80% | 0.94 | -17% | 30% | 136 |
| 9 | donchian 通道突破 | +67% | 0.85 | -21% | 26% | 108 |
| 10 | vol_price 量价 | +55% | 0.71 | -21% | 22% | 181 |
| 11 | crossover 双均线 | +45% | 0.81 | -14% | 19% | 94 |
| 12 | low_vol 低波动率 | +32% | 0.58 | -18% | 14% | 50 |
| 13 | pair_trading 配对交易(原) | +25% | — | **-6%** | 11% | 42 |
| 14 | **pair_trading 风格轮动** | **+129%** | **1.19** | **-10%** | **29%** | **47** |

### 12.2 按风险偏好选择

| 偏好 | 推荐策略 | 理由 |
|------|---------|------|
| **风险调整最优** | **composite_momentum 夏普1.25** | **多因子打分,回撤仅-17%** |
| **风险调整最优** | **composite_momentum夏普1.25/ADX夏普1.25** | **多因子或ADX自由选择** |
| 追求绝对收益 | ma_etf +191% | 收益最高，但回撤大 |
| **风险调整最优** | **vol_filter 夏普1.31** | 每单位风险回报最高 |
| 追求稳定性 | pair_trading(原) 回撤6% | 市场中性，熊市不亏（需融券） |
| 风格轮动 | **pair_trading 纯多头** +129% | 无需融券，2023年起每年正收益 |
| 攻守兼备 | combined +108% | 80%动量+20%配对 |

---

## 13. 模拟盘框架（T+1 待执行订单）

### 13.1 框架架构

```
simulation/
  ├── framework/          ← 通用模板块（与策略无关）
  │   ├── state.py        ← JSON 持久化（原子写入）
  │   ├── data.py         ← 数据加载
  │   ├── broker.py       ← 模拟交易
  │   ├── engine.py       ← 每日流程编排
  │   ├── risk.py         ← 风控检查
  │   └── notify.py       ← 微信推送
  └── strategies/         ← 各策略适配层
      └── momentum_rotation/
          ├── config.py   ← 策略模拟配置
          └── daily.py    ← 每日入口
```

### 13.2 T+1 待执行订单状态机

```
                     ┌──────────────┐
                     │  idle（无订单）│
                     └──────┬───────┘
                            │ 信号触发（风控或动量）
                            ▼
                     ┌──────────────┐
                     │  pending_order│ 存入 state.pending_order
                     │  (buy/sell/   │{"action":"buy","symbol":"159915",...}
                     │   switch)     │
                     └──────┬───────┘
                            │ 次日交易日 20:00
                            ▼
              ┌─────────────────────────────┐
              │  execute _pending_order()    │
              │  用 open[T+1] 执行           │
              │  检查涨/跌停                 │
              └──┬──────────────────┬───────┘
        可成交   │                  │  被封锁
                ▼                  ▼
        ┌──────────────┐  ┌──────────────┐
        │ 订单成交      │  │ 订单取消      │
        │ 更新持仓/资金  │  │ 记录锁定原因   │
        │ broker.buy/   │  │ 持仓不变      │
        │  sell()       │  │              │
        └──────┬───────┘  └──────┬───────┘
               │                 │
               ▼                 ▼
        ┌──────────────────────────────────┐
        │ 估值 → 用今日 close 更新总资产     │
        │ 风控 → 检查止损/止盈/极端回撤     │
        │ 信号 → 计算动量 → 新 pending_order│
        │ 持久化 → state_mgr.save()        │
        └──────────────────────────────────┘
```

### 13.3 待执行订单格式

```python
# 买入订单
{"action": "buy", "symbol": "159915", "reason": "动量信号开仓", "created": "2026-06-22"}

# 卖出订单（风控触发）
{"action": "sell", "symbol": "510050", "reason": "高波动空仓避险", "created": "2026-06-22"}

# 切换订单（双边检查）
{"action": "switch", "sell_symbol": "510050", "buy_symbol": "159915",
 "reason": "动量切换", "created": "2026-06-22"}
```

### 13.4 状态文件格式

文件路径: `simulation/output/state_策略名.json`

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
     "shares": 2300, "price": 4.2914, "amount": 9872.54,
     "commission": 1.97, "pnl": 0}
  ],
  "days_since_switch": 10,
  "peak_value": 12000.00,
  "pending_order": null,
  "strategy_name": "momentum_rotation"
}
```

### 13.5 模拟盘运行日志 CSV

每个策略独立 CSV 文件，位于 `simulation/output/sim_log_{策略id}.csv`：

```
日期,策略,操作,持仓标的,持仓名称,持仓数量,持仓均价,现金,市值,总资产,累计收益率,订单执行,明日待执行
2026-06-30,动量轮动,空仓，发出买入信号←明日开盘执行,588000,科创50,0,,10000,0,10000,0.00%,,明日开盘买入科创50[588000]
2026-07-01,动量轮动,执行昨买入：开盘买入科创50 5600股@1.7562,588000,科创50,5600,1.7565,163,9974,10137,1.37%,✅买入,-
```

**切换操作描述：**
- 信号日：`发出切换信号(A→B)←明日开盘卖出A并买入B`
- 执行日：`执行切换：开盘卖出A X股@Y → 买入B Z股@W`

**已有策略追记：** CSV 首次生成时自动从 `state_{id}.json` 读取当前状态写入历史起点行。

### 13.7 每日流程

```python
# 1. 交易日判断
if not is_trading_day(today_str):
    push_report(STRATEGY_NAME, [f"{today_str} 非交易日，跳过"])
    return

# 2. 数据到位检查
latest_day = get_latest_trading_day(ETF_SYMBOLS)
if latest_day != today_str:
    push_error_alert(STRATEGY_NAME, f"数据尚未同步完成")
    return  # 跳过（明天再试）

# 3. 加载数据
etf_data = load_latest_data(ETF_SYMBOLS, lookback_days=40, momentum_window=20)

# 4. 运行引擎
engine = DailySimEngine(state_mgr, broker, signal_func, rank_func, ...)
report = engine.run_daily(etf_data, today_idx, today_str)

# 5. 推送日报
push_daily_report(STRATEGY_NAME, build_report(report))
```

### 13.6 涨跌停检查

```python
def _check_limit_open(symbol, open_price, prev_close):
    """涨停不能买入，跌停不能卖出。"""
    limit_pct = 0.20 if symbol in LIMIT_20PCT_SYMBOLS else 0.10
    upper = prev_close * (1 + limit_pct)
    lower = prev_close * (1 - limit_pct)
    if open_price >= upper:
        return True, "涨停"
    if open_price <= lower:
        return True, "跌停"
    return False, ""
```

切换订单需双边检查：卖A买B，A跌停或B涨停都取消整个切换。

---

## 14. 管线编排器 pipeline

### 14.1 cron 配置

```bash
# crontab -l 查看
# ETF 数据同步 + 模拟盘管线
0 20 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest && \
  /home/zhulei/anaconda3/bin/python pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

### 14.2 pipeline 执行流程

```
pipeline.py
  │
  ├─ 交易日判断（chinese_calendar）
  │   非交易日 → PipelineStatus.finish("skipped") → 结束
  │
  ├─ PipelineStatus.reset() → 创建当日记录
  ├─ PipelineStatus.add_step() → 注册各步骤
  │
  ├─ Step 1: ETF 数据同步
  │   cmd: ["main.py", "--sync-only"]
  │   required: True (必需)
  │   timeout: 10800 秒 (3 小时)
  │   成功 → 继续
  │   失败 → 管线终止，模拟盘不运行
  │
  ├─ Step 2: 动量轮动模拟盘（必需）
  │   cmd: ["-m", "simulation.strategies.momentum_rotation.daily"]
  │   required: True
  │   timeout: 600 秒
  │   成功 → 推送日报到微信
  │
  ├─ Step 3~7: 可选策略模拟盘（失败不影响管线）
  │   ├─ 复合动量 (composite_momentum)
  │   ├─ MACD趋势轮动 (macd_trend_rotation)
  │   ├─ RSI趋势确认 (rsi_trend_rotation) 🆕
  │   ├─ ADX趋势强度 (adx_trend_rotation)
  │   ├─ 波动率过滤 (momentum_vol_filter)
  │   ├─ 配对交易 (pair_trading)
  │   └─ 组合策略 (combined)
  │   timeout: 300-600 秒
  │
  ├─ PipelineStatus.finish("completed") 或 ("failed")
  ├─ push_pipeline_summary() → WxPusher 推送管线汇总
  └─ push_strategy_summary() → WxPusher 推送策略汇总日报 🆕
         读取 simulation/output/ 下所有 state_{id}.json 和 sim_log_{id}.csv
         计算各策略累计/年化收益、夏普、回撤、胜率
         一条消息汇总全部模拟盘策略的横向对比
```

### 14.3 汇总日报推送格式

```
📊 ETF模拟盘策略汇总 | 07-01
════════════════════════════════════════
❶ 复合动量
  累计+119% 年化+39% 夏普1.25 回撤-17%
  启动06-30 胜率50% 120天 | 📈 588000×4700 ⏩明日开盘买入…
...
────────────────────────────────────────
✅ 管线: 8/9 完成 | 耗时 4分2秒
```

新策略自动发现：有 state_{id}.json 即自动纳入汇总，无需代码修改。
起步初期显示"数据收集中"（不足2天）或"年化起步"（不足20天）。

### 14.4 子进程实时日志

```python
# 使用 Popen 替代 run，实现实时输出
proc = subprocess.Popen([PYTHON] + cmd, stdout=PIPE, stderr=PIPE, text=True)

for line in iter(proc.stdout.readline, ""):
    print(f"    {line.rstrip()}")          # 打印到主日志
    stdout_lines.append(line.rstrip())     # 同时保存

proc.wait(timeout=timeout)
```

### 14.4 pipeline_status.json

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

### 14.5 自我修复

```python
class PipelineStatus:
    def needs_rerun(self) -> bool:
        """检测上次运行状态，异常中断则重跑。"""
        raw = self.load()
        if not raw:
            return True  # 无当日记录 → 需要跑
        ps = raw.get("pipeline_status", "")
        return ps in ("running", "failed")  # 中断或失败 → 重跑

    def reset(self):
        """创建当日空白记录（覆盖旧的）。"""
```

---

## 15. 策略开发指南

### 15.1 完整流程

```
Phase 1: 回测验证（strategies/ 下）
  ├─ 1. 复制 momentum_rotation 目录
  ├─ 2. 修改 config.py（参数、ETF池、信号参数）
  ├─ 3. 修改 engine.py（信号计算逻辑、决策逻辑）
  ├─ 4. python -m strategies.新策略.run → 回测
  ├─ 5. 检查：signal_idx=max(1,idx-1), open[T] 执行
  ├─ 6. 迭代调参（单变量 → 多变量）
  ├─ 7. 分年验证（2024/2025/2026）
  └─ 8. 满意后进入 Phase 2

Phase 2: 模拟盘部署（simulation/strategies/ 下）
  ├─ 1. 新建 simulation/strategies/新策略/
  ├─ 2. 创建 config.py
  ├─ 3. 创建 daily.py（入口）
  ├─ 4. 添加到 pipeline.py STEPS 列表
  └─ 5. 第二天 cron 自动运行

Phase 3: 文档更新
  ├─ 更新 STRATEGY_COMPARISON.md
  ├─ 更新 OPTIMIZATION_HISTORY.md
  └─ git push
```

### 15.2 回测引擎检查清单

```
[ ] signal_idx = max(1, idx - 1)  — 信号用前一日
[ ] today_data[...]["open"]        — 执行用开盘价
[ ] 涨停/跌停检查（模拟盘用）
[ ] 数据不足时返回默认值（非崩溃）
[ ] run.py 支持 --start/--end/--money/--tag
[ ] 无 np.nan 传播到决策逻辑
[ ] config.py 参数可配置
[ ] 不要在 __init__ 中写死逻辑（请在 run() 中）
```

### 15.3 模拟盘入口检查清单

```
[ ] 交易日判断（is_trading_day）
[ ] 最新交易日 == 运行日（数据到位检查）
[ ] 行情数据足够（idx >= momentum_window）
[ ] 涨跌停检查（_check_limit_open）
[ ] 日报推送（可读性强）
```

### 15.4 命名规范

| 项目 | 规范 | 示例 |
|------|------|------|
| 策略目录 | snake_case | momentum_vol_filter |
| 策略参数 | UPPER_SNAKE | MIN_SWITCH_CONVICTION |
| 引擎方法 | snake_case | _make_decision_single |
| 数据类 | PascalCase | BacktestMetrics |
| 文件名 | snake_case | momentum_signals.py |

---

## 16. 配置详解

### 16.1 全局配置（.env）

```bash
# WxPusher（微信推送）
WXPUSHER_TOKEN=AT_hKGG0UfwrCP7bpcsO8cbQkrc4bZ9G3RX
WXPUSHER_TOPIC_IDS=["39277"]
```

### 16.2 策略通用参数

（每个策略的 config.py 中包含的参数）

| 参数 | 类型 | 默认值 | 说明 | 影响 |
|------|------|--------|------|------|
| `ETF_SYMBOLS` | list[str] | 7只ETF | ETF 代码列表 | 核心 |
| `INITIAL_CAPITAL` | float | 10000 | 初始资金 | 资金规模 |
| `MOMENTUM_WINDOW` | int | 20 | 动量窗口(交易日) | 高 |
| `MIN_SWITCH_CONVICTION` | float | 0.03 | 切换置信度(3%) | 高 |
| `MIN_HOLD_DAYS` | int | 10 | 最小持仓天数 | 中 |
| `SHORT_TERM_MOMENTUM_CHECK` | bool | True | 短期动量确认 | 中 |
| `ADJUSTMENT_DAYS` | int | 5(或3) | 渐进调仓周期 | 中 |
| `COMMISSION_RATE` | float | 0.0002 | 佣金率 | 低 |
| `SLIPPAGE` | float | 0.0001 | 滑点率 | 低 |
| `RISK_MODE` | str | "A" | 风控模式(A/B/C) | 高 |
| `TOP_N` | int | 1 | 持有前N只 | 高 |
| `DB_PATH` | str | "data/etf_daily.db" | 数据库路径 | — |
| `OUTPUT_DIR` | str | 策略output | 输出目录 | — |

### 16.3 波动率过滤特有参数（momentum_vol_filter）

| 参数 | 类型 | 值 | 说明 | 影响 |
|------|------|----|------|------|
| `VOL_THRESHOLD` | float | 0.30 | 年化波动率阈值 | 高 |
| `VOL_WINDOW` | int | 20 | 波动率计算窗口 | 中 |

### 16.4 配对交易特有参数（pair_trading）

| 参数 | 类型 | 值 | 说明 | 影响 |
|------|------|----|------|------|
| `PAIRS` | list[dict] | 3对 | 配对列表 | 高 |
| `ZSCORE_PERIOD` | int | 60 | z-score 统计窗口 | 高 |
| `ZSCORE_OPEN` | float | **3.0** | 开仓阈值（优化: 2.0→3.0） | 高 |
| `ZSCORE_OPEN_GROWTH` | float\|None | None | 买成长侧阈值（None=对称） | 中 |
| `ZSCORE_OPEN_VALUE` | float\|None | None | 买价值侧阈值（None=对称） | 中 |
| `ZSCORE_CLOSE` | float | 0.3 | 平仓阈值 | 中 |
| `ZSCORE_STOP` | float | 3.0 | 止损阈值 | 中 |

### 16.5 组合策略参数（combined）

| 参数 | 类型 | 值 | 说明 |
|------|------|----|------|
| `TOTAL_CAPITAL` | float | 10000 | 总资金 |
| `MOMENTUM_PCT` | float | 0.80 | 动量占比 |
| `PAIR_PCT` | float | 0.20 | 配对占比 |

### 16.6 ETF 池

```python
ETF_POOL = {
    "510050": "上证50ETF（华夏）",     # 大盘价值
    "510300": "沪深300ETF（华泰柏瑞）", # 大中盘
    "510500": "中证500ETF（南方）",    # 中盘
    "512100": "中证1000ETF（南方）",   # 小盘
    "563000": "中证2000ETF（华夏）",   # 微盘
    "159915": "创业板ETF（易方达）",    # 成长
    "588000": "科创50ETF（华夏）",     # 科技成长
}
```

---

## 17. 运行指南

### 17.1 首次使用

```bash
# 1. 配置环境
pip install pandas numpy matplotlib pydantic-settings \
            python-dotenv rich chinese_calendar \
            requests akshare wxpusher

# 2. 配置 .env 文件
# WXPUSHER_TOKEN 和 WXPUSHER_TOPIC_IDS

# 3. 回填历史数据（约10-20分钟）
python main.py --backfill

# 4. 运行基准策略
python -m strategies.momentum_rotation.run
```

### 17.2 运行回测

```bash
# 基本用法
python -m strategies.momentum_vol_filter.run

# 自定义参数
python -m strategies.momentum_rotation.run \
  --start 2024-01-01 \              # 开始日期
  --end 2026-06-22 \                # 结束日期（空=最新）
  --money 10000 \                   # 初始资金
  --tag mytest                      # 输出标记

# 可用策略列表
python -m strategies.momentum_rotation.run           # 纯动量
python -m strategies.momentum_vol_filter.run          # 波动率过滤
python -m strategies.pair_trading.run                  # 配对交易
python -m strategies.combined.run                      # 组合策略
python -m strategies.momentum_ma_filter.run             # MA过滤
python -m strategies.momentum_ma_etf.run                # 逐ETF均线
python -m strategies.momentum_dual.run                  # 双动量
python -m strategies.dual_ma_crossover.run               # 双均线
python -m strategies.mean_reversion.run                  # 均值回归
python -m strategies.low_vol_rotation.run                # 低波动率
python -m strategies.bollinger_rotation.run               # 布林带
python -m strategies.donchian_breakout.run                # 通道突破
python -m strategies.vol_price_momentum.run               # 量价配合
```

### 17.3 运行模拟盘

```bash
# 手动测试（如当日有数据）
python -m simulation.strategies.momentum_rotation.daily

# 正式运行由 pipeline.py 自动触发
python pipeline.py
```

### 17.4 回测输出

每次回测生成至 `strategies/策略名/output/YYYYMMDD_HHMMSS_tag/`：

```
输出目录/
  ├── daily_records.csv       # 逐日净值（595行）
  ├── trade_records.csv       # 交易明细
  ├── metrics.csv             # 绩效指标汇总
  ├── equity_curve.png        # 净值曲线（含基准对比）
  ├── drawdown.png            # 回撤曲线
  ├── holding_heatmap.png     # 持仓热力图
  └── monthly_returns.png     # 月度收益热力图
```

### 17.5 cron 配置

```bash
# 查看当前 crontab
crontab -l

# ETF 数据同步 + 模拟盘管线（交易日 20:00）
0 20 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest && /home/zhulei/anaconda3/bin/python pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

---

## 18. 日志与监控

### 18.1 日志文件

| 文件 | 内容 | 位置 |
|------|------|------|
| pipeline_YYYYMMDD.log | 管线运行日志 | logs/ |
| 同步日志 | 数据同步状态 | 数据库 sync_log 表 |

### 18.2 日志级别

- pipeline.py: 子进程 stdout/stderr 实时打印
- etf_sync: rich 彩色日志
- simulation: logging.INFO 级别

### 18.3 WxPusher 推送

| 推送内容 | 触发条件 | 包含信息 |
|---------|---------|---------|
| 管线汇总 | pipeline 运行结束 | 各步骤状态/耗时/错误 |
| 模拟盘日报 | 模拟盘运行成功 | 操作/持仓/收益/排名 |
| 错误告警 | 运行异常 | 错误堆栈/日期 |

---

## 19. 依赖环境与版本兼容

### 19.1 Python 版本

- **开发环境：** Python 3.12（Anaconda）
- 兼容：Python 3.10+

### 19.2 核心依赖

| 包 | 用途 | 必备 |
|---|------|------|
| pandas | 数据处理 | ✅ |
| numpy | 数值计算 | ✅ |
| matplotlib | 图表 | ✅ |
| requests | 数据源HTTP请求 | ✅ |
| akshare | ETF列表获取 | ✅ |
| wxpusher | 微信推送 | ✅ |
| chinese_calendar | 交易日判断 | ✅ |
| python-dotenv | .env加载 | ✅ |
| pydantic-settings | 配置管理 | ⚠️ 仅etf_sync需要 |
| rich | 彩色日志 | ⚠️ 仅etf_sync需要 |

### 19.3 安装

```bash
# 一次性安装全部依赖
pip install -r requirements.txt 2>/dev/null || \
pip install pandas numpy matplotlib pydantic-settings \
            python-dotenv rich chinese_calendar \
            requests akshare wxpusher
```

---

## 20. 已知 Bug 与修复记录

| ID | 问题 | 严重 | 发现 | 修复 |
|----|------|------|------|------|
| B001 | SQL ORDER BY 拼接错误 | 🔴 | 2026-06-23 | ORDER BY 移至查询末尾 |
| B002 | 回测用 close 执行（look-ahead） | 🔴 | 2026-06-24 | 全改 open[T] 执行 |
| B003 | 模拟盘风控卖出状态丢失 | 🔴 | 2026-06-24 | 移除多余的 load() |
| B004 | 热力图色图溢出 | 🟡 | 2026-06-23 | 动态扩展色图颜色 |
| B005 | signal.py 命名冲突 | 🟡 | 2026-06-23 | 重命名 momentum_signals.py |
| B006 | today_opened 永不重置 | 🟡 | 2026-06-24 | 每日运行时重置为 False |
| B007 | 独立年份回测数据倒序 | 🔴 | 2026-06-23 | ORDER BY 位置修正 |

> 详细修复过程见 `OPTIMIZATION_HISTORY.md` 第 1 节。

---

## 21. 故障排除

### 21.1 数据同步失败

```bash
# 强制同步（跳过交易日/时间检查）
python main.py --force

# 查看同步日志
sqlite3 data/etf_daily.db "SELECT * FROM sync_log ORDER BY date DESC LIMIT 5;"

# 重新回填（如数据损坏）
python main.py --backfill
```

### 21.2 回测报错

```bash
# ImportError: cannot import name 'XXX'
# → config.py 缺少该参数，检查对应策略的 config.py

# ValueError: 数据库无数据
# → 先运行 python main.py --backfill 回填数据

# KeyError: 'momentum'
# → data.py 未计算 momentum 列，检查 momentum_window 参数
```

### 21.3 模拟盘问题

```bash
# 报 "非交易日，跳过"
# → 今日确实非交易日，或 chinese_calendar 判断错误

# 报 "数据尚未同步完成"
# → 数据同步可能失败，检查 pipeline 日志

# 状态文件损坏
# → 删除 simulation/output/state_*.json，下次会自动初始化
```

### 21.4 常见错误及解决方案

| 错误 | 原因 | 解决 |
|------|------|------|
| `ImportError: cannot import name 'MOMENTUM_WINDOW'` | config.py 缺少参数 | 添加 `MOMENTUM_WINDOW = 20` |
| `ValueError: 没有加载到任何 ETF 数据` | 数据库为空 | 运行 `python main.py --backfill` |
| `KeyError: 'momentum'` | DataFrame 无动量列 | 检查 momentum_window 是否传入 |
| `IndentationError` | 代码缩进问题 | 检查 engine.py 编辑后的缩进 |
| `FileNotFoundError: data/etf_daily.db` | 数据库路径错误 | 从项目根目录运行命令 |

---

## 22. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| **bar** | bar | K线柱，一个交易日的数据 |
| **动量** | momentum | `close[N] / close[N-M] - 1`（N日收益率） |
| **相对动量** | relative momentum | ETF动量 - 基准指数动量 |
| **绝对动量** | absolute momentum | ETF自身N日收益 > 0 |
| **z-score** | z-score | `(当前值 - 均值) / 标准差` |
| **look-ahead bias** | look-ahead bias | 使用了未来数据的回测偏差 |
| **夏普比率** | Sharpe ratio | `(E(R)-Rf)/σ(R)` 风险调整收益 |
| **最大回撤** | max drawdown | 从峰值到谷底的最大跌幅 |
| **渐进调仓** | gradual adjustment | 分多日完成买卖，降低冲击 |
| **FIFO** | FIFO | 先进先出成本核算 |
| **待执行订单** | pending order | 今日产生、明日开盘执行的订单 |
| **涨跌停** | limit up/down | ±10%(普通ETF) / ±20%(创业/科创板) |
| **双轨制** | dual source | 腾讯→新浪自动切换的数据源策略 |

---

## 23. 常见问题

### Q: 回测和模拟盘的结果为什么不同？

交易时序不同导致执行价格不同：
- 回测：`close[T-1]` 信号 → `open[T]` 执行（同日开/收盘价）
- 模拟盘：`close[T]` 信号 → `open[T+1]` 执行（隔日）
时序上等价（信号都比执行提前约 1 bar），但具体价格不同。

### Q: 为什么 7 只 ETF 的池子没有 look-ahead bias？

创业板和科创板是 A 股市场的**独立市场层次**，不是"2025年涨得好所以加上的行业板块"。
7 只 ETF（5只宽基+2只成长板）共同构成 A 股的完整市场覆盖。
真正有 bias 问题的是手动添加行业 ETF（如半导体ETF），这类做法已被排除。

### Q: 为什么不在回调时止损？

动量策略的盈利核心是"持有趋势最强的标的"。止损会打断趋势，
导致收益大幅下降。波动率过滤（高波动时离场）是比价格止损更有效的风控手段——
它衡量的是"趋势是否可靠"，而非"价格跌了多少"。

### Q: 数据同步失败了怎么办？

```bash
python main.py --force     # 强制尝试
python main.py --backfill  # 重新回填全部数据
```

### Q: pipeline 中断了怎么恢复？

下次 cron 触发时，`PipelineStatus.needs_rerun()` 检测到上次状态为
`running` 或 `failed`，会自动重置并从头运行。无需手动干预。

### Q: 如何添加新策略？

1. 复制 `strategies/momentum_rotation/` → 修改 `config.py` + `engine.py`
2. 运行 `python -m strategies.新策略.run` 验证
3. 满意后在 `simulation/strategies/` 下建对应目录
4. 添加到 `pipeline.py` 的 `STEPS` 列表
5. 更新文档

### Q: 哪个策略最好？

看目标：
- **追求夏普（推荐）：** `momentum_vol_filter`（夏普1.31，收益117%）
- **追求收益：** `momentum_ma_etf`（+191%，但回撤大）
- **追求稳定：** `pair_trading`（回撤6%，市场中性）
- **攻守兼备：** `combined`（+108%，回撤24%）

### Q: 状态文件损坏了怎么办？

```bash
rm -f simulation/output/state_momentum_rotation.json
# 下次运行会自动初始化新状态（从空仓开始）
```

### Q: 如何查看某只 ETF 的数据范围？

```bash
sqlite3 data/etf_daily.db "SELECT MIN(date), MAX(date), COUNT(*) FROM etf_daily WHERE symbol='510050';"
```

---

> **文档体系：**
> - 本文档（README.md）：项目总览与操作指南（~1500行）
> - `strategies/STRATEGY_COMPARISON.md`：策略对比分析
> - `strategies/OPTIMIZATION_HISTORY.md`：优化历程记录（~1800行）
>
> **最后更新：** 2026-06-24
> **GitHub：** [github.com/zhuleimed/etf-daily-sync-and-backtest](https://github.com/zhuleimed/etf-daily-sync-and-backtest)
