# 019 ETF 日线数据同步

场内 ETF 日线数据同步项目（第一阶段：数据同步框架）。

**独立项目**：与 004_sequoia-x 项目完全独立，通过独立 cron 调度。

## 数据来源

| 数据类型 | 主源 | 备选 | 说明 |
|---------|:----:|:----:|------|
| ETF 日线 | **腾讯** `web.ifzq.gtimg.cn` | **Sina** `quotes.sina.cn` | 双轨制自动切换 |
| 指数日线 | **Sina** `quotes.sina.cn` | — | 腾讯不支持指数K线 |
| ETF 列表 | **akshare** `fund_etf_spot_em()` | — | 约 1500+ 只 |

## 同步标的

### ETF（全量场内 ~1500+ 只）
通过 akshare 每日自动获取列表，含沪市（51xxx/56xxx/58xxx/588xxx）和深市（15xxx/16xxx/159xxx）ETF。

### 指数（4 个 + 1 个 ETF 代理）

| 名称 | 代码 | 说明 |
|------|:----:|:----:|
| 上证50 | sh000016 → `000016` | Sina 接口获取 |
| 沪深300 | sh000300 → `000300` | Sina 接口获取 |
| 中证500 | sh000905 → `000905` | Sina 接口获取 |
| 中证1000 | sh000852 → `000852` | Sina 接口获取 |
| 中证2000 | **563000** ETF | 华夏中证2000ETF 代理（腾讯接口获取）|

## 目录结构

```
019_etf_daily_sync_and_backtest/
├── main.py              # 主入口（6种运行模式）
├── .env                 # 配置（数据库路径、同步时间、WxPusher Token）
├── etf_sync/            # 核心模块
│   ├── config.py        # 配置管理（pydantic-settings）
│   ├── logger.py        # 日志（rich）
│   ├── data_source.py   # 双源数据获取（腾讯→Sina 双轨制）
│   ├── engine.py        # SQLite 数据库引擎
│   ├── sync.py          # 同步管理器（ETF 列表/日线/指数）
│   └── notify.py        # WxPusher 微信推送
├── data/                # SQLite 数据库（gitignore）
└── logs/                # 日志（gitignore）
```

## 使用方法

### 环境要求

```bash
pip install pydantic-settings python-dotenv rich chinese_calendar requests pandas akshare wxpusher
```

### 运行模式

```bash
# 标准模式：ETF 列表 + ETF 日线 + 指数日线（20:00 后自动执行）
python main.py

# 仅同步数据（跳过 ETF 列表更新）
python main.py --sync-only

# 强制模式（跳过交易日/时间检查，用于测试/补跑）
python main.py --force

# 全量回填历史数据（从 2024-01-01）
python main.py --backfill

# 仅更新 ETF 列表
python main.py --list-only
```

### 定时任务（cron）

本项目**不依赖** 004_sequoia-x 的管线编排，通过独立 cron 调度：

```
# 每个交易日 20:00 执行 ETF 数据同步
0 20 * * 1-5 cd /public/home/hpc/zhulei/superman/quant/code/019_etf_daily_sync_and_backtest && /home/zhulei/anaconda3/envs/zhulei_py312/bin/python main.py >> logs/cron_$(date +\%Y\%m\%d).log 2>&1
```

## 数据存储

使用 SQLite 数据库，默认路径 `data/etf_daily.db`。

### etf_daily 表（ETF 日线）
| 字段 | 类型 | 说明 |
|------|:----:|------|
| symbol | TEXT | ETF 代码（纯数字） |
| date | TEXT | 交易日 YYYY-MM-DD |
| open / high / low / close | REAL | OHLC 价格 |
| volume | REAL | 成交量（手） |

### index_daily 表（指数及 ETF 代理日线）
同上结构，物理隔离。

### etf_list 表（ETF 列表）
`symbol`, `name`, `delisted_date`

### sync_log 表（同步日志）
`date`, `status`, 各阶段计数, `duration_seconds`, `error_msg`

## 消息推送

同步完成后通过 **WxPusher** 推送微信通知，包含：
- 各阶段状态（✅/⏭️/❌）
- 双轨制数据源统计（腾讯 vs Sina 各成功多少只）
- 耗时统计
- 失败告警

## 双轨制数据源

ETF 数据获取采用双轨制自动切换：
1. **默认使用 Tencent 接口**
2. Tencent 失败 → 自动切换到 Sina
3. 每 50 次 Sina 请求尝试恢复 Tencent 一次
4. 日志/推送中标识当前活跃数据源

## 后续计划

- [x] 数据同步框架（Phase 1-3 管线 + WxPusher 推送）
- [ ] 策略回测模块（ETF 策略信号、回测引擎）
- [ ] 交易信号推送
