# 复合动量轮动策略 (Composite Momentum Rotation)

## 策略简介

基于**多因子复合打分系统**的 ETF 轮动策略。将传统单一动量因子扩展为四个维度的综合评分：
趋势动量、夏普比率、趋势质量、成交量确认。同时引入沪深300指数市场状态过滤器。

## 快速运行

```bash
# 默认回测（纯信号模式A）
python strategies/composite_momentum/run.py

# 推荐：风控全开模式B
python strategies/composite_momentum/run.py --risk-mode B

# 指定回测区间
python strategies/composite_momentum/run.py --start 2024-01-01 --end 2026-06-29

# 带标记保存不同参数组合
python strategies/composite_momentum/run.py --risk-mode B --tag final
```

## 文件结构

```
strategies/composite_momentum/
├── __init__.py              # 包初始化
├── config.py                # 策略参数配置
├── momentum_signals.py      # ⭐ 核心：多因子复合打分系统
├── data.py                  # 数据加载（含指数数据）
├── engine.py                # 回测引擎
├── risk.py                  # 风控模块
├── cost.py                  # 交易成本模型
├── metrics.py               # 绩效指标计算
├── reporter.py              # 报告生成（CSV + 图表）
├── run.py                   # 运行入口
├── README.md                # 本文件
├── IMPROVEMENTS.md          # 改进记录
└── output/                  # 回测输出目录
```

## 模拟盘

```bash
# 作为模拟盘独立运行
python -m simulation.strategies.composite_momentum.daily

# 通过管线运行（数据同步后自动）
python pipeline.py
```

## 策略参数一览

| 参数 | 默认值 | 说明 |
|------|--------|------|
| FACTOR_WEIGHT_TREND | 0.40 | 趋势动量因子权重 |
| FACTOR_WEIGHT_SHARPE | 0.25 | 夏普比率因子权重 |
| FACTOR_WEIGHT_QUALITY | 0.20 | 趋势质量因子权重 |
| FACTOR_WEIGHT_VOLUME | 0.15 | 成交量确认因子权重 |
| MIN_HOLD_DAYS | 5 | 最小持仓天数 |
| SWITCH_CONVICTION_STD | 0.5 | 切换阈值（标准差倍数） |
| RISK_MODE | A | 风控模式（推荐 B） |

## 基本面要求

- 数据来源：SQLite `data/etf_daily.db`
- ETF 日线：etf_daily 表（7只宽基 ETF）
- 指数数据：index_daily 表（沪深300 `000300`）
- 只需要 OHLCV 数据，无需财务字段
