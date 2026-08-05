"""
宽基ETF动量轮动策略回测框架

策略逻辑：
  每日计算 5 只宽基 ETF 的 N 日动量，持有动量最强的标的。
  切换时校验摩擦成本，采用渐进式调仓（分5天完成）。

模块：
  config.py   — 策略参数配置
  data.py     — 数据加载（SQLite → DataFrame）
  signal.py   — 动量信号计算与排序
  cost.py     — 摩擦成本校验
  risk.py     — 风控（浮动止盈/ATR止损/极端回撤）
  engine.py   — 回测引擎核心（逐日事件驱动）
  metrics.py  — 绩效指标计算
  reporter.py — 报告生成（CSV + 图表）
  run.py      — 命令行入口

用法：
  python strategies/momentum_rotation/run.py
"""
