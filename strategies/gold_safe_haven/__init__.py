"""
黄金避险轮动策略 (Gold Safe-Haven Rotation)

核心思路：
  正常市场 → 动量轮动（7只宽基ETF中选最强）
  恐慌市场 → 全仓黄金ETF(518880)避险

恐慌指数由三个子指标Z-score加权合成：
  1. 最大5日跌幅（权重40%）— 跌得有多狠
  2. 波动率比21d/63d（权重30%）— 波动率是否飙升
  3. 下跌广度（权重30%）— 多少只ETF一起跌

模块：
  - config: 策略参数（ETF池、恐慌阈值、动量窗口等）
  - data: 数据加载和辅助列计算
  - panic_signals: 恐慌指数计算
  - momentum_signals: 动量信号计算（复用momentum_rotation）
  - engine: 双模式回测引擎
  - cost: 交易摩擦成本
  - risk: 风控检查（黄金止损）
  - metrics: 绩效指标计算
  - reporter: 报告生成
  - run: CLI入口
"""
