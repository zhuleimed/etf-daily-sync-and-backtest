#!/usr/bin/env python3
"""Neural Momentum 回测入口（= run_neural 的规范别名）。

2026-10-02 修复：本文件原是从 momentum_rotation 复制的**遗留副本**——
  跑的是纯动量引擎、并把结果写进 `strategies/momentum_rotation/output/`
  （其 config.OUTPUT_DIR 也指向那边），导致
  ① `python -m strategies.neural_momentum.run` 的"neural 回测"其实=动量（曲线逐位相同）；
  ② 结果污染另一个策略的输出目录。
现在改为转发到真正的混合评分入口 run_neural.py（score = w×动量z + (1-w)×神经z，w 取配置）。

用法（与 run_neural 完全一致）：
  python -m strategies.neural_momentum.run --start 2024-01-01 --end 2026-08-04 --tag myrun
  python -m strategies.neural_momentum.run --weight-w 1.0 ...   # 1.0 = 纯动量基准
"""
from __future__ import annotations

from .run_neural import main

if __name__ == "__main__":
    main()
