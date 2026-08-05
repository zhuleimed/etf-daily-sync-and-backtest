"""生成逐日风险分 CSV（供回测/模拟盘读取）。

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.risk_timing.predict_scores
输出: strategies/risk_timing/output/risk_scores.csv (date, score)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategies.risk_timing.quick_test import load_pool_data
from strategies.risk_timing.backtest_switch import compute_risk_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("predict_scores")

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    pool_df = load_pool_data()
    risk = compute_risk_scores(pool_df)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / "risk_scores.csv"
    risk.to_csv(out, header=["score"], index_label="date")
    logger.info(f"风险分已保存: {out} ({len(risk)} 天, 均值 {risk.mean():.3f})")


if __name__ == "__main__":
    main()
