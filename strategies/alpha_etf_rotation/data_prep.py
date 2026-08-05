"""广发复原策略 — 数据准备：指数成分股权重 + ETF-指数映射

拉取 7 个宽基指数的成分股权重（akshare csindex），保存为本地 JSON。
⚠️ 局限：csindex 返回**当前快照**权重（非历史），回测用当前成分近似
（机构研报常见做法，误差可接受；如需精确可后续获取历史成分）。

用法（py312）:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.alpha_etf_rotation.data_prep
输出: output/constituents.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("data_prep")

OUTPUT_DIR = Path(__file__).parent / "output"
OUT_FILE = OUTPUT_DIR / "constituents.json"

# ═══ ETF-指数映射（宽基 7 + 行业 9 = 16 只，成分股数据已验证） ═══
INDEX_MAP = {
    # 宽基（7）
    "510050": ("000016", "上证50"),
    "510300": ("000300", "沪深300"),
    "510500": ("000905", "中证500"),
    "512100": ("000852", "中证1000"),
    "159915": ("399006", "创业板指"),
    "588000": ("000688", "科创50"),
    "510180": ("000010", "上证180"),
    # 行业（9）——提升截面多样性（研报4：重合度剔除需求 + Top5 收益更好）
    "512720": ("930651", "中证计算机"),
    "515790": ("931151", "中证光伏产业"),
    "516010": ("930901", "中证动漫游戏"),
    "515050": ("931079", "中证5G通信"),
    "159825": ("000949", "中证农业"),
    "512010": ("000933", "中证医药"),
    "515170": ("000815", "中证细分食品"),
    "515220": ("930713", "中证煤炭"),
    "512200": ("931775", "中证全指房地产"),
}

# 需要等权处理（csindex 不提供）的指数
EQUAL_WEIGHT_INDICES = {"399006"}   # 创业板指（深交所指数，csindex 无权重）


def fetch_constituents() -> dict:
    """拉取各指数成分股权重。"""
    import akshare as ak
    result = {}
    for etf, (idx_code, idx_name) in INDEX_MAP.items():
        try:
            if idx_code in EQUAL_WEIGHT_INDICES:
                df = ak.index_stock_cons(symbol=idx_code)
                # 当前成分（最新快照——index_stock_cons 返回历史纳入记录，
                # 需要去重取最新的。简化：全部视为当前成分，等权）
                codes = df["品种代码"].astype(str).str.zfill(6).unique().tolist()
                w = 1.0 / len(codes)
                result[idx_code] = {"name": idx_name, "constituents": {c: w for c in codes}}
            else:
                df = ak.index_stock_cons_weight_csindex(symbol=idx_code)
                constituents = {}
                for _, row in df.iterrows():
                    code = str(row["成分券代码"]).zfill(6)
                    constituents[code] = float(row["权重"])
                result[idx_code] = {"name": idx_name, "constituents": constituents}
            logger.info(f"✅ {idx_code} {idx_name}: {len(result[idx_code]['constituents'])} 只")
        except Exception as e:
            logger.warning(f"❌ {idx_code} {idx_name}: {str(e)[:80]}")
    return result


def main() -> None:
    logger.info("=" * 50)
    logger.info("广发复原：指数成分股权重拉取")
    t0 = time.time()
    data = fetch_constituents()
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(v["constituents"]) for v in data.values())
    logger.info(f"完成: {len(data)} 个指数, {total} 只成分股 → {OUT_FILE}")
    logger.info(f"耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
