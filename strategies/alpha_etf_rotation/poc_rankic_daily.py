"""广发复原策略 — 阶段1 POC v2（日频 RankIC × 映射方式对比）

吸取研报4《基于多因子加权的ETF轮动策略》的实证结论：
  1. 等权映射 + 权重阈值（60%/80%）优于纯权重加权
  2. 因子需低相关（mom_20/rev_5/vol_20/amt_chg_20 四因子）
  3. 估值因子在宽基 ETF 上反向（实测），剔除

三种映射模式：
  weight        — 纯权重加权（原版）
  threshold80   — 权重降序累加至 80% 的成分股，等权映射
  threshold60   — 权重降序累加至 60% 的成分股，等权映射

决策点①：任一模式日均 RankIC > 0.03（广发日频 5.03%）
铁律：py312 / 断点续跑 / 详尽日志。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ.pop("KMP_AFFINITY", None)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "alpha_poc_daily_v2.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("alpha_daily_v2")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "rankic_daily_v2_results.json"
CONSTITUENTS_FILE = OUTPUT_DIR / "constituents.json"
STOCK_DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"

# 低相关 4 因子（研报4：因子需覆盖不同数据特征）
FACTORS = [
    ("mom_20", "close", 20),
    ("rev_5", "close", 5),
    ("vol_20", "close", 20),
    ("amt_chg_20", "amount", 20),
]
FORWARD_DAYS = [5, 21]
START = "2021-01-01"
MAPPING_MODES = ["weight", "threshold80", "threshold60"]
THRESHOLDS = {"threshold80": 0.80, "threshold60": 0.60}


def load_constituents() -> dict:
    return json.loads(CONSTITUENTS_FILE.read_text(encoding="utf-8"))


def load_stock_factors(stock_db: str, constituents: dict) -> dict[str, pd.DataFrame]:
    all_codes = set()
    for info in constituents.values():
        all_codes.update(info["constituents"].keys())
    logger.info(f"成分股总数: {len(all_codes)}")
    conn = sqlite3.connect(stock_db)
    codes_list = sorted(all_codes)
    stock_data: dict[str, pd.DataFrame] = {}
    for i in range(0, len(codes_list), 200):
        chunk = codes_list[i:i + 200]
        ph = ",".join("?" * len(chunk))
        df = pd.read_sql_query(
            f"SELECT symbol, date, close, volume, amount FROM stock_daily "
            f"WHERE symbol IN ({ph}) ORDER BY symbol, date",
            conn, params=chunk,
        )
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            g["date"] = pd.to_datetime(g["date"])
            close = g["close"]
            g["mom_20"] = close.pct_change(20)
            g["rev_5"] = close.pct_change(5)
            g["vol_20"] = close.pct_change().rolling(20).std()
            g["amt_chg_20"] = g["amount"].pct_change(20)
            stock_data[sym] = g[["date"] + [f[0] for f in FACTORS]]
    conn.close()
    return stock_data


def _mapped_weights(weights: dict[str, float], mode: str) -> dict[str, float]:
    """按映射模式生成聚合权重。"""
    if mode == "weight":
        return weights
    thr = THRESHOLDS[mode]
    # 权重降序，累加至阈值
    ordered = sorted(weights.items(), key=lambda x: -x[1])
    cum, kept = 0.0, {}
    for code, w in ordered:
        if cum >= thr:
            break
        kept[code] = w
        cum += w
    # 等权映射
    n = len(kept)
    return {c: 1.0 / n for c in kept} if n else weights


def aggregate_to_etf(stock_data: dict, constituents: dict,
                     etf_close: dict[str, pd.Series],
                     mode: str) -> dict[str, pd.DataFrame]:
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP
    etf_by_index = {idx: etf for etf, (idx, _) in INDEX_MAP.items()}
    all_dates = sorted(set().union(*[set(s.index) for s in etf_close.values()]))
    dates = pd.DatetimeIndex(all_dates)

    factor_dfs: dict[str, pd.DataFrame] = {}
    for fname, _, _ in FACTORS:
        mat = pd.DataFrame(index=dates, columns=list(etf_by_index.values()), dtype=float)
        for idx_code, info in constituents.items():
            etf = etf_by_index.get(idx_code)
            if etf is None:
                continue
            mw = _mapped_weights(info["constituents"], mode)
            total = pd.Series(0.0, index=dates)
            for code, w in mw.items():
                sd = stock_data.get(code)
                if sd is None or fname not in sd.columns:
                    continue
                s = sd.set_index("date")[fname].reindex(dates)
                total = total.add(s * w, fill_value=0.0)
            mat[etf] = total
        factor_dfs[fname] = mat
    return factor_dfs


def compute_daily_ics(factor_dfs: dict, etf_close: dict[str, pd.Series],
                      mode: str) -> dict:
    from scipy.stats import spearmanr
    dates = factor_dfs["mom_20"].index
    dates = dates[dates >= pd.Timestamp(START)]
    close_df = pd.DataFrame(etf_close).reindex(dates)

    results = {"mode": mode, "months": []}
    done = set()
    if RESULT_FILE.exists():
        old = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        if old.get("mode") == mode:
            results = old
            done = {r["date"] for r in results.get("months", []) if r.get("status") == "ok"}

    fwd = {N: close_df.shift(-N) / close_df - 1 for N in FORWARD_DAYS}

    for date in dates:
        dkey = date.strftime("%Y-%m-%d")
        if dkey in done:
            continue
        row_ics = {}
        for N in FORWARD_DAYS:
            ret_s = fwd[N].loc[date].dropna()
            if len(ret_s) < 5:
                continue
            for fname, mat in factor_dfs.items():
                fv = mat.loc[date]
                valid = fv.dropna().index.intersection(ret_s.index)
                if len(valid) < 5:
                    continue
                ic, _ = spearmanr(fv[valid], ret_s[valid])
                if not np.isnan(ic):
                    row_ics.setdefault(f"fwd{N}", {})[fname] = float(ic)
            # 等权合成
            d = row_ics.get(f"fwd{N}", {})
            if len(d) >= 3:
                vals = np.array([d[f] for f in [x[0] for x in FACTORS] if f in d])
                row_ics[f"fwd{N}"]["combo"] = float(vals.mean())
        if row_ics:
            results.setdefault("months", []).append({"date": dkey, "status": "ok", "ics": row_ics})
            save_results(results)
    return results


def save_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


def summarize(results: dict) -> None:
    oks = [m for m in results.get("months", []) if m.get("status") == "ok"]
    if not oks:
        return
    logger.info(f"模式[{results['mode']}] 有效天数: {len(oks)}")
    for N in FORWARD_DAYS:
        for fname in [x[0] for x in FACTORS] + ["combo"]:
            ics = [m["ics"].get(f"fwd{N}", {}).get(fname) for m in oks]
            ics = [x for x in ics if x is not None]
            if len(ics) < 30:
                continue
            arr = np.array(ics)
            logger.info(f"  未来{N}日 {fname:<10} 日均IC={arr.mean():+.4f} ICIR={arr.mean()/arr.std():.2f} "
                        f"IC>0={(arr>0).mean()*100:.0f}% (n={len(arr)})")


def main() -> None:
    logger.info("=" * 60)
    logger.info("广发复原 POC v2：日频 RankIC × 映射方式对比")
    t0 = time.time()

    constituents = load_constituents()
    stock_data = load_stock_factors(STOCK_DB, constituents)
    logger.info(f"个股因子({len(FACTORS)}个): {len(stock_data)} 只, {time.time()-t0:.0f}s")

    conn = sqlite3.connect(str(ETF_DB))
    etf_close = {}
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP
    for etf in INDEX_MAP:
        df = pd.read_sql_query(
            "SELECT date, close FROM etf_daily WHERE symbol=? ORDER BY date",
            conn, params=(etf,),
        )
        df["date"] = pd.to_datetime(df["date"])
        etf_close[etf] = df.set_index("date")["close"]
    conn.close()

    for mode in MAPPING_MODES:
        factor_dfs = aggregate_to_etf(stock_data, constituents, etf_close, mode)
        logger.info(f"[{mode}] 聚合完成: {time.time()-t0:.0f}s")
        results = compute_daily_ics(factor_dfs, etf_close, mode)
        summarize(results)

    logger.info(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
