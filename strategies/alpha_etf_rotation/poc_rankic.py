"""广发复原策略 — 阶段1 POC：个股因子聚合到指数 → ETF 截面 RankIC 验证

核心假设（广发方法）：个股 Alpha 因子按成分股权重聚合到指数层面，
ETF 因子承载个股截面信息——这是 019 LSTM-POC（只用 ETF 价格，IC=-0.018）
失败的根本差异点。

流程：
  1. 加载 sequoia-x 个股日线（仅成分股子集 ~2180 只）
  2. 计算 8 个核心因子（动量/反转/量价/估值/波动）
  3. 按成分股权重聚合 → 7 个 ETF 因子值
  4. 月度 WF：每月最后交易日因子 → 下月收益 → 截面 RankIC
  5. 决策点①：月均 RankIC > 0.03 才继续阶段 2（择时）

铁律：py312 / 断点续跑（按月落盘）/ 详尽日志。

用法:
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m strategies.alpha_etf_rotation.poc_rankic
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
        logging.FileHandler(PROJECT_ROOT / "logs" / "alpha_poc.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("alpha_poc")

OUTPUT_DIR = Path(__file__).parent / "output"
RESULT_FILE = OUTPUT_DIR / "rankic_results.json"
CONSTITUENTS_FILE = OUTPUT_DIR / "constituents.json"

STOCK_DB = "/public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/data/sequoia_v2.db"
ETF_DB = PROJECT_ROOT / "data" / "etf_daily.db"

# 因子定义：(名称, 计算列, 窗口) —— 全部用滞后数据，无 look-ahead
FACTORS = [
    ("mom_20", "close", 20),      # 20 日动量
    ("mom_60", "close", 60),      # 60 日动量
    ("rev_5", "close", 5),        # 5 日反转（短期涨幅，负相关预期）
    ("rev_20", "close", 20),      # 20 日反转
    ("vol_20", "close", 20),      # 20 日波动率（std）
    ("amt_chg_20", "amount", 20), # 20 日成交额变化率
    ("ep", "peTTM", None),        # 盈利收益率 1/PE（估值）
    ("bp", "pbMRQ", None),        # 账面市值比 1/PB
]


def load_constituents() -> dict[str, dict]:
    data = json.loads(CONSTITUENTS_FILE.read_text(encoding="utf-8"))
    return data


def load_stock_factors(stock_db: str, constituents: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """加载成分股日线并计算因子 → {股票代码: DataFrame(日期+因子列)}"""
    # 收集所有成分股
    all_codes = set()
    for idx_info in constituents.values():
        all_codes.update(idx_info["constituents"].keys())
    logger.info(f"成分股总数: {len(all_codes)}")

    conn = sqlite3.connect(stock_db)
    # 分批查询避免 SQL 过长
    codes_list = sorted(all_codes)
    stock_data: dict[str, pd.DataFrame] = {}
    batch = 200
    for i in range(0, len(codes_list), batch):
        chunk = codes_list[i:i + batch]
        placeholders = ",".join("?" * len(chunk))
        df = pd.read_sql_query(
            f"SELECT symbol, date, close, volume, amount, peTTM, pbMRQ "
            f"FROM stock_daily WHERE symbol IN ({placeholders}) ORDER BY symbol, date",
            conn, params=chunk,
        )
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date").reset_index(drop=True)
            g["date"] = pd.to_datetime(g["date"])
            close = g["close"]
            for fname, col, window in FACTORS:
                if window is None:
                    if col == "peTTM":
                        g[fname] = 1.0 / g["peTTM"].replace(0, np.nan)
                    elif col == "pbMRQ":
                        g[fname] = 1.0 / g["pbMRQ"].replace(0, np.nan)
                elif fname.startswith("rev"):
                    g[fname] = close.pct_change(window)
                elif fname == "vol_20":
                    g[fname] = close.pct_change().rolling(window).std()
                elif fname == "amt_chg_20":
                    g[fname] = g["amount"].pct_change(window)
                else:  # mom
                    g[fname] = close.pct_change(window)
            stock_data[sym] = g[["date"] + [f[0] for f in FACTORS]]
    conn.close()
    return stock_data


def aggregate_to_etf(
    stock_data: dict[str, pd.DataFrame],
    constituents: dict[str, dict],
    etf_close: dict[str, pd.Series],
) -> pd.DataFrame:
    """个股因子按成分股权重聚合 → 每日 7 个 ETF 的因子值矩阵。

    返回: {因子名: DataFrame(index=日期, columns=ETF代码)}
    """
    # 日期并集（取 ETF 交易日）
    all_dates = sorted(set().union(*[set(s.index) for s in etf_close.values()]))
    dates = pd.DatetimeIndex(all_dates)

    etf_by_index = {}  # index_code -> etf symbol
    from strategies.alpha_etf_rotation.data_prep import INDEX_MAP
    for etf, (idx, _) in INDEX_MAP.items():
        etf_by_index[idx] = etf

    factor_dfs: dict[str, pd.DataFrame] = {}
    for fname, _, _ in FACTORS:
        mat = pd.DataFrame(index=dates, columns=list(etf_by_index.values()), dtype=float)
        for idx_code, info in constituents.items():
            etf = etf_by_index.get(idx_code)
            if etf is None:
                continue
            weights = info["constituents"]  # {code: w}
            # 逐股票取因子序列，按权重加权（重采样到 ETF 交易日）
            total = pd.Series(0.0, index=dates)
            wsum = 0.0
            for code, w in weights.items():
                sd = stock_data.get(code)
                if sd is None or fname not in sd.columns:
                    continue
                s = sd.set_index("date")[fname].reindex(dates)
                total = total.add(s * w, fill_value=0.0)
                wsum += w
            mat[etf] = total / wsum if wsum > 0 else np.nan
        factor_dfs[fname] = mat
    return factor_dfs


def run_rankic_poc() -> None:
    logger.info("=" * 60)
    logger.info("广发复原 POC：个股因子聚合 → ETF 截面 RankIC")
    logger.info(f"Python: {sys.executable}")
    t0 = time.time()

    constituents = load_constituents()
    logger.info(f"指数: {len(constituents)} 个")

    # ── 个股因子 ──
    stock_data = load_stock_factors(STOCK_DB, constituents)
    logger.info(f"个股因子完成: {len(stock_data)} 只, 耗时 {time.time()-t0:.0f}s")

    # ── ETF 收盘价 ──
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

    # ── 聚合因子 ──
    factor_dfs = aggregate_to_etf(stock_data, constituents, etf_close)
    logger.info(f"因子聚合完成: {len(factor_dfs)} 个因子, 耗时 {time.time()-t0:.0f}s")

    # ── 月度 WF：每月最后交易日因子 → 下月 ETF 收益 → RankIC ──
    results = load_existing()
    done = {r["month"] for r in results.get("months", []) if r.get("status") == "ok"}

    dates = factor_dfs["mom_20"].index
    months = dates.to_period("M").unique()
    month_last = {p: dates[dates.to_period("M") == p].max() for p in months}

    for p in months:
        mkey = str(p)
        if mkey in done:
            continue
        ref = month_last[p]
        if ref < pd.Timestamp("2021-01-01"):
            continue  # 训练预热：2020 年数据
        # 下月收益
        next_month = (p + 1).to_timestamp("M")  # 下月第一天
        nxt = month_last.get(p + 1)
        if nxt is None:
            continue
        # ETF 下月收益 = ref 到 nxt 的收益
        rets = {}
        for etf, s in etf_close.items():
            if ref in s.index and nxt in s.index:
                rets[etf] = s.loc[nxt] / s.loc[ref] - 1
        ret_s = pd.Series(rets)

        # 各因子 RankIC
        month_ics = {}
        for fname, mat in factor_dfs.items():
            if ref not in mat.index:
                continue
            fv = mat.loc[ref]
            valid = fv.dropna().index.intersection(ret_s.dropna().index)
            if len(valid) < 5:
                continue
            from scipy.stats import spearmanr
            ic, _ = spearmanr(fv[valid], ret_s[valid])
            month_ics[fname] = float(ic) if not np.isnan(ic) else None

        # 等权因子合成（全部因子简单平均 z-score）
        zs = pd.DataFrame({f: mat.loc[ref] for f, mat in factor_dfs.items()})
        zs = (zs - zs.mean()) / (zs.std() + 1e-9)
        combo = zs.mean(axis=1).dropna()
        valid = combo.index.intersection(ret_s.dropna().index)
        if len(valid) >= 5:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(combo[valid], ret_s[valid])
            month_ics["combo"] = float(ic) if not np.isnan(ic) else None

        results.setdefault("months", []).append({
            "month": mkey, "status": "ok", "ref": ref.strftime("%Y-%m-%d"),
            "ics": month_ics,
        })
        save_results(results)  # 铁律四
        combo_ic = month_ics.get("combo")
        logger.info(f"[{mkey}] combo IC={combo_ic:+.4f}" if combo_ic is not None else f"[{mkey}] 无有效IC")

    # ── 汇总 ──
    oks = [m for m in results.get("months", []) if m.get("status") == "ok"]
    if oks:
        logger.info("=" * 60)
        logger.info(f"有效月份: {len(oks)}")
        for fname in [f[0] for f in FACTORS] + ["combo"]:
            ics = [m["ics"].get(fname) for m in oks if m["ics"].get(fname) is not None]
            if ics:
                arr = np.array(ics)
                logger.info(f"  {fname:<12} 月均IC={arr.mean():+.4f} ICIR={arr.mean()/arr.std():.2f} "
                            f"IC>0占比={(arr>0).mean()*100:.0f}% (n={len(arr)})")
        results["summary"] = {
            fname: round(float(np.mean([m["ics"][fname] for m in oks if m["ics"].get(fname) is not None])), 4)
            for fname in [f[0] for f in FACTORS] + ["combo"]
            if any(m["ics"].get(fname) is not None for m in oks)
        }
        save_results(results)
    logger.info(f"总耗时: {time.time()-t0:.0f}s")


def load_existing() -> dict:
    if RESULT_FILE.exists():
        return json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    run_rankic_poc()
