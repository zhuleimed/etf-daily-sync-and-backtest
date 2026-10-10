"""选股打分指纹 —— 用来识别"两个策略其实是同一个选股大脑"。

背景
----
2026-09-25 的接线事故：8 个候选策略的 `daily.py` 是同一模板复制出来的，
`signal_func` 全部指向动量轮动，于是 7 个策略产出逐日完全相同的资金曲线。
事后加的巡检用"每天总资产是否相同"来推断"是不是同一个大脑"——
但**净值和大脑并不等价**，2026-10-09 就因此误报了一次：

    · 两个策略买到了同一只 ETF（复合动量/Neural/尾部风险都买 510050）
      → 净值逐分相同，但打分完全不同；
    · 零长假清零后所有空仓策略都是 10000.00 纹丝不动
      → 13 个策略的净值也完全相同。

所以改成直接比对**打分向量**——它才是"选股大脑"的指纹。
同一份打分 ⇒ 同一个大脑；不同打分 ⇒ 不同大脑（哪怕买到了同一只）。

退化约定（重要）
----------------
**全 0 分**（或全为 NaN / 空）返回空串，表示"这天没有可辨识的信号"。
这类输出只说明"全市场都不满足开仓条件"，MACD/RSI/ADX 在熊市天天如此——
任何观望中的策略都会给出同一个全 0 向量，拿它比对必然误报。

注意判据是"全 0"而不是"全部相同"：尾部风险触发时只给最低波动那只 1.0 分
（其余为 NaN），这是**一个明确的决策**（"买它"），必须保留指纹。
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def signal_fingerprint(scores: Any, target: str | None = None) -> str:
    """把"打分向量 + 最终选中标的"压成一个短指纹字符串。

    `target` 必须一起进指纹：有些策略的打分是**共用**的，区别只在选第几名。
    例如中位数#2 和纯动量的打分量一模一样（都来自同一份动量），差别是它把
    第 2 名提到第 1 名——只看打分会把这两个策略误判成同一个大脑。

    Parameters
    ----------
    scores : pd.Series | dict | None
        signal_func 的返回值：index=ETF 代码，values=打分。NaN 会被剔除。
    target : str | None
        排名函数最终给出的第 1 名（引擎的 ranking[1]）。

    Returns
    -------
    str
        12 位十六进制指纹；**全 0 分（或空）返回 ""**（见模块开头"退化约定"）。
    """
    try:
        s = pd.Series(scores).dropna()
    except Exception:
        return ""
    if s.empty:
        return ""

    # 取 6 位小数：真克隆是同一份代码算出来的，浮点值逐位相同；
    # 这里的舍入只是防住无意义的末位噪声，不影响判定。
    pairs = sorted((str(k), round(float(v), 6)) for k, v in s.items())
    if all(v == 0.0 for _, v in pairs):  # 全 0 分 = 全市场不满足开仓条件
        return ""

    payload = ",".join(f"{k}={v:.6f}" for k, v in pairs) + f"|target={target or ''}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
