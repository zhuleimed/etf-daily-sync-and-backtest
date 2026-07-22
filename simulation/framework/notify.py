"""模拟盘日报推送模块 — 通过 WxPusher 推送到微信。

配置信息（WxPusher Token & Topic ID）从环境变量或 .env 文件读取。

支持批量聚合模式：设置 BATCH_MODE=1 环境变量后，push_daily_report
不会立即推送，而是收集到临时文件。最后调用 flush_batch_reports()
一次性推送合并后的日报。
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from wxpusher import WxPusher

load_dotenv()

# 批量聚合用的临时文件
_BATCH_FILE = Path(__file__).resolve().parent.parent.parent / "batch_reports.json"
_BATCH_CANDIDATE_FILE = Path(__file__).resolve().parent.parent.parent / "batch_candidate_reports.json"

def _get_batch_file():
    """根据 BATCH_MODE 返回对应的批量文件路径。"""
    mode = os.environ.get("BATCH_MODE", "")
    if mode == "candidate":
        return _BATCH_CANDIDATE_FILE
    return _BATCH_FILE


def _get_config() -> tuple[str, list[str]]:
    """获取 WxPusher 配置。"""
    token = os.getenv("WXPUSHER_TOKEN", "")
    topic_ids_raw = os.getenv("WXPUSHER_TOPIC_IDS", '["39277"]')
    topic_ids = json.loads(topic_ids_raw)
    return token, topic_ids


def send_message(title: str, content: str, content_type: int = 1) -> bool:
    """通过 WxPusher 推送消息。"""
    token, topic_ids = _get_config()
    if not token:
        print("[WxPusher] 未配置 Token，跳过推送")
        return False

    try:
        result = WxPusher.send_message(
            content=content,
            token=token,
            topic_ids=topic_ids,
            content_type=content_type,
        )
        if result.get("code") == 1000:
            return True
        print(f"[WxPusher] 推送失败: {result}")
        return False
    except Exception as e:
        print(f"[WxPusher] 推送异常: {e}")
        return False


def push_daily_report(
    strategy_name: str,
    report_lines: list[str],
) -> bool:
    """推送策略日报。

    当 BATCH_MODE=1 时，报告写入临时文件而非立即推送。
    """
    # ── 批量模式：收集到文件 ──
    if os.environ.get("BATCH_MODE", "") in ("1", "candidate"):
        bf = _get_batch_file()
        reports = []
        if bf.exists():
            try:
                reports = json.loads(_BATCH_FILE.read_text(encoding="utf-8"))
            except Exception:
                reports = []
        reports.append({
            "name": strategy_name,
            "lines": report_lines,
        })
        bf.write_text(json.dumps(reports, ensure_ascii=False, indent=2))
        return True

    # ── 正常模式：立即推送 ──
    today = date.today().strftime("%Y-%m-%d")
    title = f"📊 {strategy_name} | {today}"
    content = "\n".join([f"📊 {strategy_name} 日报 | {today}", ""] + report_lines)
    return send_message(title, content)


def _flush_batch_file(batch_file: Path, batch_label: str) -> bool:
    """将指定批量文件的日报合并为一条消息推送，然后清理。"""
    if not batch_file.exists():
        return False

    try:
        reports = json.loads(batch_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not reports:
        batch_file.unlink(missing_ok=True)
        return False

    today = date.today().strftime("%Y-%m-%d")
    lines = [f"📊 {batch_label} | {today}", "═" * 40, ""]

    for i, r in enumerate(reports):
        lines.append(f"▎{r['name']}")
        lines.append("─" * 35)
        for line in r["lines"]:
            if not line.startswith("📊") and "日报" not in line:
                lines.append(line)
        if i < len(reports) - 1:
            lines.append("")

    content = "\n".join(lines)
    result = send_message(f"📊 {batch_label} | {today}", content)
    batch_file.unlink(missing_ok=True)
    return result


def flush_batch_reports(batch_label: str = "动量类策略合集") -> bool:
    """推送原有动量类批量报告。"""
    return _flush_batch_file(_BATCH_FILE, batch_label)


def flush_candidate_reports(batch_label: str = "候选策略合集") -> bool:
    """推送新纳入候选策略的批量报告。"""
    return _flush_batch_file(_BATCH_CANDIDATE_FILE, batch_label)


def push_error_alert(strategy_name: str, error: str) -> bool:
    """推送错误告警。"""
    today = date.today().strftime("%Y-%m-%d")
    content = f"❌ {strategy_name} 运行异常 | {today}\n\n{error}"
    return send_message(f"❌ {strategy_name} 异常", content)
