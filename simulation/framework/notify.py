"""模拟盘日报推送模块 — 通过 WxPusher 推送到微信。

配置信息（WxPusher Token & Topic ID）从环境变量或 .env 文件读取。
"""

from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv
from wxpusher import WxPusher

load_dotenv()


def _get_config() -> tuple[str, list[str]]:
    """获取 WxPusher 配置。"""
    token = os.getenv("WXPUSHER_TOKEN", "")
    topic_ids_raw = os.getenv("WXPUSHER_TOPIC_IDS", '["39277"]')
    import json
    topic_ids = json.loads(topic_ids_raw)
    return token, topic_ids


def send_message(title: str, content: str, content_type: int = 1) -> bool:
    """通过 WxPusher 推送消息。

    Args:
        title: 消息标题。
        content: 消息正文。
        content_type: 1=纯文本，2=HTML。

    Returns:
        是否推送成功。
    """
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

    Args:
        strategy_name: 策略名称。
        report_lines: 日报内容行列表。

    Returns:
        是否推送成功。
    """
    today = date.today().strftime("%Y-%m-%d")
    title = f"📊 {strategy_name} | {today}"
    content = "\n".join([f"📊 {strategy_name} 日报 | {today}", ""] + report_lines)
    return send_message(title, content)


def push_error_alert(strategy_name: str, error: str) -> bool:
    """推送错误告警。"""
    today = date.today().strftime("%Y-%m-%d")
    content = f"❌ {strategy_name} 运行异常 | {today}\n\n{error}"
    return send_message(f"❌ {strategy_name} 异常", content)
