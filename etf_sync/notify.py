"""消息推送模块：通过 WxPusher 发送微信通知。

与 004_sequoia-x 的 wxpusher 推送模式一致。
"""

from __future__ import annotations

from datetime import date, datetime

from etf_sync.config import Settings
from etf_sync.logger import get_logger

logger = get_logger(__name__)


def push_sync_summary(settings: Settings, result: dict) -> None:
    """推送同步完成摘要到微信。

    Args:
        settings: 系统配置（需含 wxpusher_token）。
        result: run_full 返回的结果字典（含 phases, duration_seconds 等）。
    """
    if not settings.wxpusher_token:
        logger.info("未配置 WxPusher Token，跳过推送")
        return

    today_str: str = date.today().strftime("%m-%d")
    now_str: str = datetime.now().strftime("%H:%M")
    phases: dict = result.get("phases", {})
    r1: dict = phases.get("etf_list", {})
    r2: dict = phases.get("etf_daily", {})
    r3: dict = phases.get("index_daily", {})
    elapsed: float = result.get("duration_seconds", 0)

    status_icon: str = "✅" if result.get("status") == "ok" else "⚠️"
    etf_icon: str = "✅" if r2.get("status") == "ok" else "⏭️" if r2.get("status") == "skipped" else "❌"
    idx_icon: str = "✅" if r3.get("status") == "ok" else "⏭️" if r3.get("status") == "skipped" else "❌"

    # 数据源追踪
    source_info = ""
    tencent_ok = r2.get("tencent_count", 0)
    sina_ok = r2.get("sina_count", 0)
    if tencent_ok > 0 or sina_ok > 0:
        source_info = f"\n📡 数据源: 腾讯{tencent_ok}只 / Sina{sina_ok}只"

    message: str = (
        f"019 ETF 数据同步完成 | {today_str}\n\n"
        f"状态: {status_icon} {result.get('status', 'unknown')}\n"
        f"执行: {now_str}\n\n"
        f"ETF 列表: {r1.get('total', 0)} 只"
        f"（+{len(r1.get('new_listed', []))}/-{len(r1.get('delisted', []))}）\n"
        f"ETF 日线: {etf_icon} {r2.get('etf_count', 0)} 只\n"
        f"指数日线: {idx_icon} {r3.get('index_count', 0)} 个{source_info}\n"
        f"耗时: {elapsed:.0f} 秒"
    )

    if result.get("error"):
        message += f"\n\n⚠️ 异常: {result['error']}"

    try:
        from wxpusher import WxPusher

        r = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("推送同步摘要成功")
        else:
            logger.warning(f"推送失败: {r}")
    except Exception as e:
        logger.warning(f"推送异常: {e}")


def push_skip_notice(settings: Settings, reason: str = "") -> None:
    """推送非交易日跳过通知。

    Args:
        settings: 系统配置。
        reason: 跳过原因描述。
    """
    if not settings.wxpusher_token:
        return

    today_str: str = date.today().strftime("%m-%d")
    now_str: str = datetime.now().strftime("%H:%M")
    body: str = f"⏭️ {reason}" if reason else "⏭️ 非交易日，跳今日同步"

    message: str = (
        f"019 ETF 数据同步 | {today_str}\n\n"
        f"{body}\n"
        f"{now_str} 跳过"
    )

    try:
        from wxpusher import WxPusher

        r = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("推送跳过通知成功")
    except Exception as e:
        logger.warning(f"推送跳过通知异常: {e}")


def push_error_alert(settings: Settings, phase: str, error: str) -> None:
    """推送同步失败告警。

    Args:
        settings: 系统配置。
        phase: 失败阶段名称。
        error: 错误信息。
    """
    if not settings.wxpusher_token:
        return

    today_str: str = date.today().strftime("%m-%d")

    message: str = (
        f"019 ETF 数据同步失败 | {today_str}\n\n"
        f"❌ 阶段 [{phase}] 执行异常\n"
        f"错误: {error}\n\n"
        f"请检查日志文件"
    )

    try:
        from wxpusher import WxPusher

        r = WxPusher.send_message(
            content=message,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("推送失败告警成功")
    except Exception as e:
        logger.warning(f"推送失败告警异常: {e}")


def push_backfill_summary(
    settings: Settings, r1: dict, r2: dict, r3: dict, elapsed: float
) -> None:
    """推送回填完成通知（--backfill 模式专用）。

    Args:
        settings: 系统配置。
        r1: sync_etf_list() 返回结果。
        r2: sync_etf_daily() 返回结果。
        r3: sync_index_daily() 返回结果。
        elapsed: 总耗时（秒）。
    """
    if not settings.wxpusher_token:
        return
    today_str: str = date.today().strftime("%m-%d")
    src = f"腾讯{r2.get('tencent_count',0)}/Sina{r2.get('sina_count',0)}"
    # total_new_records: 从 _flush 返回值获取（不在 dict 中，从 duration 推算）
    records_hint = f"{r2.get('etf_count', 0) * 550:,}"  # 粗略估算
    message: str = (
        f"019 ETF 数据回填完成 | {today_str}\n\n"
        f"✅ 全量回填成功\n"
        f"ETF 列表: {r1.get('total', 0)} 只\n"
        f"ETF 日线: {r2.get('etf_count', 0)} 只\n"
        f"指数日线: {r3.get('index_count', 0)} 个\n"
        f"📡 数据源: {src}\n"
        f"耗时: {elapsed:.0f} 秒"
    )
    try:
        from wxpusher import WxPusher
        r = WxPusher.send_message(
            content=message, token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids, content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("回填完成推送成功")
        else:
            logger.warning(f"回填推送失败: {r}")
    except Exception as e:
        logger.warning(f"回填推送异常: {e}")


def push_data_sync_summary(
    settings: Settings, r2: dict, r3: dict, elapsed: float
) -> None:
    """推送数据同步完成通知（--sync-only 模式专用）。

    Args:
        settings: 系统配置。
        r2: sync_etf_daily() 返回结果。
        r3: sync_index_daily() 返回结果。
        elapsed: 总耗时（秒）。
    """
    if not settings.wxpusher_token:
        return
    today_str: str = date.today().strftime("%m-%d")
    src = f"腾讯{r2.get('tencent_count',0)}/Sina{r2.get('sina_count',0)}"
    message: str = (
        f"019 ETF 数据同步完成 | {today_str}\n\n"
        f"ETF 日线: {r2.get('etf_count', 0)} 只\n"
        f"指数日线: {r3.get('index_count', 0)} 个\n"
        f"📡 数据源: {src}\n"
        f"耗时: {elapsed:.0f} 秒"
    )
    try:
        from wxpusher import WxPusher
        r = WxPusher.send_message(
            content=message, token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids, content_type=1,
        )
        if r.get("code") == 1000:
            logger.info("同步推送成功")
        else:
            logger.warning(f"同步推送失败: {r}")
    except Exception as e:
        logger.warning(f"同步推送异常: {e}")
