"""配置管理模块：从环境变量或 .env 文件加载系统配置。

参考 004_sequoia-x 的 pydantic-settings 配置模式，
但只包含本项目需要的字段。
"""

import json
from typing import Annotated

from pydantic import BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_json_list(value: str | list[str]) -> list[str]:
    """将 JSON 数组字符串解析为 Python 列表。"""
    if isinstance(value, list):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"期望 JSON 数组，获取到 {type(parsed).__name__}")
    return [str(item) for item in parsed]


class Settings(BaseSettings):
    """系统配置，从环境变量或 .env 文件加载。

    Attributes:
        db_path: SQLite 数据库路径。
        start_date: 数据同步起始日期 "YYYY-MM-DD"。
        sync_after_hour: 同步时间门控（小时），默认 20。
        sync_after_minute: 同步时间门控（分钟），默认 0。
        wxpusher_token: WxPusher 应用的 AppToken。
        wxpusher_topic_ids: WxPusher 推送的 Topic ID 列表。
    """

    db_path: str = "data/etf_daily.db"
    start_date: str = "2024-01-01"
    sync_after_hour: int = 20
    sync_after_minute: int = 0
    wxpusher_token: str = ""
    wxpusher_topic_ids: Annotated[
        list[str],
        BeforeValidator(_parse_json_list),
    ] = Field(default=["39277"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """返回全局 Settings 单例。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
