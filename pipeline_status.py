"""
pipeline 状态追踪模块

由 pipeline.py 在每个交易日使用，记录编排器各步骤的执行状态。
状态持久化到 JSON 文件，支持：
  - 跨进程状态共享（pipeline.py 与各步骤子进程）
  - 异常中断后的状态可读
  - 微信推送汇总

"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# ── 状态文件路径 ──
_DEFAULT_STATUS_PATH = Path(__file__).parent / "pipeline_status.json"


class PipelineStatus:
    """
    管线状态 — 每个交易日一条记录。

    用法：
        ps = PipelineStatus()
        ps.reset()
        ps.add_step("sync", "数据同步")
        ps.start_step("sync")
        # ... 执行同步 ...
        ps.complete_step("sync", success=True)
        ps.finish("completed")
    """

    def __init__(self, status_path: str | Path = _DEFAULT_STATUS_PATH):
        self.status_path = Path(status_path)
        self._data: dict[str, Any] = {}

    # ── 属性 ──

    @property
    def today(self) -> str:
        return date.today().isoformat()

    # ── 读写 ──

    def load(self) -> dict[str, Any]:
        """从磁盘加载当日状态，若文件不存在或非当日则返回空 dict。"""
        if not self.status_path.exists():
            self._data = {}
            return self._data
        try:
            with open(self.status_path) as f:
                raw = json.load(f)
            if raw.get("date") == self.today:
                self._data = raw
            else:
                self._data = {}
        except (json.JSONDecodeError, OSError):
            self._data = {}
        return self._data

    def save(self) -> None:
        """原子写入 JSON（临时文件 -> os.replace）。"""
        fd, tmp = tempfile.mkstemp(
            suffix=".json",
            prefix="pipeline_status_",
            dir=self.status_path.parent,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.status_path)
        except Exception:
            os.unlink(tmp)
            raise

    # ── 生命周期 ──

    def reset(self) -> None:
        """创建当日空白记录。"""
        self._data = {
            "date": self.today,
            "pipeline_status": "running",
            "started_at": datetime.now().strftime("%H:%M:%S"),
            "finished_at": None,
            "current_step": None,
            "steps": {},
        }
        self.save()

    def add_step(self, step_id: str, name: str) -> None:
        """注册一个步骤。"""
        steps = self._data.setdefault("steps", {})
        if step_id not in steps:
            steps[step_id] = {
                "name": name,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration": None,   # 秒
                "error": None,
                "detail": {},
            }
        self.save()

    def start_step(self, step_id: str) -> None:
        """标记步骤开始。"""
        self._data["current_step"] = step_id
        step = self._data.setdefault("steps", {}).get(step_id)
        if step:
            step["status"] = "running"
            step["started_at"] = datetime.now().strftime("%H:%M:%S")
        self.save()

    def complete_step(
        self,
        step_id: str,
        success: bool = True,
        detail: dict | None = None,
        error: str | None = None,
    ) -> None:
        """标记步骤完成（成功或失败）。"""
        step = self._data.setdefault("steps", {}).get(step_id)
        if step:
            finished = datetime.now()
            step["status"] = "completed" if success else "failed"
            step["finished_at"] = finished.strftime("%H:%M:%S")
            if step.get("started_at"):
                try:
                    start = datetime.strptime(step["started_at"], "%H:%M:%S")
                    step["duration"] = round(
                        (finished - start.replace(year=finished.year,
                                                   month=finished.month,
                                                   day=finished.day)).total_seconds()
                    )
                except ValueError:
                    pass
            if detail:
                step["detail"] = detail
            if error:
                step["error"] = error
        self._data["current_step"] = None
        self.save()

    def finish(self, status: str = "completed") -> None:
        """标记整个管线完成。

        Args:
            status: "completed" | "failed" | "skipped"
        """
        self._data["pipeline_status"] = status
        self._data["finished_at"] = datetime.now().strftime("%H:%M:%S")
        self.save()

    def needs_rerun(self) -> bool:
        """检测当日是否需要重新运行（上次异常中断）。"""
        raw = self.load()
        if not raw:
            return True
        ps = raw.get("pipeline_status", "")
        return ps in ("running", "failed")

    def to_dict(self) -> dict:
        return dict(self._data)


# ════════════════════════════════════════════════════════════
#  推送汇总（WxPusher）
# ════════════════════════════════════════════════════════════

def push_pipeline_summary(
    status_data: dict[str, Any],
    notify_func,
) -> None:
    """推送 pipeline 汇总到微信。

    Args:
        status_data: PipelineStatus 的 _data dict。
        notify_func: 推送函数，签名 notify_func(title, content, content_type=1)。
    """
    steps = status_data.get("steps", {})
    lines = [f"📊 ETF 模拟盘管线 | {status_data.get('date', '')}"]

    for sid, s in steps.items():
        emoji = {"completed": "✅", "failed": "❌", "running": "🔄",
                 "pending": "⏳", "skipped": "⏭️"}.get(s.get("status", ""), "❓")
        dur = s.get("duration")
        dur_str = f"({dur // 60}分{dur % 60}秒)" if dur else ""
        err = s.get("error", "")
        err_str = f" ⚠️{err}" if err else ""
        lines.append(f"\n{emoji} {s['name']}{dur_str}{err_str}")

    lines.append(f"\n🏁 状态: {status_data.get('pipeline_status', 'unknown')}")
    if status_data.get("finished_at"):
        lines.append(f"⏱ 完成: {status_data['finished_at']}")

    notify_func(
        title=f"ETF 模拟盘管线 | {status_data.get('date', '')}",
        content="\n".join(lines),
        content_type=1,
    )
