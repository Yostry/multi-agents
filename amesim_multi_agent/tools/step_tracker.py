"""
步骤跟踪器 — 为每个子Agent提供实时执行步骤的可见性。

用法:
    from .tools.step_tracker import get_tracker, reset_tracker

    tracker = get_tracker()
    tracker.start("selector", "search", "正在搜索元件", "关键词: pump")
    # ... do work ...
    tracker.end("selector", "search", "ok", "找到 3 个候选")

输出格式:
    [Selector] ▶ 正在搜索元件: 关键词: pump
    [Selector]   OK (2.3s): 找到 3 个候选
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StepInfo:
    """单个步骤的信息"""
    step_id: str
    agent_name: str
    description: str       # 简短描述如 "正在搜索元件"
    status: str            # "running" | "ok" | "fail" | "warn"
    detail: str = ""       # 补充细节
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class StepTracker:
    """轻量级步骤跟踪器，实时输出子Agent的执行进度。

    单线程 asyncio 环境使用，通过 start/end 配对跟踪每个步骤。
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._steps: dict[str, StepInfo] = {}  # key: "agent_name:step_id"

    def start(
        self,
        agent_name: str,
        step_id: str,
        description: str,
        detail: str = "",
    ) -> StepInfo:
        """开始一个步骤，立即打印到终端。"""
        step = StepInfo(
            step_id=step_id,
            agent_name=agent_name,
            description=description,
            detail=detail,
            status="running",
            start_time=time.time(),
        )
        key = f"{agent_name}:{step_id}"
        self._steps[key] = step

        if self._enabled:
            agent_tag = f"[{agent_name}]"
            detail_str = f": {detail}" if detail else ""
            print(f"  {agent_tag} ▶ {description}{detail_str}")

        return step

    def end(
        self,
        agent_name: str,
        step_id: str,
        status: str = "ok",
        detail: str = "",
    ) -> StepInfo | None:
        """结束一个步骤，打印结果。"""
        key = f"{agent_name}:{step_id}"
        step = self._steps.get(key)
        if step is None:
            return None

        step.status = status
        step.end_time = time.time()
        if detail:
            step.detail = detail

        if self._enabled:
            agent_tag = f"[{agent_name}]"
            elapsed = step.elapsed
            icon = {"ok": "OK", "fail": "✗", "warn": "⚠", "running": "…"}.get(status, "?")
            detail_str = f": {step.detail}" if step.detail else ""
            print(f"  {agent_tag}   {icon} ({elapsed:.1f}s){detail_str}")

        return step

    def warn(self, agent_name: str, step_id: str, detail: str = ""):
        """标记步骤为警告"""
        self.end(agent_name, step_id, status="warn", detail=detail)

    def fail(self, agent_name: str, step_id: str, detail: str = ""):
        """标记步骤为失败"""
        self.end(agent_name, step_id, status="fail", detail=detail)

    def reset(self):
        """重置跟踪器状态"""
        self._steps.clear()


# ── 模块级单例 ────────────────────────────────────────────

_tracker: StepTracker | None = None


def get_tracker() -> StepTracker:
    """获取模块级 StepTracker 实例"""
    global _tracker
    if _tracker is None:
        _tracker = StepTracker()
    return _tracker


def reset_tracker():
    """重置模块级 StepTracker"""
    global _tracker
    if _tracker is not None:
        _tracker.reset()
    _tracker = StepTracker()
