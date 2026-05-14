"""
思维链缓冲区 (ThinkBuffer)

为循环思考总管 Agent 提供持久化的推理历史记录。
每一轮思考都记录 thought/action/observation 三元组,
支持序列化为 LLM 上下文和人类可读的调试摘要。

设计原则:
  - 轻量: 纯 Python 列表+字典, 无外部依赖
  - 截断: 支持按 token 预算自动截断旧记录
  - 可查: 支持按阶段、状态、关键字的筛选查询
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ThinkStep:
    """单次思考步骤"""
    step_id: int                           # 步骤序号 (从 1 开始)
    phase: str                             # 当前阶段: plan | execute | reflect
    thought: str                           # LLM 的推理内容 (自然语言)
    action: str                            # 执行的动作 (工具名或 Agent 名)
    action_input: str = ""                 # 动作的输入参数 (截断)
    observation: str = ""                  # 动作的结果 (截断)
    success: bool = True                   # 是否成功
    sub_phase: str = ""                    # 子阶段: core_select | port_expand | connect | param
    timestamp: float = field(default_factory=time.time)

    def to_summary(self) -> str:
        status = "✓" if self.success else "✗"
        obs_preview = self.observation[:80].replace("\n", " ")
        return (
            f"[Step {self.step_id}] {status} {self.phase}/{self.sub_phase} | "
            f"Thought: {self.thought[:60]}... | "
            f"Action: {self.action}({self.action_input[:40]}) | "
            f"Obs: {obs_preview}"
        )

    def to_llm_context(self) -> str:
        """格式化为 LLM 的历史上下文"""
        status = "成功" if self.success else "失败"
        lines = [
            f"## 步骤 {self.step_id} [{self.phase}/{self.sub_phase}] — {status}",
            f"**推理**: {self.thought}",
            f"**动作**: {self.action}({self.action_input})",
            f"**结果**: {self.observation}",
            "",
        ]
        return "\n".join(lines)


# ============================================================
# 思维链缓冲区
# ============================================================

class ThinkBuffer:
    """循环思考的完整历史记录。

    用法:
        buf = ThinkBuffer(max_steps=30)
        buf.record(
            phase="plan", thought="需要先识别核心元件",
            action="search_component", action_input="储罐",
            observation="找到 TPFHECH000, 3 个端口"
        )
        # 获取最近 N 步作为 LLM 上下文
        context = buf.recent_context(n=10)
    """

    def __init__(self, max_steps: int = 50, max_context_chars: int = 8000):
        self._steps: list[ThinkStep] = []
        self._counter = 0
        self.max_steps = max_steps
        self.max_context_chars = max_context_chars

    # ---- 写入 ----

    def record(
        self,
        *,
        phase: str,
        thought: str,
        action: str,
        action_input: str = "",
        observation: str = "",
        success: bool = True,
        sub_phase: str = "",
    ) -> ThinkStep:
        """记录一次思考步骤。返回创建的 ThinkStep。"""
        self._counter += 1
        step = ThinkStep(
            step_id=self._counter,
            phase=phase,
            thought=thought,
            action=action,
            action_input=action_input[:500],
            observation=observation[:2000],
            success=success,
            sub_phase=sub_phase,
        )
        self._steps.append(step)
        self._trim()
        return step

    def _trim(self):
        """超过最大步数时删除最旧的记录。"""
        while len(self._steps) > self.max_steps:
            self._steps.pop(0)

    # ---- 读取 ----

    @property
    def steps(self) -> list[ThinkStep]:
        return list(self._steps)

    @property
    def last(self) -> ThinkStep | None:
        return self._steps[-1] if self._steps else None

    @property
    def total(self) -> int:
        return len(self._steps)

    def recent(self, n: int = 10) -> list[ThinkStep]:
        """返回最近 n 步。"""
        return self._steps[-n:]

    def by_phase(self, phase: str) -> list[ThinkStep]:
        """按阶段筛选。"""
        return [s for s in self._steps if s.phase == phase]

    def failures(self) -> list[ThinkStep]:
        """返回所有失败的步骤。"""
        return [s for s in self._steps if not s.success]

    def consecutive_failures(self) -> int:
        """返回最近的连续失败次数。"""
        count = 0
        for s in reversed(self._steps):
            if not s.success:
                count += 1
            else:
                break
        return count

    # ---- LLM 上下文生成 ----

    def recent_context(self, n: int = 10) -> str:
        """将最近 n 步格式化为 LLM 上下文文本。"""
        steps = self._steps[-n:]
        parts = ["# 历史推理记录\n"]
        for s in steps:
            parts.append(s.to_llm_context())
        text = "\n".join(parts)

        # 按字符数截断
        if len(text) > self.max_context_chars:
            text = text[-self.max_context_chars:]
            text = "...(更早的记录已截断)\n\n" + text[text.find("## 步骤"):]

        return text

    def compact_context(self, n: int = 5) -> str:
        """生成紧凑版上下文 (每步仅一行), 适合在 prompt 中嵌入。"""
        steps = self._steps[-n:]
        lines = ["## 最近操作记录"]
        for s in steps:
            status = "✓" if s.success else "✗"
            obs = s.observation[:100].replace("\n", " ")
            lines.append(
                f"  {status} [{s.phase}/{s.sub_phase}] {s.action}: {obs}"
            )
        return "\n".join(lines)

    # ---- 调试 ----

    def print_history(self, n: int = 20):
        """打印最近 n 步的摘要到终端。"""
        steps = self._steps[-n:]
        print(f"\n{'='*70}")
        print(f"  思维链历史 (最近 {len(steps)} 步, 共 {self.total} 步)")
        print(f"{'='*70}")
        for s in steps:
            print(s.to_summary())
        print(f"{'='*70}\n")

    def summary(self) -> dict:
        """返回统计摘要。"""
        phases = {}
        for s in self._steps:
            phases[s.phase] = phases.get(s.phase, 0) + 1
        return {
            "total_steps": self.total,
            "phases": phases,
            "failures": len(self.failures()),
            "consecutive_failures": self.consecutive_failures(),
            "last_phase": self.last.phase if self.last else None,
        }

    def clear(self):
        """清空历史。"""
        self._steps.clear()
        self._counter = 0
