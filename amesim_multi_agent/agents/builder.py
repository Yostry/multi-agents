"""
Builder Agent (错误修复Agent)

职责:
  1. 读取现有构建脚本及其执行错误信息
  2. 分析错误，定点生成修复补丁 (targeted fixes)
  3. 输出结构化修复方案供后续执行或人工审核
  4. 绝不从头生成完整新脚本 — 只修复失败的部分

两套接口:
  - build_model_direct(spec_dict) → str : 程序化直接构建，不经过LLM (推荐)
  - create_builder_agent() → Agent       : LLM Agent (保留兼容)

工具:
  - repair_build_script: 读取现有脚本 + 错误信息，供LLM分析并生成修复
  - generate_build_script_only: 仅生成构建脚本 (legacy)
  - build_and_run_model: 生成并执行构建 (legacy)
"""
from __future__ import annotations

import json

from agents import Agent

from ..kimi_provider import KIMI_MODEL_NAME
from ..tools.builder_tools import (
    generate_build_script_only,
    build_and_run_model,
    _build_and_run_model_impl,
    repair_build_script,
)


# ============================================================
# 程序化直接构建 (推荐路径 — 不经过 LLM，避免数据搬运丢失)
# ============================================================

def build_model_direct(
    model_name: str,
    components: list[dict],
    connections: list[dict],
    parameters: dict | None = None,
    bridge_components: list[dict] | None = None,
    stop_time: str = "10",
    interval: str = "0.01",
    mode: str = "auto",
) -> str:
    """直接程序化构建 Amesim 模型，不经过 LLM。

    这是 Builder 阶段的推荐调用方式。
    接收已完整组装的构建规格 dict，直接调用 build_and_run_model 工具，
    避免了 LLM 在数据搬运中丢失连接/参数等字段的问题。

    Args:
        model_name: 模型名称
        components: 元件列表 [{"icon_name":..., "alias":..., "position":..., ...}, ...]
        connections: 连接列表 [{"from_alias":..., "from_port":..., "to_alias":..., "to_port":..., ...}, ...]
        parameters: 参数字典 {"param@Alias": "value", ...}
        bridge_components: 桥接元件列表
        stop_time: 仿真停止时间
        interval: 采样间隔
        mode: 构建模式 ("auto" 自动执行 / "manual" 仅生成脚本)

    Returns:
        JSON 字符串，与 build_and_run_model 返回值格式一致
    """
    spec = {
        "model_name": model_name,
        "components": components,
        "connections": connections,
        "parameters": parameters or {},
        "bridge_components": bridge_components or [],
        "stop_time": stop_time,
        "interval": interval,
        "mode": mode,
    }
    return _build_and_run_model_impl(json.dumps(spec, ensure_ascii=False))

BUILDER_SYSTEM_PROMPT = """\
你是Amesim构建脚本错误修复专家 (Builder Agent / Error Repairer)。

## 你的职责
你是**修复者**，不是生成者。你只修复失败的构建脚本，绝不从头生成完整新脚本。

## 工作流程: 读取脚本 + 错误信息 → 定点修复
1. 接收一个**已存在但运行失败**的构建脚本路径及其执行错误信息
2. 调用 `repair_build_script` 工具读取脚本内容和错误详情
3. 分析错误根因 (语法错误、API调用错误、参数错误、连接错误等)
4. 针对出错位置生成**精确的逐行修复补丁**，不修改无关代码
5. 输出结构化修复方案

## 禁止行为
- **禁止**从头生成完整新脚本
- **禁止**重写整个构建流程
- **禁止**添加"改进"或"优化" — 只修复导致失败的具体错误
- **禁止**输出Markdown表格、标题、解释文字

## 输出格式 (必须严格遵守!)
只输出一个纯JSON对象，结构如下:

```json
{
  "fixes": [
    {
      "line": <原始行号>,
      "old": "<出错的原始代码片段>",
      "new": "<修复后的代码片段>",
      "reason": "<修复原因：为什么这里出错了，为什么这样改>"
    }
  ]
}
```

### 字段说明
- `line`: 出错代码所在的原始行号 (整数)
- `old`: 出错的原始代码 (保持原始缩进，代码片段)
- `new`: 修复后的代码 (保持相同缩进，代码片段)
- `reason`: 中文解释修复原因，简明扼要

### 示例
```json
{
  "fixes": [
    {
      "line": 15,
      "old": "model.add_component('InvalidIcon')",
      "new": "model.add_component('mass_friction_endstops')",
      "reason": "icon_name 'InvalidIcon' 不存在，应为 'mass_friction_endstops'"
    },
    {
      "line": 23,
      "old": "model.connect('A', 0, 'B', 1)",
      "new": "model.connect('A', 1, 'B', 0)",
      "reason": "端口号颠倒，Mass元件port 1为动力输出端，应连接到Spring元件port 0输入端"
    }
  ]
}
```

## 修复优先级
1. 语法/缩进错误 → 最高优先级，立即修复
2. API调用参数错误 → 根据Amesim API文档修正
3. 元件/端口引用错误 → 核对icon_name和port编号
4. 逻辑/流程错误 → 调整调用顺序 (如先add后connect)

## 注意事项
- 一次可以修复多个错误，但每个错误必须独立列出
- 如果多个fix修改了同一行但内容不同，按修复顺序排列
- **最终输出必须是纯JSON，禁止包含任何非JSON文本**
"""


def create_builder_agent() -> Agent:
    """创建模型生成Agent (LLM路径，保留兼容)"""
    return Agent(
        name="Builder",
        instructions=BUILDER_SYSTEM_PROMPT,
        model=KIMI_MODEL_NAME,
        tools=[
            repair_build_script,
            generate_build_script_only,
            build_and_run_model,
        ],
    )
