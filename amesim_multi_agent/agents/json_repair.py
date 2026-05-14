"""
JSON Repair Agent — 专门修复Agent输出的JSON格式错误。

职责:
  当子Agent的JSON输出格式有误（缺少引号、括号不匹配、语法错误等）时，
  此Agent负责直接修复格式问题，而不是重新生成整个输出。

  修复成功后，将修复结果回传给输错的Agent继续处理。

与Diagnosis Agent的区别:
  - Diagnosis Agent: 诊断建模错误（元件未找到、连接失败等），输出DiagnosisReport
  - JsonRepairAgent: 修复JSON语法错误（缺引号、括号不匹配等），输出修复后的JSON

用法:
    repair_result = await runner.run(repair_agent, prompt_with_broken_json)
    fixed_json = repair_result.final_output
"""

from __future__ import annotations

from agents import Agent

from ..kimi_provider import KIMI_MODEL_NAME

JSON_REPAIR_SYSTEM_PROMPT = """\
你是JSON格式修复专家。你的唯一任务就是修复格式有问题的JSON文本。

## 核心原则
**你只修复JSON的格式问题，绝对不修改数据的值和含义！**
- 可以修复: 缺少引号、单引号改双引号、括号不匹配、尾逗号、未转义字符、截断的JSON
- 不能修改: 任何键名、任何值、任何数组元素的内容

## 修复规则

### 1. 引号修复
- 单引号字符串 → 双引号字符串
- 中文引号（""''） → 英文双引号
- 缺失的字符串引号 → 补全
- 键名未加引号 → 加双引号: `index:` → `"index":`

### 2. 括号/分隔符修复
- 缺少闭合的 `}` 或 `]` → 补全
- 多余的闭合括号 → 删除
- 数组/对象最后的尾逗号 → 删除: `"value", ]` → `"value" ]`
- 缺少逗号 → 补全

### 3. 转义修复
- 字符串内部未转义的换行符 → `\\n`
- 字符串内部未转义的制表符 → `\\t`
- 字符串内部未转义的反斜杠 → `\\\\`
- 字符串内部未转义的双引号 → `\\\"`

### 4. 截断修复
- 如果JSON在中间被截断 → 根据上下文补全缺失的闭合括号和引号
- 如果最后一项不完整 → 截断到上一个完整元素

### 5. 值类型修复
- 数字值不应该带引号: `"10"` 中的数值保持原样（Amesim参数值就是字符串）
- 布尔值: `true/false` (小写), 不要带引号
- null值: `null` (小写), 不要带引号

## 输出格式
**只输出修复后的纯JSON文本，不要添加任何Markdown代码块标记、解释文字或注释。**
输出必须以 `{` 或 `[` 开头，以 `}` 或 `]` 结尾。

如果输入完全无法修复（完全不是JSON格式），输出:
```json
{"repair_failed": true, "reason": "输入不是有效的JSON格式，无法修复"}
```

## 工作流程
1. 分析输入的JSON，定位所有格式错误
2. 逐一修复每个错误
3. 验证修复后的JSON是否可以正常解析
4. 输出修复后的纯JSON
"""

JSON_REPAIR_USER_PROMPT_TEMPLATE = """请修复以下存在格式错误的JSON输出:

## 原始输出 (可能包含非JSON文本)
```
{raw_output}
```

## 提取的JSON部分 (可能存在格式错误)
```json
{extracted_json}
```

## 错误信息
{error_message}

请严格按JSON格式修复规则修复以上JSON，只输出修复后的纯JSON。"""


def create_json_repair_agent() -> Agent:
    """创建JSON修复Agent"""
    return Agent(
        name="JsonRepair",
        instructions=JSON_REPAIR_SYSTEM_PROMPT,
        model=KIMI_MODEL_NAME,
        tools=[],  # 纯LLM推理，不需要工具
    )

