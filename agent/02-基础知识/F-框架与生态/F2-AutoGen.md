# F2 · AutoGen（多智能体）

> 对应关卡：L6 ~ L7
> 面试权重：中高。JD 点名。重点不是背 API，而是**说清"多智能体什么时候有用、什么时候是负担"**。

---

## 1. 一句话结论

AutoGen 的核心思想是**把多个带角色的 Agent 放进一场对话里，让它们通过发言来协作**（对话式编排）。
它擅长"角色分工明确、需要来回讨论"的任务；但它也是**最容易烧钱和跑飞**的框架——所以面试必问"你怎么控制它"。

---

## 2. 概念

### 2.1 与 LangGraph 的根本差异

| | AutoGen | LangGraph |
| --- | --- | --- |
| 编排方式 | 对话驱动（谁说、说什么） | 图驱动（节点、边、条件） |
| 控制力 | 弱，靠终止条件和角色约束 | 强，流程显式可见 |
| 适合 | 讨论、评审、头脑风暴、代码生成-执行循环 | 有明确状态机的业务流程 |
| 风险 | 跑飞、烧钱、共识幻觉 | 图设计复杂 |

一句话：**AutoGen 是"开会"，LangGraph 是"流程图"。**

### 2.2 两代 API（必须分清）

**0.2 时代（大量老教程）**

| 组件 | 作用 |
| --- | --- |
| `ConversableAgent` | 可对话的基类 |
| `AssistantAgent` | 默认由 LLM 生成回复的助手 |
| `UserProxyAgent` | 代理用户：可执行代码、可人工介入 |
| `GroupChat` + `GroupChatManager` | 多 Agent 群聊与调度 |
| `register_function` | 给 Agent 注册工具 |
| `code_execution_config` | 是否允许执行代码（有安全风险） |

**0.4+ 时代（`autogen-agentchat` / `autogen-core`）**

- 异步、事件驱动，API 完全重写
- `AssistantAgent` 需要 `model_client`
- 团队用 `RoundRobinGroupChat` / `SelectorGroupChat` / `Swarm` / `MagenticOneGroupChat`
- 终止条件是一等公民：`MaxMessageTermination`、`TextMentionTermination`、`TokenUsageTermination`、`TimeoutTermination`
- 支持 `save_state` / `load_state` 做中断恢复

> 生态提示：微软后续推出了 Microsoft Agent Framework（整合 AutoGen 与 Semantic Kernel）。面试时你可以说"AutoGen 的对话式思路沉淀到了新的 Agent Framework 里"，说明你在跟生态。

### 2.3 多智能体的三种典型拓扑

1. **两两对话**：一个执行者 + 一个评审者（写-审-改循环）
2. **轮询团队（RoundRobin）**：固定顺序发言，简单可控，成本可预测
3. **选择器团队（Selector）**：由模型或规则决定下一个发言者，灵活但不可预测

### 2.4 必须掌握的控制手段

| 问题 | 手段 |
| --- | --- |
| 无限对话 | `MaxMessageTermination(n)` |
| 互相客套不结束 | 约定退出暗号 + `TextMentionTermination("APPROVED")` |
| 成本失控 | `TokenUsageTermination`、模型分级（小模型做评审）、限制上下文 |
| 角色漂移 | system message 写死职责与禁止事项 |
| 需要人审 | 人工介入模式 / 中断后 `save_state` 恢复 |
| 可复现性差 | 固定 temperature=0、固定发言顺序、固定种子 |

---

## 3. 最小可运行代码

### 3.1 0.4+：写稿 + 审稿团队

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key="你的key",          # 生产环境从环境变量读
    )

    writer = AssistantAgent(
        "writer",
        model_client=model_client,
        system_message="你是文案。负责写初稿，每次只输出一版完整文案，不解释过程。",
    )
    reviewer = AssistantAgent(
        "reviewer",
        model_client=model_client,
        system_message=(
            "你是审稿人。只提最多 3 条具体修改意见，不要重写全文。"
            "当你认为可以定稿时，只输出 APPROVED。"
        ),
    )

    team = RoundRobinGroupChat(
        [writer, reviewer],
        termination_condition=TextMentionTermination("APPROVED") | MaxMessageTermination(8),
    )

    result = await team.run(task="写一段 50 字的产品介绍：智能简历助手，帮求职者改简历。")
    print(result.messages[-1].content)
    await model_client.close()


asyncio.run(main())
```

两个细节值得说：**终止条件是"或"关系**（约定暗号 + 硬上限兜底）；**评审者被禁止重写**，否则会陷入"你改我也改"的循环。

### 3.2 给 Agent 挂工具（0.4+）

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool


def get_order_status(order_id: str) -> str:
    """查询订单状态。"""
    return "已发货，预计明天送达"


order_tool = FunctionTool(get_order_status, description="查询订单物流状态，参数为订单号")

assistant = AssistantAgent(
    "assistant",
    model_client=model_client,
    tools=[order_tool],
    system_message="你是客服，查询类问题必须调用工具，不要凭记忆回答。",
)
```

### 3.3 0.2 写法（读懂老代码用）

```python
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "你的key"}]}

assistant = AssistantAgent("assistant", llm_config=llm_config)
user = UserProxyAgent(
    "user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},   # 生产环境必须用容器隔离
)

user.initiate_chat(assistant, message="用 Python 计算 1 到 100 的和并给出结果。", max_turns=3)
```

---

## 4. 常见坑

1. **不设终止条件**：两个 Agent 互相感谢到天荒地老，账单爆炸。
2. **让评审者重写全文**：写手和评审互相覆盖，永远收敛不了。评审只给意见，写手改稿。
3. **Agent 数量堆砌**：三个以上角色，收益急剧下降、成本急剧上升。先证明单 Agent + 工具不够用。
4. **共享同一个模型客户端不当**：创建多个 client 不复用，连接和限流都会出问题。
5. **开代码执行不隔离**：`code_execution_config` 能执行任意代码，必须容器化、断网、限资源。
6. **状态不可恢复**：长任务跑一半崩了，全部重来。要用 `save_state`。
7. **把 AutoGen 当万能**：固定流程用它反而更难调；流程明确就用图或代码。
8. **忽略上下文膨胀**：每个 Agent 都累积全量消息，几轮之后 token 爆掉，要做摘要或选择性传递。

---

## 5. 练习

### 练习 1（入门）· 终止条件

为什么只写 `TextMentionTermination("APPROVED")` 不够？再加什么？至少说出两种终止条件。

### 练习 2（核心）· 设计三角色团队

设计"需求分析-开发-测试"团队：写出三个 Agent 的 system message 要点、发言顺序、以及如何判定"任务完成"。

### 练习 3（核心）· 成本控制

你的 AutoGen 团队单次任务花了 40 万 token。列出至少 4 个优化手段，并说明优先级。

### 练习 4（进阶）· 选型判断

以下任务用 AutoGen 还是 LangGraph 还是单 Agent？说明理由：
A. 一份合同的法务+财务双视角评审；B. 用户下单后的固定风控三步流程；C. 查天气。

---

## 6. 参考答案与自测标准

### 练习 1 答案

- 只写暗号：模型可能永远不说 APPROVED，或者被注入后提前说。
- 补充：`MaxMessageTermination(8)` 防跑飞；`TokenUsageTermination(n)` 控成本；`TimeoutTermination(120)` 控时长。

**自测标准**：能说出"软条件 + 硬上限"双层兜底。

---

### 练习 2 答案

- 需求分析：`"你只负责把用户需求拆成不超过 5 条可验收的条目，输出 JSON，不要写代码。"`
- 开发：`"你只负责按需求条目实现代码，写完输出 DONE。不要修改需求。"`
- 测试：`"你只负责验证每条需求，输出通过/不通过列表。全部通过时输出 ALL_PASS。"`
- 顺序：RoundRobin 或 Selector
- 完成判定：`TextMentionTermination("ALL_PASS") | MaxMessageTermination(12)`

**自测标准**：每个角色的 system message 都包含"只做什么"和"不做什么"。

---

### 练习 3 答案（按优先级）

1. 加 `TokenUsageTermination` 和消息上限（止血）
2. 缩短历史：只传最近 N 轮 + 摘要（最大浪费源）
3. 模型分级：评审/分类用小模型，执行用大模型
4. 减少角色数，能合并就合并
5. 缓存重复的子任务结果
6. 提示词里禁止复述全文（要求"只输出修改点"）

**自测标准**：能说出"上下文重传是多智能体成本的主要来源"。

---

### 练习 4 答案

- A：AutoGen（多视角讨论，角色差异化明显）
- B：LangGraph（流程固定、需要审计和重试，不适合自由对话）
- C：单 Agent 或直接调 API（上多智能体是过度设计）

**自测标准**：判断依据是"是否需要多角色自由讨论"，而不是"框架越复杂越高级"。

---

## 7. 面试问法

| 面试官可能这么问 | 你要答到的点 |
| --- | --- |
| 为什么用多智能体？ | 角色分工带来视角差异（写-审、法务-财务）；但先说明单 Agent + 工具能解决 80% |
| AutoGen 怎么防止无限对话？ | 终止条件组合：暗号 + 最大消息数 + token 上限 + 超时 |
| AutoGen 和 LangGraph 区别 | 对话式 vs 图式；控制力、可预测性、调试难度对比 |
| 多智能体成本怎么控？ | 上下文裁剪、模型分级、角色精简、token 终止、缓存 |
| 你的 Agent 跑偏了怎么排查 | 看消息轨迹（trace）、定位是哪一轮角色越界、收紧 system message |
| 代码执行安全吗 | 必须容器隔离、断网、限资源；默认关闭 |

---

## 8. 本关验收

- [ ] 能说出 AutoGen 0.2 与 0.4+ 的差异，知道老代码长什么样
- [ ] 能写出带双重终止条件的团队代码
- [ ] 能设计三个角色并说明各自的边界
- [ ] 能说出 4 种成本控制手段
- [ ] 能一句话回答"AutoGen 与 LangGraph 怎么选"
