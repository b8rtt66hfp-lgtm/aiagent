# C1 · ReAct 与 Agent 循环

> 对应关卡：L4（项目一的理论部分）
> 面试权重：最高。这是 Agent 岗位的"必答题"，答不上来基本就结束了。

---

## 1. 一句话结论

**Agent = LLM + 工具 + 循环 + 状态。**
ReAct 的本质是：让模型"想一步 → 做一步 → 看结果 → 再想"，直到它能给出最终答案。**没有循环和工具回填的"Agent"，只是套了壳的聊天。**

---

## 2. 概念

### 2.1 为什么需要循环

单次模型调用只能回答"它已经知道的"。而真实任务需要：

- 查实时数据（天气、库存、订单）
- 执行动作（发邮件、建工单、改数据库）
- 多步推理（先查 A，根据 A 的结果决定查 B）

模型无法在一次调用里完成"调用 → 拿到结果 → 继续推理"，所以必须由**你的代码**负责循环。

### 2.2 ReAct 的原始形态

论文里的 ReAct 是纯文本格式：

```
Thought: 我需要查北京天气
Action: get_weather
Action Input: {"city": "北京"}
Observation: 北京 晴 25℃
Thought: 已有答案
Final Answer: 北京今天晴，25 度
```

你需要解析文本里的 Action。**缺点是格式不稳定、解析容易失败。**

### 2.3 现代形态：工具调用（Tool Calling）

现在主流模型原生支持结构化工具调用，返回的是：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {"id": "call_abc", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\":\"北京\"}"}}
  ]
}
```

你不再需要正则解析，循环变成：

```
1. 把 messages 发给模型（带 tools 定义）
2. 模型返回 tool_calls？
   - 是 → 逐个执行 → 把结果作为 role=tool 的消息追加 → 回到 1
   - 否 → content 就是最终答案 → 结束
```

三个关键字段：

| 字段 | 作用 |
| --- | --- |
| `tool_calls[].id` | 调用凭证，工具结果必须带着同一个 id 回去 |
| `function.name` | 选哪个工具 |
| `function.arguments` | JSON 字符串，需要解析 + 校验 |

### 2.4 循环的四种终止条件

1. 模型不再返回 `tool_calls`（正常结束）
2. 达到 `max_steps`（防死循环兜底）
3. 总超时 / 总 token 超预算（控成本）
4. 业务终止工具被调用（如 `submit_answer`）

**生产环境必须同时有 1 + 2**，其他按需。

### 2.5 Agent 还是 Workflow

| | Workflow（固定流程） | Agent（自主循环） |
| --- | --- | --- |
| 路径 | 预先写死 | 模型决定 |
| 可预测性 | 高 | 低 |
| 调试难度 | 低 | 高 |
| 适合 | 步骤明确、合规要求高 | 步骤不确定、需要探索 |

**面试加分点**：主动说"大部分业务场景应该先考虑 Workflow，Agent 只用在路径真的不确定的地方"。这说明你有工程判断力，而不是拿锤子找钉子。

### 2.6 两个常见变体

- **Plan-and-Execute**：先让模型产出完整计划，再逐步执行。优点是全局视角、可审计；缺点是无法应对执行中的意外。
- **Reflection**：执行后让模型自评并修改。优点是质量提升；缺点是成本翻倍、可能越改越差。

先掌握 ReAct，其他都建立在它之上。

---

## 3. 最小可运行代码（完整可跑）

```python
import json
from openai import OpenAI

client = OpenAI(api_key="你的key", base_url="https://api.deepseek.com")


def get_weather(city: str) -> str:
    return f"{city}：晴，25℃"


TOOLS = {
    "get_weather": {
        "schema": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气。用户问天气时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京"}
                },
                "required": ["city"],
            },
        },
        "func": get_weather,
    }
}


def run_agent(question: str, max_steps: int = 6) -> str:
    messages = [
        {"role": "system", "content": "需要外部信息时先调用工具，不要凭记忆编造。拿到结果后直接回答用户。"},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=[t["schema"] for t in TOOLS.values()],
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # 终止条件 1：模型不再调工具
        if not msg.tool_calls:
            return msg.content

        # 执行工具，把结果回填
        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                result = TOOLS[name]["func"](**args)
            except KeyError:
                result = f"没有名为 {name} 的工具，请从可用工具中选择。"
            except Exception as e:
                result = f"工具执行失败：{e}。请调整参数，不要重复相同调用。"

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,        # 必须对应，否则接口报错
                "content": str(result)[:2000],  # 裁剪，防上下文爆炸
            })

    # 终止条件 2：步数兜底
    return "抱歉，任务步骤过多，已停止。请换一种更明确的问法。"
```

跑通的标志：问"北京天气怎么样"，你能在日志里看到 `assistant(tool_calls)` → `tool` → `assistant(content)` 三条消息。

---

## 4. 常见坑

1. **不设 `max_steps`**：模型可能反复调同一个工具，账单和延迟一起爆炸。
2. **`tool_call_id` 对不上**：接口直接报错；一次返回多个 `tool_calls` 时必须逐个回填。
3. **工具结果不裁剪**：一个接口返回 50KB JSON，两轮就撑爆上下文。
4. **异常直接抛给用户**：应该把错误转成文字回填给模型，让它自己纠错。
5. **把 `tool_calls` 当最终答案**：`content` 为空不代表失败，要继续循环。
6. **历史消息不完整**：只追加 `tool` 消息却漏了带 `tool_calls` 的 `assistant` 消息，接口会校验失败。
7. **温度太高**：工具选择是分类任务，`temperature=0` 更稳定。
8. **不记录中间过程**：出问题无法定位是哪一步错了，必须打 trace。

---

## 5. 练习

### 练习 1（入门）· 写伪代码

不看本文，写出 ReAct 循环的 5 步伪代码，并标出两个终止条件。

### 练习 2（核心）· 加一个工具

给上面的代码增加 `search_order(order_id)`，要求：参数缺失时返回什么、订单不存在时返回什么、返回结果只保留 3 个字段。

### 练习 3（核心）· 死循环检测

模型连续 3 次调用同一个工具、同样的参数。你的代码应该如何检测并处理？

### 练习 4（进阶）· 并行工具调用

模型一次返回了两个 `tool_calls`。你会串行执行还是并行？说明理由，并说出并行时要注意什么。

### 练习 5（进阶）· 判断选型

"用户上传发票 → OCR → 校验金额 → 入账"这个需求，用 Agent 还是 Workflow？说明理由。

---

## 6. 参考答案与自测标准

### 练习 1 答案

```
1. 组装 messages（system + 历史 + 用户问题）
2. 带 tools 调用模型
3. 若返回 tool_calls：逐个执行，追加 assistant 消息 + tool 结果消息，回到 2
4. 若无 tool_calls：返回 content，结束
5. 若达到 max_steps：返回兜底话术，结束
```

**自测标准**：能说出"循环的燃料是工具结果，出口是模型不再调工具"。

---

### 练习 2 答案

```python
def search_order(order_id: str) -> str:
    if not order_id:
        return "缺少订单号，请向用户确认后再查询。"
    data = {"ORD-001": {"status": "已发货", "amount": 199, "eta": "明天"}}
    order = data.get(order_id)
    if not order:
        return "订单不存在，请让用户核对订单号。"
    return f"状态{order['status']}，金额{order['amount']}元，预计{order['eta']}送达"
```

要点：错误返回是"给模型下一步动作"，不是异常；结果裁剪到模型需要的字段。

---

### 练习 3 答案

维护一个 `(tool_name, arguments)` 的集合或最近调用签名：

```python
seen = set()
key = (name, json.dumps(args, sort_keys=True))
if key in seen:
    result = "该调用已执行过且结果相同，请勿重复调用，改用其他方法或直接回答。"
else:
    seen.add(key)
    result = execute(...)
```

连续 3 次相同调用可以直接终止并返回兜底答案（成本优先）。

**自测标准**：能说出"检测的是签名而不是次数"。

---

### 练习 4 答案

- 相互独立 → 并行（`asyncio.gather`），省延迟
- 有依赖（第二个参数来自第一个结果）→ 串行

并行注意：共享资源、限流、某个失败时如何回填（每个 `tool_call_id` 都要有对应结果，失败也要回一条错误消息）。

**自测标准**：能说出"每个 tool_call 都必须有响应，否则下一轮接口报错"。

---

### 练习 5 答案

Workflow。步骤固定、需要审计、失败要可重试，不需要模型自主决策。Agent 只适合"路径不确定"的场景。

**自测标准**：能主动指出"这里用 Agent 是过度设计"。

---

## 7. 面试问法

| 面试官可能这么问 | 你要答到的点 |
| --- | --- |
| 讲一下 ReAct | 想-做-看-再想；现代实现用 tool_calls 而非文本解析 |
| 手写 Agent 循环（白板） | 消息追加 4 步 + 两个终止条件 + id 对账 |
| 怎么防止死循环 | max_steps + 重复调用检测 + 总超时/token 预算 |
| 工具执行失败怎么办 | 错误转文字回填给模型，让它纠错；不要直接抛 |
| 什么时候不用 Agent | 流程固定、强合规、需要可预测成本时用 Workflow |
| 多轮工具调用上下文怎么管 | 工具结果裁剪、历史摘要、只保留必要轮次 |

---

## 8. 本关验收

- [ ] 能不看代码，白板写出完整 ReAct 循环
- [ ] 能说出 `tool_call_id` 的作用
- [ ] 能设计至少 3 种终止条件
- [ ] 能给已有循环加上重复调用检测
- [ ] 能讲清 Agent 与 Workflow 的边界
