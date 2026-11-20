# F1 · LangChain

> 对应关卡：L6（框架阶段，建议在手写 ReAct 之后学）
> 面试权重：高。JD 点名。但面试官真正关心的是：**你能不能讲清它替你封装了什么，以及什么时候不该用它。**

---

## 1. 一句话结论

LangChain 的核心价值是**统一的抽象接口 + 庞大的生态集成**：换个模型、换个向量库、换个工具，代码大体不用改。
代价是**抽象层深、版本变动大、出错难调试**。所以标准答案是：**懂原理 → 用它提效 → 能判断何时不用。**

---

## 2. 概念

### 2.1 包结构（面试常被问"你用的哪个包"）

| 包 | 作用 |
| --- | --- |
| `langchain-core` | 核心抽象：消息、Runnable、Prompt 模板、工具接口 |
| `langchain-openai` 等 | 各模型厂商的官方集成包 |
| `langchain` | 高层 Agent/链的入口（1.x 里 `create_agent` 在这） |
| `langchain-community` | 社区集成（各种第三方 API、向量库、加载器） |
| `langchain-classic` | 1.x 之后存放旧版 Chain / `AgentExecutor` |
| `langgraph` | 有状态、可循环、可持久化的图编排（生产常用） |
| `langsmith` | 可观测与评测平台 |

> 版本提醒：0.1 / 0.2 / 0.3 / 1.x 之间 API 变化很大，网上大量教程是 0.1 的写法。面试时说清"我用的是 1.x，旧写法是……"，会显得你真的用过。

### 2.2 Runnable 与 LCEL

`Runnable` 是统一接口：所有组件都支持 `invoke` / `ainvoke` / `stream` / `astream` / `batch`。
LCEL（LangChain Expression Language）用 `|` 把组件串成管道：

```
prompt | model | parser
```

它的本质是"数据流管道"：前一个组件的输出是后一个的输入。好处是自动支持流式、异步、批处理和重试。

### 2.3 消息类型

| 类型 | 用途 |
| --- | --- |
| `SystemMessage` | 系统指令 |
| `HumanMessage` | 用户输入 |
| `AIMessage` | 模型回复（可能含 `tool_calls`） |
| `ToolMessage` | 工具执行结果（带 `tool_call_id` 对账） |

Agent 循环本质就是不断往消息列表里追加 `AIMessage(tool_calls)` 和 `ToolMessage`。

### 2.4 工具

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """在企业知识库中检索资料。用户询问内部制度、流程时使用。"""
    ...
```

`@tool` 会从函数签名 + docstring 自动生成 JSON Schema，docstring 就是"工具描述"，直接决定模型选不选它。

### 2.5 Agent 的两种写法

**新写法（1.x，基于 LangGraph）：**

```python
from langchain.agents import create_agent

agent = create_agent(model="openai:gpt-4o-mini", tools=[search_docs], system_prompt="...")
result = agent.invoke({"messages": [{"role": "user", "content": "报销流程是什么"}]})
```

**旧写法（0.x 教程随处可见）：**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),   # 放历史工具调用记录，必须写
])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=True)
executor.invoke({"input": "..."})
```

`agent_scratchpad` 这个占位符就是"模型的草稿纸"：把上一轮的 `tool_calls` 和工具结果回填进去，模型才能接着推理。**面试问"Agent 循环怎么实现的"，答这个就能对上。**

### 2.6 LangGraph 的定位

`create_agent` 适合"标准循环"，但真实业务经常需要：

- 分支判断（先查权限再查数据）
- 人工审批中断、隔天恢复
- 多步骤有明确状态机
- 循环 + 重试 + 回退

这些用 `StateGraph` 表达：定义 `State`、加节点（函数）、加边（普通边/条件边），编译成可执行图，配 `checkpointer` 做持久化。

---

## 3. 最小可运行代码

### 3.1 LCEL 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}。回答控制在{limit}字以内。"),
    ("human", "{question}"),
])

chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0) | StrOutputParser()

print(chain.invoke({"role": "技术顾问", "limit": 100, "question": "什么是 ReAct？"}))
```

流式只需换成 `chain.stream(...)`——这就是统一接口的价值。

### 3.2 结构化输出

```python
from pydantic import BaseModel, Field

class Ticket(BaseModel):
    title: str = Field(description="工单标题")
    priority: int = Field(ge=1, le=5)

structured_llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Ticket)
ticket = structured_llm.invoke("服务器连续 5 分钟 CPU 100%")
```

底层是用 function calling 约束输出，比自己写"请输出 JSON"稳定得多。

### 3.3 带工具的 Agent

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """在企业知识库中检索。用户问内部制度、流程、规定时使用。"""
    return "报销需在费用发生后 30 天内提交，附发票原件。"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_docs],
    system_prompt="你是行政助手。回答必须基于检索结果，没有依据就说不知道。",
)

result = agent.invoke({"messages": [{"role": "user", "content": "报销多久内要交？"}]})
print(result["messages"][-1].content)
```

### 3.4 LangGraph 最小骨架

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str

def answer_node(state: State) -> dict:
    return {"answer": f"回答：{state['question']}"}

g = StateGraph(State)
g.add_node("answer", answer_node)
g.add_edge(START, "answer")
g.add_edge("answer", END)
app = g.compile()

print(app.invoke({"question": "hi"}))
```

---

## 4. 常见坑

1. **版本地狱**：`pip install langchain` 装到哪个版本决定了 API 长什么样。项目必须锁版本（`pyproject.toml` 写死）。
2. **抽象泄漏**：链一长，报错栈几十层，定位不到是 prompt 错还是解析错。解决：分步 `invoke` 调试、开 `verbose`、接 LangSmith。
3. **AgentExecutor 不设 `max_iterations`**：默认值是有限的，但不设或设太大，都可能烧钱甚至死循环。
4. **docstring 乱写**：工具描述含糊，模型就乱选工具。写清"什么时候用、什么时候不用"。
5. **异步混用**：在 `async` 里调同步 `invoke`，会阻塞事件循环。统一用 `ainvoke/astream`。
6. **为用而用**：一个固定两步的流程，用 LCEL 两行就完了，没必要上 Agent。
7. **把 token 统计当免费**：长链每次全量传历史，成本会指数上升。
8. **忽略 memory 的语义**：`ConversationBufferMemory` 会无脑全量塞回上下文，长对话必炸。

---

## 5. 练习

### 练习 1（入门）· 改写为 LCEL

把下面这段"手写调用"改写成 LCEL 链，并保留流式能力：

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "你是翻译"}, {"role": "user", "content": text}],
)
print(resp.choices[0].message.content)
```

### 练习 2（核心）· 解释执行顺序

```python
chain = prompt | model | parser
result = chain.invoke({"question": "hi"})
```

`prompt` 收到的是什么？`model` 收到的是什么？`parser` 收到的是什么？为什么这样可以自动支持 `stream`？

### 练习 3（核心）· Agent 循环对应

把旧版 `AgentExecutor` 的执行过程，用"消息列表"描述出来（至少 4 步），并说明 `agent_scratchpad` 对应手写版里的什么东西。

### 练习 4（进阶）· 选型

以下场景分别选 `create_agent`、`StateGraph` 还是"不用框架直接写"？说明理由：
A. 单工具问答；B. 需要人工审批后才能执行退款；C. 固定三步的数据清洗管道。

---

## 6. 参考答案与自测标准

### 练习 1 答案

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是翻译"),
    ("human", "{text}"),
])
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

for chunk in chain.stream({"text": text}):
    print(chunk, end="")
```

**自测标准**：能说出 `stream` 能透传，是因为每个组件都实现了 Runnable 的流式接口。

---

### 练习 2 答案

- `prompt` 收到变量字典，输出 `ChatPromptValue`（一组消息）
- `model` 收到消息列表，输出 `AIMessage`
- `parser` 收到 `AIMessage`，输出字符串
- 自动支持 stream：管道逐块传递，模型每吐一个 token，parser 就处理一块

**自测标准**：能画出"字典 → 消息 → AIMessage → 字符串"的数据类型流转。

---

### 练习 3 答案

1. 用户输入 → `HumanMessage`
2. 模型返回带 `tool_calls` 的 `AIMessage`
3. 框架执行对应工具 → 生成 `ToolMessage`（带 `tool_call_id`）
4. 把 2、3 追加进消息列表，再次调用模型
5. 模型不再调工具，返回最终答案 → 结束

`agent_scratchpad` 对应的就是手写版里的"中间轮次记录"（tool_calls + tool 结果）。

**自测标准**：能说出 `tool_call_id` 的作用是"把结果和请求对上"。

---

### 练习 4 答案

- A：`create_agent`（或直接手写循环，工具只有一个时手写更可控）
- B：`StateGraph` + checkpointer + interrupt（需要中断、持久化、恢复）
- C：不用框架（固定流程用普通函数 + LCEL 即可，上 Agent 反而增加不确定性）

**自测标准**：判断标准是"是否需要模型自主决策"和"是否需要状态持久化"，不是"框架越新越好"。

---

## 7. 面试问法

| 面试官可能这么问 | 你要答到的点 |
| --- | --- |
| 讲讲 LangChain 的架构 | core 抽象（Runnable/消息/工具）+ 集成包 + 高层 Agent + LangGraph + LangSmith |
| LCEL 是什么？ | 管道式组合，统一 invoke/stream/batch，本质是数据流 |
| AgentExecutor 内部怎么跑？ | 消息追加循环 + tool_calls 执行 + ToolMessage 回填 + max_iterations |
| 为什么用 LangGraph | 需要分支、循环、人工中断、持久化恢复；图 = 显式状态机 |
| LangChain 有什么问题 | 抽象泄漏、版本变动、调试难、性能开销；所以生产常只用 core + 自研循环 |
| 你自己手写过 Agent 吗 | 必须答"写过"，并讲清循环 4 步（这是分水岭问题） |

**必背一句**："LangChain 我主要用它的模型接入和工具抽象，Agent 循环我自己手写过一遍，所以框架里每一步在干什么我是清楚的。"

---

## 8. 本关验收

- [ ] 能画出 LangChain 的包结构与各自职责
- [ ] 能用 LCEL 写一条带流式的链
- [ ] 能讲清 `AgentExecutor` 的消息循环，并映射到自己手写的版本
- [ ] 能说出 LangGraph 适合什么、`create_agent` 适合什么
- [ ] 能说出 3 个 LangChain 的缺点
