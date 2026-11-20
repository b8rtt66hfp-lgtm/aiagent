# E1 · Agent 服务架构与可观测

> 对应关卡：L6（项目三的理论部分）
> 面试权重：高。决定你是"做过 Demo"还是"上过线"。

---

## 1. 一句话结论

**Demo 和生产的差距，不在功能，而在失败处理。**
面试官问"上线要注意什么"，你要能一口气说出：状态存哪、超时怎么设、失败怎么降级、钱花在哪、出问题怎么查。

---

## 2. 概念

### 2.1 服务架构

```
Client ──HTTP/SSE──> API 服务（无状态，可水平扩展）
                        │
                        ├── 会话存储（Redis / Postgres）
                        ├── 任务队列（长任务异步执行）
                        ├── LLM 网关（限流、重试、降级、计费）
                        ├── 工具服务（外部 API / MCP）
                        └── 观测（日志 / trace / 指标）
```

三个关键决策：

1. **服务无状态**：会话数据放外部存储，任意实例都能接管请求。
2. **长任务异步化**：超过 10 秒的任务进队列，立即返回 `task_id`，客户端轮询或订阅 SSE。
3. **LLM 调用统一走网关**：所有重试、限流、降级、成本统计集中在一层，不散落在业务代码里。

### 2.2 可观测三件套

| 类型 | 记什么 | 用途 |
| --- | --- | --- |
| 日志 | 结构化 JSON + `request_id` + `session_id` | 定位"哪个请求" |
| Trace | 一次 run 拆成多个 span：`llm.call`、`tool.call` | 定位"哪一步" |
| 指标 | 延迟 P50/P95、token 消耗、工具成功率、步数分布、失败率 | 发现趋势和容量问题 |

**Agent 的 trace 必须记录**：每轮的 prompt 摘要、工具名与参数摘要、返回摘要、耗时、token 数。
否则用户说"答错了"，你只能靠猜。

### 2.3 稳定性的四道防线

1. **超时**：每次 LLM/工具调用有超时，整个 run 有总预算
2. **重试**：只对超时、5xx、429 重试，指数退避 + 抖动
3. **熔断**：连续失败达到阈值，短时间内直接快速失败，保护下游
4. **降级**：主模型不可用切备用模型；工具不可用返回"暂不可用"，而不是让循环卡死

### 2.4 限流与成本

- 用户级限流（每人每分钟 N 次）
- 全局并发上限（保护模型配额）
- 单次 run 的 token 预算（超了强制停止）
- 模型分级：分类/摘要用小模型，复杂推理用大模型
- 缓存：相同问题/相同文档的响应可缓存

**面试要能报数字**："一次典型任务约 3 轮、每轮 2000 input token，单次成本约 X 分钱"——有量化的候选人立刻高一个档次。

### 2.5 安全

| 风险 | 措施 |
| --- | --- |
| 密钥泄漏 | 环境变量/密钥服务，不进代码、日志、prompt |
| 提示词注入 | 数据与指令分离、输出校验、工具最小权限 |
| 工具越权 | 身份由服务端注入，工具内部鉴权 |
| 敏感数据 | 日志脱敏，PII 不入 trace |
| 滥用 | 限流、配额、审计日志 |

### 2.6 评测回归

把黄金集放进 CI：每次改 prompt/模型/检索，自动跑一遍，对比准确率、成本、P95 延迟。
没有这个，你的"优化"都是玄学。

---

## 3. 最小可运行代码

### 3.1 一个轻量 trace 装饰器

```python
import time
import uuid
import logging
from contextvars import ContextVar

logger = logging.getLogger("agent")
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def traced(step: str):
    def deco(fn):
        async def wrapper(*args, **kwargs):
            rid = request_id_var.get()
            start = time.perf_counter()
            try:
                result = await fn(*args, **kwargs)
                logger.info({
                    "event": "span_end", "request_id": rid, "step": step,
                    "ok": True, "ms": round((time.perf_counter() - start) * 1000, 1),
                })
                return result
            except Exception as e:
                logger.error({
                    "event": "span_end", "request_id": rid, "step": step,
                    "ok": False, "ms": round((time.perf_counter() - start) * 1000, 1),
                    "error": type(e).__name__,
                })
                raise
        return wrapper
    return deco


@traced("llm.call")
async def call_llm(messages: list[dict]) -> str:
    ...
```

### 3.2 FastAPI + SSE 骨架

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uuid

app = FastAPI()


@app.post("/chat")
async def chat(payload: dict, request: Request):
    request_id_var.set(request.headers.get("X-Request-Id", str(uuid.uuid4())))
    session_id = payload["session_id"]

    async def event_stream():
        try:
            async for token in run_agent_stream(payload["message"], session_id):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("stream failed")
            yield 'data: {"error":"服务暂时不可用，请重试"}\n\n'

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

要点：流开始后无法再改 HTTP 状态码，**错误必须作为事件推给客户端**。

### 3.3 指标记录（伪代码结构）

```python
metrics = {
    "run_total": 0,
    "run_failed": 0,
    "tokens_total": 0,
    "latency_ms": [],
    "tool_calls": {},
}
```

生产环境接 Prometheus / OpenTelemetry；自建项目里至少把这几项写进日志并定期统计。

---

## 4. 常见坑

1. **会话存在内存里**：重启丢失，多实例串话。
2. **长任务同步等待**：网关 60 秒超时，任务被直接掐断。
3. **只打普通日志**：出问题无法按 `request_id` 串起一次完整 run。
4. **trace 记录完整 prompt**：包含用户隐私和密钥，既贵又违规。
5. **重试没有幂等**：重复建单、重复发信。
6. **无限流**：一个用户刷接口就能打爆配额。
7. **流式接口里抛异常**：连接直接断，前端只看到网络错误。
8. **没有成本监控**：月底账单出来才发现超支。
9. **上线前不跑评测**：模型或 prompt 一改，效果回退无人发现。

---

## 5. 练习

### 5.1 练习 1（入门）· 架构选择

一个"上传 100 页 PDF 并生成摘要"的需求，用同步接口还是异步任务？说明理由和完整交互流程。

### 5.2 练习 2（核心）· 排查慢请求

用户反馈"有时要等 40 秒"。你有日志和 trace，列出至少 5 个可能原因，并说明每个原因在 trace 里长什么样。

### 5.3 练习 3（核心）· 限流设计

为一个企业内部 Agent 设计限流方案，要求兼顾"个人滥用"和"整体配额"。

### 5.4 练习 4（进阶）· 降级策略

主模型超时率突然从 1% 涨到 30%。写出你的处理顺序。

### 5.5 练习 5（进阶）· 成本优化

单次任务成本 0.3 元，日活 2000 人、人均 5 次。列出 3 个优化手段并估算效果。

---

## 6. 参考答案与自测标准

### 6.1 练习 1 答案

异步任务。
流程：上传 → 返回 `task_id`（202）→ 后台队列处理（解析、切分、逐段摘要）→ 客户端轮询 `/tasks/{id}` 或订阅 SSE → 完成返回结果链接。
理由：耗时远超 HTTP 超时，且失败需要重试。

**自测标准**：能说出"202 + 任务查询"的标准模式。

### 6.2 练习 2 答案

1. 模型侧慢：`llm.call` span 耗时长
2. 工具慢：`tool.call` span 异常长，或某个工具串行执行
3. 循环步数多：trace 里 span 数量比正常多
4. 排队/限流：请求在网关等待，`queue.wait` 时间长
5. 上下文过大：token 数异常高
6. 下游重试：同一工具出现多次重试 span
7. 数据库慢查询：会话读写 span 长

**自测标准**：能把"慢"定位到具体 span，而不是只说"模型慢"。

### 6.3 练习 3 答案

- 用户级：每分钟 5 次、每日 100 次
- 全局：并发 50（Semaphore）
- 全局日配额：按部门分配
- 超限行为：429 + `Retry-After`
- 关键操作（如建单）单独限流，避免被普通查询挤占

**自测标准**：能区分"保护用户公平"和"保护系统容量"。

### 6.4 练习 4 答案

1. 确认监控数据，排除自己的误报
2. 自动降级到备用模型/备用区域
3. 降低单次 run 的最大步数和输出长度
4. 开启熔断，快速失败并给用户友好提示
5. 排查是自己的流量突增还是上游故障
6. 恢复后逐步放量，观察指标

**自测标准**：能说出"先止血，再定位，最后恢复"。

### 6.5 练习 5 答案

日成本 = 0.3 × 2000 × 5 = 3000 元。

1. 上下文压缩：历史不全量传，预计降 30%~50% input token
2. 模型分级：简单请求走小模型，预计降 40% 成本
3. 缓存高频问题，命中率 20% 可省 20%
4. 降低最大步数，避免长尾任务
5. 输出限长

**自测标准**：能给出"手段 + 预估比例"，而不是泛泛而谈。

---

## 7. 面试问法

| 面试官可能这么问 | 你要答到的点 |
| --- | --- |
| 你的 Agent 怎么部署的 | 无状态服务 + Redis 会话 + 队列 + Docker；水平扩展 |
| 怎么监控 | 日志（request_id）+ trace（span）+ 指标（延迟/token/成功率） |
| 模型挂了怎么办 | 超时/重试/熔断/降级，逐层说 |
| 怎么控制成本 | 上下文压缩、模型分级、缓存、步数上限、token 预算 |
| 流式接口出错怎么处理 | 推 error 事件，不能改状态码 |
| 怎么保证改动不回退 | 黄金集 + CI 自动评测 + 灰度 |

---

## 8. 本关验收

- [ ] 能画出可水平扩展的 Agent 服务架构
- [ ] 能说出 trace 里应该记录哪些字段
- [ ] 能针对"40 秒慢请求"列出 5 个排查方向
- [ ] 能设计用户级 + 全局双层限流
- [ ] 能报出自己项目的成本、延迟、成功率数字

> 数字是这一关的核心产出——没有数字，前面的架构讲得再好也像背的。
