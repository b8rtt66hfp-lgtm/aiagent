# F3 · API 集成与工具封装

> 对应关卡：L3 ~ L4
> 面试权重：高。JD 点名"API 集成"。这项能力决定你的 Agent 是"玩具"还是"能接真实系统"。

---

## 1. 一句话结论

**API 集成 = 协议理解 + 鉴权 + 超时 + 重试 + 错误语义 + 幂等 + 可观测。**
面试官问"你怎么接第三方接口"，想听的不是 `requests.get()`，而是**接口挂了、慢了、限流了、重复调用了，你的系统怎么表现**。

---

## 2. 概念

### 2.1 HTTP 基础（必须秒答）

| 概念 | 要点 |
| --- | --- |
| 方法 | GET 查、POST 建/执行、PUT 全量改、PATCH 局部改、DELETE 删 |
| 状态码 | 2xx 成功；3xx 重定向；4xx **客户端问题，重试无用**；5xx **服务端问题，可重试**；429 限流，可重试但要看 `Retry-After` |
| 头部 | `Content-Type`、`Authorization`、`Idempotency-Key`、`X-Request-Id` |
| 幂等性 | GET/PUT/DELETE 天然幂等，POST 不幂等（重复提交会建两条订单） |

**这是面试高频**："接口 500 了要不要重试？"答：5xx/超时/连接错误可重试；4xx（除 429）重试只会浪费资源和放大错误。

### 2.2 异步 HTTP 客户端

Agent 一次任务里往往并发调多个外部接口，必须用异步：

```python
import httpx

async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as client:
    r = await client.get("https://api.example.com/orders", params={"page": 1})
    r.raise_for_status()
    data = r.json()
```

超时要分四类（面试可加分）：连接超时、读超时、写超时、连接池等待超时。只给一个 `timeout=10` 是粗糙做法。

### 2.3 鉴权的三种常见形态

1. **API Key**：放 header（`X-API-Key`）或 query，**绝不能写进 prompt、不能存代码里**，用环境变量/密钥管理。
2. **Bearer Token**：`Authorization: Bearer <token>`，有过期时间，需要刷新逻辑。
3. **OAuth2 client credentials**：用 client_id/secret 换 access_token，带 `expires_in`，提前刷新并缓存。

项目里至少要做到：**token 缓存 + 过期前刷新 + 刷新失败降级**。

### 2.4 重试的正确姿势

```
可重试：连接失败、超时、5xx、429
不可重试：401/403（先修鉴权）、400（参数错）、404
策略：指数退避 + 随机抖动 + 最大次数 + 总超时预算
```

为什么要**抖动（jitter）**：避免所有客户端在同一毫秒重试，形成重试风暴。

### 2.5 幂等性

- 对外：写操作带 `Idempotency-Key`（UUID），服务端按 key 去重。
- 对内：Agent 重试工具调用时，不能让"创建工单"被执行两次。设计工具时把"写操作"标记出来，重试前先查状态。

### 2.6 把 API 封装成 Agent 工具（本岗位核心）

一个合格的工具有四个要素：

```python
from typing import Literal
from pydantic import BaseModel, Field


class QueryOrderArgs(BaseModel):
    order_id: str = Field(description="订单号，形如 ORD-2026-0001")
    detail: Literal["summary", "full"] = Field(default="summary", description="返回详细程度")


def query_order(order_id: str, detail: str = "summary") -> dict:
    """查询订单状态。仅用于查询，不修改任何数据。"""
    ...
```

要点：

1. **描述写清"什么时候用、什么时候不用"**，模型靠描述选工具。
2. **参数用 schema 约束**（枚举、格式），比自然语言叮嘱可靠。
3. **错误要转成人话返回给模型**：`"订单不存在，请向用户确认订单号"` 比抛 traceback 有用。
4. **结果要裁剪**：接口返回 200 个字段，只留模型需要的 3 个，否则上下文爆炸。
5. **权限最小化**：查订单的工具不能有退款权限。

---

## 3. 最小可运行代码

### 3.1 带重试与抖动的异步客户端

```python
import asyncio
import random
import httpx


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


async def request_json(
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    client: httpx.AsyncClient | None = None,
    **kwargs,
) -> dict:
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0))
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await client.request(method, url, **kwargs)
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt == max_attempts:
                    raise ApiError(f"网络错误: {e}", retryable=True) from e
                await asyncio.sleep(base_delay * 2 ** (attempt - 1) * (1 + random.random()))
                continue

            if resp.status_code < 400:
                return resp.json()

            retryable = resp.status_code >= 500 or resp.status_code == 429
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", base_delay))
                await asyncio.sleep(wait)

            if not retryable or attempt == max_attempts:
                raise ApiError(
                    f"接口错误 {resp.status_code}: {resp.text[:200]}",
                    status=resp.status_code,
                    retryable=retryable,
                )
            await asyncio.sleep(base_delay * 2 ** (attempt - 1) * (1 + random.random()))
    finally:
        if own_client:
            await client.aclose()
```

### 3.2 封装成模型可调用的工具

```python
def query_order_tool(order_id: str) -> str:
    """查询订单状态。参数：order_id 订单号。"""
    try:
        data = asyncio.run(request_json("GET", f"https://api.shop.com/orders/{order_id}"))
    except ApiError as e:
        # 关键：把错误翻译成模型能理解的话，而不是 traceback
        if e.status == 404:
            return "订单不存在，请让用户确认订单号是否正确。"
        return f"查询失败（{e.status}），请稍后重试，不要重复调用。"
    return f"订单 {data['id']} 状态：{data['status']}，金额 {data['amount']} 元。"
```

---

## 4. 常见坑

1. **重试 4xx**：参数错了重试 3 次，既慢又污染日志。
2. **没有总超时**：单次 10s，重试 3 次 = 30s，接口已经超时了你的任务还在跑。要设**总预算**。
3. **异步里用同步库**：`requests` 会阻塞事件循环，整个服务变成串行。
4. **密钥写进代码/日志/prompt**：日志里打印 headers 是常见泄漏点。
5. **`resp.json()` 不处理非 JSON**：5xx 时返回 HTML 错误页，直接抛解析异常，掩盖真实错误。
6. **把完整响应塞给模型**：工具结果动辄几千 token，成本和延迟都会炸。必须裁剪。
7. **工具描述写"查询订单"**：模型不知道什么时候该用。要写"用户询问订单状态、物流进度时使用；不要用它修改订单"。
8. **忽略 429 的 `Retry-After`**：自己盲猜 1 秒，被限流更久。

---

## 5. 练习

### 练习 1（入门）· 判断可重试性

下列情况哪些应该重试？说明理由：A. 500；B. 401；C. 429；D. 连接超时；E. 400 参数错误；F. 404。

### 练习 2（核心）· 补全退避

写出第 n 次重试的等待时间公式，要求：指数增长、有上限、带抖动。并解释为什么必须有上限。

### 练习 3（核心）· 工具错误设计

你的"创建工单"工具调用失败了，用 3 句话分别给出：① 参数错误时返回给模型的内容；② 服务端 500 时返回的内容；③ 你已经重试 3 次仍然失败时返回的内容。

### 练习 4（进阶）· 幂等

Agent 在"创建订单"时网络超时，无法确定是否创建成功。你会怎么设计，避免用户被扣两次钱？

### 练习 5（进阶）· 并发集成

你需要在一个任务里查 5 个不同系统的数据，每个接口平均 800ms。串行要 4 秒。写出并发实现，并说明如何控制"最多同时 3 个请求"。

---

## 6. 参考答案与自测标准

### 练习 1 答案

- A 500：重试（服务端临时故障）
- B 401：不重试（凭据问题，先刷新 token，刷新后再试一次是另一种逻辑）
- C 429：重试，按 `Retry-After`
- D 连接超时：重试
- E 400：不重试（参数错，重试无意义）
- F 404：不重试

**自测标准**：能一句话总结"4xx 除了 429 都不重试"。

---

### 练习 2 答案

```
wait = min(cap, base * 2**(n-1)) * (1 + random.uniform(0, 1))
```

有上限的原因：不加 cap，第 10 次重试要等 8 分钟，请求早已超时，属于无意义等待。

**自测标准**：能说出"抖动是为了打散重试时间，cap 是为了保证总预算可控"。

---

### 练习 3 答案

1. 参数错误：`"参数不合法：缺少必填字段 title。请向用户确认工单标题后重试。"`（明确、可行动、不要重复调用）
2. 500：`"工单系统暂时不可用，请告知用户稍后再试，不要立即重复提交。"`
3. 重试耗尽：`"创建工单失败，已重试 3 次。请停止重试并告知用户联系人工客服。"`

**自测标准**：三条都不是 traceback，且都告诉模型"下一步该做什么"。

---

### 练习 4 答案

- 客户端生成 `Idempotency-Key`（UUID），随创建请求发送，重试时复用**同一个 key**
- 服务端按 key 去重，重复请求返回第一次的结果
- 超时后先查询状态（按 key 或业务唯一键），确认未创建再重试
- 写操作单独标记，Agent 层禁止无脑重试

**自测标准**：能说出"重试必须携带同一个幂等键，而不是每次生成新的"。

---

### 练习 5 答案

```python
import asyncio

sem = asyncio.Semaphore(3)


async def fetch(client, url):
    async with sem:
        return await request_json("GET", url, client=client)


async def main(urls):
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(fetch(client, u) for u in urls))
    return results
```

要点：共享一个 `AsyncClient`（复用连接池）；`Semaphore` 限制并发；`gather` 中某个失败会整体抛出，需要 `return_exceptions=True` 或包 try。

**自测标准**：能说出"为什么要共享 client"和"gather 默认的异常行为"。

---

## 7. 面试问法

| 面试官可能这么问 | 你要答到的点 |
| --- | --- |
| 你怎么保证调用第三方接口的稳定性？ | 超时（分类）/重试（只对 5xx、429、超时）/退避+抖动/总预算/熔断降级 |
| 接口 500 了怎么办？ | 可重试 + 退避；连续失败熔断；返回模型可理解的错误 |
| 写接口重复提交怎么防？ | 幂等键、先查后写、重试复用同一 key |
| 怎么把接口给 Agent 用？ | schema 约束参数、描述写使用场景、错误转人话、结果裁剪、权限最小化 |
| 大量并发请求怎么处理？ | 异步 + 连接池 + Semaphore 限流 + 队列削峰 |
| 密钥怎么管理？ | 环境变量/密钥服务，不进代码、不进日志、不进 prompt |

**加分回答**：主动提"我给每个外部调用都带了 request-id 和 trace，出问题能定位到具体是哪个工具的第几次重试"。

---

## 8. 本关验收

- [ ] 能当场写出带重试和退避的异步请求函数
- [ ] 能说出 4 类超时和 3 类可重试错误
- [ ] 能把一个真实接口封装成"模型能用对"的工具
- [ ] 能解释幂等键如何防重复下单
- [ ] 能说出工具结果为什么要裁剪
