# MiniMax Anthropic 兼容 API 说明

> 面向本项目使用者的补充文档：把官方 Anthropic 兼容页、`/anthropic/v1/messages` 接口页、模型查询页和错误码页里最常用的信息整理到一起，方便你在 `Mini Agent` 中直接落地。

## 这篇文档解决什么问题

如果你正在做下面这些事，这篇文档会比较顺手：

- 想直接用 Anthropic SDK 调 MiniMax 模型
- 想弄清楚 `Mini Agent` 里的 `provider: anthropic` 到底会请求什么地址
- 想知道流式输出、工具调用、多轮对话回传时要注意什么
- 想快速排查鉴权失败、限流、参数错误或 Token 超限

## 接入方式速查

| 使用场景 | Base URL 应该怎么填 | 备注 |
| --- | --- | --- |
| 在本项目的 `config.yaml` 中配置 | `https://api.minimax.io` 或 `https://api.minimaxi.com` | **不要手动追加** `/anthropic`；`mini_agent/llm/llm_wrapper.py` 会按 `provider` 自动补全 |
| 直接使用 Anthropic SDK | `https://api.minimaxi.com/anthropic` | 这是官方中国站文档给出的兼容入口 |
| 直接调用 HTTP 接口 | `https://api.minimaxi.com/anthropic/v1/messages` | 对应 `POST /anthropic/v1/messages` |

> 如果你使用的是海外平台，请将域名替换为对应平台的 API Host，并保留相同的兼容路径结构。

## 在 Mini Agent 里如何配置

本项目默认支持 Anthropic 协议，最简配置如下：

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://api.minimaxi.com"
provider: "anthropic"
model: "MiniMax-M2.7"
temperature: 0.7
```

这里有一个很容易踩的点：

- `config.yaml` 里的 `api_base` 填**根地址**即可
- 代码会在 MiniMax 域名下自动补成 `.../anthropic`
- 如果你把 `api_base` 直接写成 `https://api.minimaxi.com/anthropic`，当前仓库也会先规范化再处理，但文档约定仍推荐只填根地址，配置更清晰

对应实现可以参考：

- `mini_agent/llm/llm_wrapper.py`
- `mini_agent/llm/anthropic_client.py`
- `mini_agent/config/config-example.yaml`

## 核心接口一览

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/anthropic/v1/messages` | 发起文本对话、流式输出、工具调用 |
| `GET` | `/anthropic/v1/models` | 查询当前可用模型列表 |
| `GET` | `/anthropic/v1/models/{model_id}` | 查询单个模型详情 |

### 认证与请求头

所有接口都使用 Bearer Token：

- `Authorization: Bearer <API_KEY>`
- `Content-Type: application/json`

排查线上问题时，建议同时保留响应 Header 中的 `trace_id`，官方错误码页明确说明了这个字段有助于技术支持定位问题。

## 最小可用示例

### 方案一：直接使用 Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="YOUR_API_KEY_HERE",
    base_url="https://api.minimaxi.com/anthropic",
)

message = client.messages.create(
    model="MiniMax-M2.7",
    max_tokens=1024,
    temperature=0.7,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请用三句话介绍 Mini Agent。"}
            ],
        }
    ],
)

for block in message.content:
    if block.type == "thinking":
        print(block.thinking)
    elif block.type == "text":
        print(block.text)
```

### 方案二：直接发 HTTP 请求

```bash
curl --request POST \
  --url https://api.minimaxi.com/anthropic/v1/messages \
  --header "Authorization: Bearer YOUR_API_KEY_HERE" \
  --header "Content-Type: application/json" \
  --data '{
    "model": "MiniMax-M2.7",
    "max_tokens": 1024,
    "temperature": 0.7,
    "messages": [
      {
        "role": "user",
        "content": "请用一句话解释什么是 Agent。"
      }
    ]
  }'
```

### 方案三：在本项目里继续用 `Mini Agent`

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://api.minimaxi.com"
provider: "anthropic"
model: "MiniMax-M2.7"
temperature: 0.7
```

这时仓库内部会自动把 Anthropic 兼容路径拼成可请求的 base URL，你不用额外改 SDK 参数。

## 请求体重点字段

`POST /anthropic/v1/messages` 最常用的字段如下：

| 字段 | 是否常用 | 说明 |
| --- | --- | --- |
| `model` | 必填 | 模型 ID。官方兼容页列出了 `MiniMax-M2.7`、`MiniMax-M2.7-highspeed`、`MiniMax-M2.5`、`MiniMax-M2.5-highspeed`、`MiniMax-M2.1`、`MiniMax-M2.1-highspeed`、`MiniMax-M2` 等模型；更稳妥的做法是调用 `/anthropic/v1/models` 获取实时列表 |
| `messages` | 必填 | 对话历史。通常至少包含一条用户消息 |
| `system` | 常用 | 系统提示词，可用字符串，也可用文本内容块数组 |
| `stream` | 常用 | 设为 `true` 时启用流式输出 |
| `max_tokens` | 常用 | 控制输出长度上限。若生成因为长度结束或返回 Token 相关错误，优先下调或按最新平台文档调整 |
| `temperature` | 常用 | 取值范围为 `(0, 1]`；官方兼容文档建议使用 `1.0`，而本仓库默认会显式传 `0.7`，你也可以在 `config.yaml` 中自行覆盖 |
| `top_p` | 可选 | 核采样参数，取值范围为 `(0, 1]` |
| `tools` | 工具调用场景常用 | 定义可供模型调用的工具 |
| `tool_choice` | 工具调用场景常用 | 控制工具选择策略 |
| `thinking` | 可选 | 控制推理内容输出 |
| `metadata` | 可选 | 透传元信息，便于业务跟踪 |

### `messages` 支持哪些内容

官方兼容说明页面中，`messages` 的内容块支持情况可概括为：

- 支持：`text`、`tool_use`、`tool_result`、`thinking`
- 暂不支持：`image`、`document`

这意味着当前 Anthropic 兼容文本接口更适合：

- 文本问答
- 编程与 Agent 工作流
- 工具调用 / Function Call
- 带思维链输出的多轮对话

而不适合直接把图片、文档内容当作输入塞进同一个接口。

## 响应结构怎么读

非流式响应中，最常关注这些字段：

| 字段 | 说明 |
| --- | --- |
| `id` | 本次响应的唯一 ID |
| `type` | 固定为 `message` |
| `role` | 固定为 `assistant` |
| `model` | 本次响应使用的模型 |
| `content` | 内容块数组，通常会看到 `thinking` 和 `text` |
| `usage.input_tokens` | 输入 Token 数 |
| `usage.output_tokens` | 输出 Token 数 |
| `stop_reason` | 停止原因，常见值为 `end_turn`、`max_tokens`、`stop_sequence` |
| `base_resp` | 底层状态信息，`status_code=0` 通常表示成功 |

如果你只把 `content` 当成纯字符串处理，很容易在工具调用场景里把关键块弄丢。更稳妥的做法是把它当成**内容块列表**处理。

## 流式输出事件说明

当 `stream=true` 时，接口会以事件流持续返回增量。常见事件顺序如下：

1. `message_start`
2. `ping`
3. `content_block_start`
4. `content_block_delta`
5. `content_block_stop`
6. `message_delta`
7. `message_stop`

其中最有用的增量类型通常是：

- `thinking_delta`
- `text_delta`
- `signature_delta`

如果你要把思考过程和最终文本分别展示给用户，建议按内容块索引分别缓存，而不是把所有流式片段直接拼进同一个字符串。

## 工具调用与多轮对话的关键注意事项

官方 Anthropic 兼容页特别强调：**在多轮 Function Call / Tool Use 场景中，必须把模型返回的完整 assistant 消息回填到历史中。**

换句话说，不要只保留最终文本；应该保留完整的 `response.content`，其中可能包含：

- `thinking`
- `text`
- `tool_use`
- 其他兼容内容块

推荐的回合顺序是：

1. 用户发起请求
2. 模型返回 assistant 消息（其中可能包含工具调用）
3. 你的程序执行工具
4. 把**完整 assistant 消息**追加回对话历史
5. 再把工具结果以 `tool_result` 形式回传给模型
6. 发起下一轮请求

如果你在第 4 步只回填纯文本，模型很容易在下一轮“失忆”，表现通常是：

- 工具参数前后不一致
- 推理链断开
- 明明调用过工具，却继续重复提问

## 模型发现接口

### 获取模型列表

```bash
curl --request GET \
  --url https://api.minimaxi.com/anthropic/v1/models \
  --header "Authorization: Bearer YOUR_API_KEY_HERE"
```

适合用来做两件事：

- 启动时动态检查模型是否可用
- 避免把模型名硬编码到发布流程里

### 获取单个模型详情

```bash
curl --request GET \
  --url https://api.minimaxi.com/anthropic/v1/models/MiniMax-M2.7 \
  --header "Authorization: Bearer YOUR_API_KEY_HERE"
```

模型详情响应中通常会包含：

- `id`
- `created_at`
- `display_name`
- `type`

## 兼容性边界

根据官方 Anthropic 兼容说明页，可以把支持情况简单记成下面这样：

### 完全支持

- `model`
- `max_tokens`
- `stream`
- `system`
- `temperature`
- `top_p`
- `tools`
- `tool_choice`
- `thinking`
- `metadata`

### 会被忽略

- `top_k`
- `stop_sequences`
- `service_tier`
- `mcp_servers`
- `context_management`
- `container`

### 当前不支持

- 图像输入
- 文档输入

如果你从原生 Anthropic 应用迁移过来，遇到“参数传了但效果像没生效”的情况，优先检查是不是落在“会被忽略”的这一组里。

## 常见错误与排查建议

下表整理了和文本接口最相关的一批错误码：

| 错误码 | 常见含义 | 排查建议 |
| --- | --- | --- |
| `1002` | 请求频率超限 | 降低并发或按官方速率限制方案申请更高额度 |
| `1004` | 未授权 / Token 不匹配 | 检查 `Authorization` 头和 API Key 所属平台是否一致 |
| `1008` | 余额不足 | 检查账户余额或 Token Plan 配额 |
| `1039` | Token 限制 | 降低 `max_tokens` 或缩短输入上下文 |
| `2013` | 参数错误 | 重点检查 JSON 结构、字段类型和枚举值 |
| `2049` | 无效 API Key | 确认 API Key 是否复制完整、是否来自当前平台 |
| `2056` | 超出 Token Plan 资源限制 | 等待下一个时间窗口，或升级资源方案 |

补充两个实战建议：

- 提交问题时附上 `trace_id`，比单贴报错文案更容易定位
- 如果你从本项目切到原生 SDK 调试，先确认 base URL 是否已经带上 `/anthropic`

## TPS 和性能理解

官方 FAQ 对 TPS（Tokens Per Second）的说明可以简化为：

$$
TPS = \frac{输出\ token\ 数量}{最后一个\ token\ 输出时间 - 第一个\ token\ 输出时间}
$$

也就是说，TPS 衡量的是**开始吐字以后**的生成速度，而不是整次请求从发出到结束的总耗时。页面里标注的 TPS 更适合作为参考值，不适合作为严格 SLA。

## 相关官方文档

- [Anthropic API 兼容](https://platform.minimaxi.com/docs/api-reference/text-anthropic-api)
- [文本对话（Anthropic API 兼容）](https://platform.minimaxi.com/docs/api-reference/text-chat-anthropic)
- [获取模型列表](https://platform.minimaxi.com/docs/api-reference/models/anthropic/list-models)
- [获取单个模型详情](https://platform.minimaxi.com/docs/api-reference/models/anthropic/retrieve-model)
- [错误码查询](https://platform.minimaxi.com/docs/api-reference/errorcode)
- [接口相关 FAQ](https://platform.minimaxi.com/docs/faq/about-apis)

## 一句话总结

如果你是 **直接使用 Anthropic SDK**，请把 base URL 设成带 `/anthropic` 的兼容入口；如果你是 **在本仓库里使用 `Mini Agent`**，则继续把 `api_base` 写成平台根地址即可，仓库会自动补全 Anthropic 兼容路径。