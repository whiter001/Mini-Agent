# MiniMax Anthropic-Compatible API Guide

> This note condenses the most useful details from MiniMax's Anthropic-compatible overview, the `POST /anthropic/v1/messages` reference, the model discovery endpoints, and the error-code page into one practical guide for this repository.

## Why this guide exists

Use this guide when you want to:

- call MiniMax through the Anthropic SDK directly
- understand what `provider: anthropic` means inside `Mini Agent`
- handle streaming output and tool use correctly
- debug auth failures, rate limits, parameter errors, or token-limit issues quickly

## Base URL quick reference

| Scenario | What to configure as base URL | Notes |
| --- | --- | --- |
| `Mini Agent` config | `https://api.minimax.io` or `https://api.minimaxi.com` | Do **not** append `/anthropic` manually; `mini_agent/llm/llm_wrapper.py` adds the provider-specific suffix automatically |
| Anthropic SDK directly | `https://api.minimaxi.com/anthropic` | This is the Anthropic-compatible base shown in the official China-site docs |
| Raw HTTP request | `https://api.minimaxi.com/anthropic/v1/messages` | Full endpoint for message creation |

If you are on a different MiniMax platform host, replace the domain and keep the same Anthropic-compatible path structure.

## How to configure this repository

For this project, the minimal Anthropic-style configuration is:

```yaml
api_key: "YOUR_API_KEY_HERE"
api_base: "https://api.minimaxi.com"
provider: "anthropic"
model: "MiniMax-M2.7"
temperature: 0.7
```

Important detail:

- `config.yaml` should usually contain the platform root URL
- `mini_agent/llm/llm_wrapper.py` normalizes MiniMax domains and appends `/anthropic` for the Anthropic provider
- `mini_agent/llm/anthropic_client.py` then passes that URL to the Anthropic SDK client

Relevant files:

- `mini_agent/llm/llm_wrapper.py`
- `mini_agent/llm/anthropic_client.py`
- `mini_agent/config/config-example.yaml`

## Core endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/anthropic/v1/messages` | text generation, streaming responses, tool use |
| `GET` | `/anthropic/v1/models` | list currently available models |
| `GET` | `/anthropic/v1/models/{model_id}` | inspect a specific model |

### Authentication and headers

All three endpoints use Bearer auth:

- `Authorization: Bearer <API_KEY>`
- `Content-Type: application/json`

When debugging production issues, keep the response `trace_id` header if available. The official error-code page explicitly recommends including it when asking MiniMax for support.

## Minimal examples

### Anthropic SDK example

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
                {"type": "text", "text": "Explain Mini Agent in three sentences."}
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

### Raw HTTP example

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
        "content": "Explain what an Agent is in one sentence."
      }
    ]
  }'
```

## Request body fields that matter most

| Field | Practical importance | Notes |
| --- | --- | --- |
| `model` | required | The Anthropic compatibility page lists `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-M2.5-highspeed`, `MiniMax-M2.1`, `MiniMax-M2.1-highspeed`, and `MiniMax-M2`; using `/anthropic/v1/models` is the safest way to discover the live list |
| `messages` | required | Conversation history; at minimum send the latest user turn |
| `system` | common | System instruction, as a string or text-block array |
| `stream` | common | Set to `true` for streaming output |
| `max_tokens` | common | Output cap; if generation stops early or you hit token-limit errors, lower it or adjust against the latest platform limits |
| `temperature` | common | Valid range is `(0, 1]`; the official compatibility page recommends `1.0`, while this repository explicitly sends `0.7` by default unless you override it in `config.yaml` |
| `top_p` | optional | Nucleus sampling parameter in `(0, 1]` |
| `tools` | common for agent workflows | Declares callable tools |
| `tool_choice` | common for agent workflows | Controls tool selection behavior |
| `thinking` | optional | Requests reasoning content |
| `metadata` | optional | Useful for tracing business context |

### Content block support

According to the compatibility notes, message content supports:

- `text`
- `tool_use`
- `tool_result`
- `thinking`

Current limitations:

- image input is not supported
- document input is not supported

That makes this endpoint a strong fit for text-heavy assistants, coding workflows, and tool-using agents, but not for multimodal document or image ingestion.

## How to read the response

For non-streaming responses, these fields matter most:

| Field | Meaning |
| --- | --- |
| `id` | unique response ID |
| `type` | always `message` |
| `role` | always `assistant` |
| `model` | model used for this response |
| `content` | ordered list of content blocks, often including `thinking` and `text` |
| `usage.input_tokens` | input token count |
| `usage.output_tokens` | output token count |
| `stop_reason` | usually `end_turn`, `max_tokens`, or `stop_sequence` |
| `base_resp` | low-level status details; `status_code = 0` usually means success |

Treat `content` as structured blocks, not as a single flat string. That becomes especially important once tool use enters the picture.

## Streaming event flow

When `stream=true`, the endpoint emits a sequence of streaming events. The common flow is:

1. `message_start`
2. `ping`
3. `content_block_start`
4. `content_block_delta`
5. `content_block_stop`
6. `message_delta`
7. `message_stop`

The most useful delta types are typically:

- `thinking_delta`
- `text_delta`
- `signature_delta`

If you display reasoning and final text separately, buffer them by block index instead of blindly concatenating every chunk together.

## Tool use and multi-turn conversations

The official Anthropic-compatible guide calls out one rule very clearly: in multi-turn function-calling flows, you must feed the **full assistant message** back into the next turn.

In practice, do not keep only the rendered text. Preserve the complete `response.content`, which may include:

- `thinking`
- `text`
- `tool_use`
- other compatible content blocks

A safe loop looks like this:

1. send the user request
2. receive the assistant message
3. execute the requested tool
4. append the **full assistant message** to history
5. append the tool result as `tool_result`
6. send the next request

If step 4 only stores a flattened text answer, the next turn often loses context and starts behaving inconsistently.

## Model discovery endpoints

### List available models

```bash
curl --request GET \
  --url https://api.minimaxi.com/anthropic/v1/models \
  --header "Authorization: Bearer YOUR_API_KEY_HERE"
```

Good use cases:

- verify model availability at startup
- avoid hard-coding model names in deployment pipelines

### Retrieve one model

```bash
curl --request GET \
  --url https://api.minimaxi.com/anthropic/v1/models/MiniMax-M2.7 \
  --header "Authorization: Bearer YOUR_API_KEY_HERE"
```

Typical fields include:

- `id`
- `created_at`
- `display_name`
- `type`

## Compatibility boundaries

### Fully supported

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

### Accepted but ignored

- `top_k`
- `stop_sequences`
- `service_tier`
- `mcp_servers`
- `context_management`
- `container`

### Not currently supported

- image input
- document input

If you migrate from a native Anthropic workflow and a parameter appears to have no effect, check this ignored-parameter set first.

## Common errors and quick fixes

| Code | Typical meaning | What to check |
| --- | --- | --- |
| `1002` | rate limit exceeded | reduce concurrency or request a higher quota |
| `1004` | unauthorized / token mismatch | verify Bearer auth and make sure the key matches the platform host |
| `1008` | insufficient balance | check account balance or Token Plan quota |
| `1039` | token limit | lower `max_tokens` or shorten the prompt/history |
| `2013` | invalid parameter | re-check JSON structure, field types, and enum values |
| `2049` | invalid API key | confirm the key is complete and belongs to the current platform |
| `2056` | Token Plan resource limit reached | wait for the next quota window or upgrade resources |

Two practical habits help a lot:

- include `trace_id` when reporting issues
- when switching from this repository to direct SDK debugging, verify whether your base URL already includes `/anthropic`

## TPS in plain language

The official FAQ defines TPS as:

$$
TPS = \frac{output\ tokens}{time\ of\ last\ token - time\ of\ first\ token}
$$

So TPS measures generation speed **after the first token starts arriving**, not the full wall-clock latency from request start to request end.

## Official references

- [Anthropic API compatibility overview](https://platform.minimaxi.com/docs/api-reference/text-anthropic-api)
- [Anthropic-compatible text messages endpoint](https://platform.minimaxi.com/docs/api-reference/text-chat-anthropic)
- [Anthropic model list](https://platform.minimaxi.com/docs/api-reference/models/anthropic/list-models)
- [Anthropic model detail](https://platform.minimaxi.com/docs/api-reference/models/anthropic/retrieve-model)
- [Error code reference](https://platform.minimaxi.com/docs/api-reference/errorcode)
- [API FAQ](https://platform.minimaxi.com/docs/faq/about-apis)

## One-line takeaway

If you use the Anthropic SDK directly, point it at the Anthropic-compatible base URL with `/anthropic`; if you use `Mini Agent`, keep `api_base` at the platform root and let the repository append the Anthropic suffix automatically.