"""
Anthropic API Router - 内置Claude API支持
支持所有Claude 3, 3.5, 3.7, 4, 4.5模型
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Union, AsyncIterator

import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from open_webui.models.models import Models
from open_webui.models.users import UserModel, Users
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access
from open_webui.utils.misc import pop_system_message
from open_webui.env import SRC_LOG_LEVELS, BYPASS_MODEL_ACCESS_CONTROL

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))

router = APIRouter()

# Anthropic API 常量
API_VERSION = "2023-06-01"
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
MAX_IMAGE_SIZE = 5 * 1024 * 1024
REQUEST_TIMEOUT = 120

# 模型最大输出tokens
MODEL_MAX_TOKENS = {
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-sonnet-20240620": 8192,
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-opus-latest": 4096,
    "claude-3-5-sonnet-latest": 8192,
    "claude-3-5-haiku-latest": 8192,
    "claude-3-7-sonnet-latest": 16384,
    "claude-sonnet-4-20250514": 16384,
    "claude-sonnet-4-0": 16384,
    "claude-opus-4-20250514": 16384,
    "claude-opus-4-0": 16384,
    "claude-sonnet-4-5-20250929": 32768,
    "claude-sonnet-4-5": 32768,
    "claude-opus-4-1-20250805": 32768,
    "claude-opus-4-1": 32768,
    "claude-opus-4-5-20251101": 32768,
    "claude-opus-4-5": 32768,
}

# 模型上下文长度
MODEL_CONTEXT_LENGTH = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-opus-latest": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-haiku-latest": 200000,
    "claude-3-7-sonnet-latest": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-sonnet-4-0": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-opus-4-0": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-opus-4-1-20250805": 200000,
    "claude-opus-4-1": 200000,
    "claude-opus-4-5-20251101": 200000,
    "claude-opus-4-5": 200000,
}

# Beta headers
BETA_HEADER = "prompt-caching-2024-07-31"
PDF_BETA_HEADER = "pdfs-2024-09-25"
OUTPUT_128K_BETA = "output-128k-2025-02-19"
TOKEN_EFFICIENT_TOOLS = "token-efficient-tools-2025-02-19"

# 所有Claude模型列表
CLAUDE_MODELS = [
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-latest",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
    "claude-opus-4-5-20251101",
    "claude-opus-4-5",
]

# 支持extended thinking的模型
THINKING_MODELS = [
    "claude-3-7-sonnet-latest",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-0",
    "claude-opus-4-20250514",
    "claude-opus-4-0",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
    "claude-opus-4-5-20251101",
    "claude-opus-4-5",
]


def get_anthropic_connection(request: Request) -> Optional[dict]:
    """从全局连接中获取Anthropic连接配置"""
    # AppConfig.__getattr__ 会自动返回 .value，所以无需手动访问
    connections = request.app.state.config.GLOBAL_CONNECTIONS or []
    for conn in connections:
        if conn.get("type") == "anthropic":
            return conn
    return None


def get_user_api_key_for_anthropic(user: UserModel, connection_id: str) -> Optional[str]:
    """获取用户为Anthropic连接配置的API密钥"""
    if not user:
        return None
    user_data = Users.get_user_by_id(user.id)
    if user_data and user_data.settings:
        api_keys = user_data.settings.api_keys or {}
        return api_keys.get(connection_id)
    return None


async def cleanup_response(
    response: Optional[aiohttp.ClientResponse],
    session: Optional[aiohttp.ClientSession],
):
    if response:
        response.close()
    if session:
        await session.close()


def process_image(image_data: dict) -> dict:
    """处理图片数据转换为Anthropic格式"""
    if image_data["image_url"]["url"].startswith("data:image"):
        mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
        media_type = mime_type.split(":")[1].split(";")[0]

        if media_type not in SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported media type: {media_type}")

        image_size = len(base64_data) * 3 / 4
        if image_size > MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image size exceeds {MAX_IMAGE_SIZE/(1024*1024)}MB limit"
            )

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }
    else:
        return {
            "type": "image",
            "source": {"type": "url", "url": image_data["image_url"]["url"]},
        }


def process_content(content: Union[str, List[dict]]) -> List[dict]:
    """处理消息内容转换为Anthropic格式"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    processed_content = []
    for item in content:
        if item["type"] == "text":
            processed_content.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            processed_content.append(process_image(item))
    return processed_content


def process_messages(messages: List[dict]) -> List[dict]:
    """处理消息列表转换为Anthropic格式"""
    processed_messages = []
    for message in messages:
        processed_content = process_content(message["content"])
        processed_messages.append(
            {"role": message["role"], "content": processed_content}
        )
    return processed_messages


def convert_openai_tools_to_anthropic(tools: List[dict]) -> List[dict]:
    """将OpenAI工具格式转换为Anthropic格式"""
    anthropic_tools = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "function":
            func = tool.get("function", {})
            anthropic_tools.append({
                "name": func.get("name"),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        elif isinstance(tool, dict) and "name" in tool:
            anthropic_tools.append({
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            })
    return anthropic_tools


@router.get("/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    """获取可用的Claude模型列表"""
    connection = get_anthropic_connection(request)
    if not connection:
        return {"data": []}

    models = []
    for name in CLAUDE_MODELS:
        models.append({
            "id": name,
            "name": name,
            "object": "model",
            "owned_by": "anthropic",
            "context_length": MODEL_CONTEXT_LENGTH.get(name, 200000),
            "supports_vision": name != "claude-3-5-haiku-20241022",
            "supports_thinking": name in THINKING_MODELS,
        })

    return {"data": models}


async def get_all_models(request: Request, user: UserModel = None) -> dict:
    """获取所有Anthropic模型（供内部使用）"""
    connection = get_anthropic_connection(request)
    if not connection:
        return {"data": []}

    models = []
    for name in CLAUDE_MODELS:
        models.append({
            "id": name,
            "name": name,
            "object": "model",
            "owned_by": "anthropic",
            "anthropic": {"id": name},
            "connection_type": "external",
            "urlIdx": 0,
        })

    return {"data": models}


@router.post("/chat/completions")
async def generate_chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
    bypass_filter: Optional[bool] = False,
):
    """处理聊天完成请求"""
    if BYPASS_MODEL_ACCESS_CONTROL:
        bypass_filter = True

    connection = get_anthropic_connection(request)
    if not connection:
        raise HTTPException(
            status_code=400,
            detail="未配置Anthropic连接，请管理员在全局连接中添加",
        )

    # 密钥逻辑：管理员用管理员密钥，普通用户必须用自己的密钥
    admin_api_key = connection.get("api_key", "")
    connection_id = connection.get("id", "")
    base_url = connection.get("url", "https://api.anthropic.com").rstrip("/")

    # 构建完整的 API URL
    # 如果 URL 已经包含 /v1/messages，直接使用；否则添加路径
    if "/v1/messages" in base_url:
        api_url = base_url
    else:
        api_url = f"{base_url}/v1/messages"

    if user.role == "admin":
        api_key = admin_api_key
    else:
        user_api_key = get_user_api_key_for_anthropic(user, connection_id) if connection_id else None
        api_key = user_api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="请在设置 > 外部连接中配置对应的 API Key",
        )

    payload = {**form_data}
    metadata = payload.pop("metadata", None)
    model_id = form_data.get("model")

    # 检查模型信息
    model_info = Models.get_model_by_id(model_id)
    if model_info:
        if model_info.base_model_id:
            payload["model"] = model_info.base_model_id
            model_id = model_info.base_model_id

        if not bypass_filter and user.role == "user":
            if not (
                user.id == model_info.user_id
                or has_access(user.id, type="read", access_control=model_info.access_control)
            ):
                raise HTTPException(status_code=403, detail="Model not found")
    # 对于全局连接的Claude模型，允许所有已验证用户访问

    # 处理消息
    system_message, messages = pop_system_message(payload.get("messages", []))
    model_name = payload.get("model", model_id)

    # 获取模型最大tokens
    max_tokens_limit = MODEL_MAX_TOKENS.get(model_name, 4096)
    max_tokens = min(payload.get("max_tokens", max_tokens_limit), max_tokens_limit)

    # 构建Anthropic请求payload
    anthropic_payload = {
        "model": model_name,
        "messages": process_messages(messages),
        "max_tokens": max_tokens,
        "stream": payload.get("stream", False),
    }

    # 添加可选参数
    if payload.get("temperature") is not None:
        anthropic_payload["temperature"] = float(payload["temperature"])
    if payload.get("top_p") is not None:
        anthropic_payload["top_p"] = float(payload["top_p"])
    if payload.get("top_k") is not None:
        anthropic_payload["top_k"] = int(payload["top_k"])

    # 添加系统消息（包含隐藏的模型身份提示词）
    # 自动注入模型身份，前端不可见
    model_identity_prompt = f"You are {model_name}."

    if system_message:
        # 将模型身份提示词与用户系统消息合并
        system_str = f"{model_identity_prompt}\n\n{str(system_message)}"
    else:
        system_str = model_identity_prompt

    if len(system_str) > 4000:
        anthropic_payload["system"] = [
            {
                "type": "text",
                "text": system_str,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        anthropic_payload["system"] = system_str

    # 处理工具
    if "tools" in payload:
        anthropic_tools = convert_openai_tools_to_anthropic(payload["tools"])
        if anthropic_tools:
            if len(anthropic_tools) > 0:
                anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            anthropic_payload["tools"] = anthropic_tools
            anthropic_payload["tool_choice"] = {"type": "auto"}

    # 构建请求头
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }

    beta_headers = []
    # 检查是否需要cache beta header
    has_system_cache = isinstance(anthropic_payload.get("system"), list)
    has_tools_cache = (
        isinstance(anthropic_payload.get("tools"), list)
        and len(anthropic_payload.get("tools", [])) > 0
    )
    if has_system_cache or has_tools_cache:
        beta_headers.append(BETA_HEADER)

    # 添加token-efficient-tools beta header
    if "tools" in anthropic_payload:
        beta_headers.append(TOKEN_EFFICIENT_TOOLS)

    # 添加128K output beta header
    if model_name in THINKING_MODELS:
        beta_headers.append(OUTPUT_128K_BETA)

    if beta_headers:
        headers["anthropic-beta"] = ",".join(beta_headers)

    # 发送请求
    r = None
    session = None
    streaming = False

    log.info(f"[Anthropic] Sending request to: {api_url}")
    log.info(f"[Anthropic] Model: {model_name}, Stream: {anthropic_payload.get('stream')}")

    try:
        session = aiohttp.ClientSession(
            trust_env=True,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )

        r = await session.post(
            api_url,
            headers=headers,
            json=anthropic_payload,
        )

        log.info(f"[Anthropic] Response status: {r.status}")

        if anthropic_payload.get("stream"):
            streaming = True
            return StreamingResponse(
                stream_anthropic_response(r),
                media_type="text/event-stream",
                background=BackgroundTask(cleanup_response, response=r, session=session),
            )
        else:
            response_data = await r.json()

            if r.status != 200:
                error_msg = response_data.get("error", {}).get("message", str(response_data))
                log.error(f"[Anthropic] API error: {error_msg}")
                raise HTTPException(status_code=r.status, detail=error_msg)

            # 转换响应为OpenAI格式
            return convert_anthropic_response_to_openai(response_data, model_name)

    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Anthropic API error: {e}")
        raise HTTPException(
            status_code=r.status if r else 500,
            detail=f"Anthropic API error: {str(e)}",
        )
    finally:
        if not streaming:
            await cleanup_response(r, session)


async def stream_anthropic_response(response: aiohttp.ClientResponse) -> AsyncIterator[str]:
    """流式处理Anthropic响应并转换为OpenAI SSE格式"""
    try:
        buffer = ""
        async for chunk in response.content.iter_any():
            if not chunk:
                continue

            buffer += chunk.decode("utf-8")

            # 处理缓冲区中的完整行
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                # Anthropic SSE 格式: "event: xxx" 或 "data: {...}"
                if line.startswith("event:"):
                    # 跳过 event 行，我们只关心 data
                    continue

                if line.startswith("data:"):
                    try:
                        data_str = line[5:].strip()
                        if not data_str:
                            continue

                        data = json.loads(data_str)
                        event_type = data.get("type", "")

                        # 处理不同类型的事件
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                # 转换为OpenAI SSE格式
                                openai_chunk = {
                                    "id": "chatcmpl-anthropic",
                                    "object": "chat.completion.chunk",
                                    "created": 0,
                                    "model": "claude",
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }]
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n"

                        elif event_type == "message_stop":
                            openai_chunk = {
                                "id": "chatcmpl-anthropic",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": "claude",
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }]
                            }
                            yield f"data: {json.dumps(openai_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        elif event_type == "error":
                            error_msg = data.get("error", {}).get("message", "Unknown error")
                            log.error(f"Anthropic API error: {error_msg}")
                            # 仍然发送一个结束信号
                            yield "data: [DONE]\n\n"
                            return

                    except json.JSONDecodeError as e:
                        log.debug(f"JSON decode error: {e}, line: {line}")
                        continue
                    except Exception as e:
                        log.error(f"Error processing stream line: {e}")
                        continue

    except Exception as e:
        log.error(f"Stream error: {e}")
        yield "data: [DONE]\n\n"


def convert_anthropic_response_to_openai(response_data: dict, model: str) -> dict:
    """将Anthropic响应转换为OpenAI格式"""
    content_blocks = response_data.get("content", [])
    text_content = ""

    for block in content_blocks:
        if block.get("type") == "text":
            text_content += block.get("text", "")

    usage = response_data.get("usage", {})

    return {
        "id": response_data.get("id", "chatcmpl-anthropic"),
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content,
            },
            "finish_reason": response_data.get("stop_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        }
    }
