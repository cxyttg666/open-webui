import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, ConfigDict

from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.constants import ERROR_MESSAGES

log = logging.getLogger(__name__)

router = APIRouter()

####################################
# 数据模型
####################################


class GlobalConnectionModel(BaseModel):
    id: str
    name: str
    type: str  # openai, ollama, anthropic, custom
    url: str
    api_key: str = ""  # 管理员配置的API密钥
    auth_type: str = "bearer"  # bearer, none, session, system_oauth
    config: Optional[dict] = None  # 额外配置

    model_config = ConfigDict(from_attributes=True)


class GlobalConnectionResponse(BaseModel):
    """返回给非管理员用户的连接信息（不含API密钥）"""
    id: str
    name: str
    type: str
    url: str
    auth_type: str = "bearer"
    config: Optional[dict] = None

    model_config = ConfigDict(from_attributes=True)


class GlobalConnectionForm(BaseModel):
    name: str
    type: str
    url: str
    api_key: str = ""
    auth_type: str = "bearer"
    config: Optional[dict] = None


class GlobalConnectionUpdateForm(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    url: Optional[str] = None
    api_key: Optional[str] = None
    auth_type: Optional[str] = None
    config: Optional[dict] = None


####################################
# API 路由
####################################


@router.get("/")
async def get_global_connections(
    request: Request,
    user=Depends(get_verified_user),
):
    """获取所有全局连接配置
    - 管理员：返回完整信息（含API密钥）
    - 普通用户：返回不含API密钥的信息
    """
    # AppConfig.__getattr__ 会自动返回 .value，所以无需手动访问
    connections = request.app.state.config.GLOBAL_CONNECTIONS or []

    if user.role == "admin":
        # 管理员可以看到完整信息
        return [GlobalConnectionModel(**conn) for conn in connections]
    else:
        # 普通用户只能看到基本信息（不含API密钥）
        return [
            GlobalConnectionResponse(
                id=conn.get("id", ""),
                name=conn.get("name", ""),
                type=conn.get("type", ""),
                url=conn.get("url", ""),
                auth_type=conn.get("auth_type", "bearer"),
                config=conn.get("config"),
            )
            for conn in connections
        ]


@router.post("/", response_model=GlobalConnectionModel)
async def create_global_connection(
    request: Request,
    form_data: GlobalConnectionForm,
    user=Depends(get_admin_user),
):
    """创建新的全局连接配置（仅管理员）"""
    connections = list(request.app.state.config.GLOBAL_CONNECTIONS or [])

    # 移除URL尾部的斜杠
    url = form_data.url.rstrip("/")

    # 创建新连接
    new_connection = GlobalConnectionModel(
        id=str(uuid.uuid4()),
        name=form_data.name,
        type=form_data.type,
        url=url,
        api_key=form_data.api_key,
        auth_type=form_data.auth_type,
        config=form_data.config,
    )

    connections.append(new_connection.model_dump())
    # AppConfig.__setattr__ 会自动处理 .value 和 .save()
    request.app.state.config.GLOBAL_CONNECTIONS = connections

    # 同步到OPENAI_API配置（如果是openai类型）
    if form_data.type == "openai":
        _sync_openai_config(request)

    log.info(f"Admin {user.id} created global connection: {new_connection.name}")
    return new_connection


@router.get("/{connection_id}")
async def get_global_connection(
    request: Request,
    connection_id: str,
    user=Depends(get_verified_user),
):
    """获取指定的全局连接配置"""
    connections = request.app.state.config.GLOBAL_CONNECTIONS or []

    for conn in connections:
        if conn["id"] == connection_id:
            if user.role == "admin":
                return GlobalConnectionModel(**conn)
            else:
                return GlobalConnectionResponse(
                    id=conn.get("id", ""),
                    name=conn.get("name", ""),
                    type=conn.get("type", ""),
                    url=conn.get("url", ""),
                    auth_type=conn.get("auth_type", "bearer"),
                    config=conn.get("config"),
                )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ERROR_MESSAGES.NOT_FOUND,
    )


@router.post("/{connection_id}", response_model=GlobalConnectionModel)
async def update_global_connection(
    request: Request,
    connection_id: str,
    form_data: GlobalConnectionUpdateForm,
    user=Depends(get_admin_user),
):
    """更新全局连接配置（仅管理员）"""
    connections = list(request.app.state.config.GLOBAL_CONNECTIONS or [])

    for i, conn in enumerate(connections):
        if conn["id"] == connection_id:
            # 更新字段
            update_data = form_data.model_dump(exclude_unset=True)

            # 移除URL尾部的斜杠
            if "url" in update_data:
                update_data["url"] = update_data["url"].rstrip("/")

            connections[i] = {**conn, **update_data}
            request.app.state.config.GLOBAL_CONNECTIONS = connections

            # 同步到OPENAI_API配置
            conn_type = connections[i].get("type", "")
            if conn_type == "openai":
                _sync_openai_config(request)

            log.info(
                f"Admin {user.id} updated global connection: {connections[i]['name']}"
            )
            return GlobalConnectionModel(**connections[i])

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ERROR_MESSAGES.NOT_FOUND,
    )


@router.delete("/{connection_id}")
async def delete_global_connection(
    request: Request,
    connection_id: str,
    user=Depends(get_admin_user),
):
    """删除全局连接配置（仅管理员）"""
    connections = list(request.app.state.config.GLOBAL_CONNECTIONS or [])

    for i, conn in enumerate(connections):
        if conn["id"] == connection_id:
            deleted_conn = connections.pop(i)
            request.app.state.config.GLOBAL_CONNECTIONS = connections

            # 同步到OPENAI_API配置
            if deleted_conn.get("type") == "openai":
                _sync_openai_config(request)

            log.info(
                f"Admin {user.id} deleted global connection: {deleted_conn['name']}"
            )
            return {"success": True}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=ERROR_MESSAGES.NOT_FOUND,
    )


def _sync_openai_config(request: Request):
    """将全局连接同步到OPENAI_API配置（仅openai类型）"""
    connections = request.app.state.config.GLOBAL_CONNECTIONS or []

    openai_urls = []
    openai_keys = []
    openai_configs = {}

    for conn in connections:
        if conn.get("type") == "openai":
            openai_urls.append(conn.get("url", ""))
            openai_keys.append(conn.get("api_key", ""))
            idx = len(openai_urls) - 1
            openai_configs[str(idx)] = conn.get("config") or {}

    request.app.state.config.OPENAI_API_BASE_URLS = openai_urls
    request.app.state.config.OPENAI_API_KEYS = openai_keys
    request.app.state.config.OPENAI_API_CONFIGS = openai_configs
