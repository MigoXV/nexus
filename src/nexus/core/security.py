# src/ux_speech_gateway/core/security.py
from fastapi import Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging


class CustomHTTPBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        # For now, just log the token (in a real-world scenario, verify token here)
        logging.debug(f"Received token: {credentials.credentials}")
        return credentials
    

# 作用：安全相关内容的集中地。

# 可能包含：

# 认证（token 校验、API key 验证）。

# 授权（权限检查）。

# 对 gRPC 调用增加 metadata（如用户 ID、租户 ID）。

# 好处：

# HTTP 路由/Service 只调用 check_auth() / 使用 Depends(current_user)，不用关心细节。