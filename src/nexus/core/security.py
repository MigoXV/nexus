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
