# src/ux_speech_gateway/core/middleware.py
import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from .logging import setup_logging


setup_logging()


class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())  # Generate a unique request ID
        start_time = time.time()
        
        response = await call_next(request)

        # Log request and response details (including execution time)
        exec_time = time.time() - start_time
        logging.info(
            f"Request ID: {request_id}, Method: {request.method}, Path: {request.url.path}, Time: {exec_time:.4f}s"
        )
        
        # Attach request_id to response headers
        response.headers['X-Request-ID'] = request_id

        return response
