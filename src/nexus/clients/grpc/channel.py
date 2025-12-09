from __future__ import annotations

import grpc


def make_target(host: str, port: int) -> str:
    return f"{host}:{port}"


def create_insecure_channel(host: str, port: int) -> grpc.Channel:
    """
    Create an insecure gRPC channel.

    You can add channel options here for keepalive/retries, etc.
    """
    target = make_target(host, port)

    options = [
        # 你可以按需开启/调整
        ("grpc.keepalive_time_ms", 30000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 1),
    ]

    return grpc.insecure_channel(target, options=options)
