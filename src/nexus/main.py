from __future__ import annotations

import uvicorn

from .app import create_app

app = create_app()


def run():
    """
    Entry point for Poetry script or direct execution.
    """
    uvicorn.run(
        "ux_speech_gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    run()
