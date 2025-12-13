from __future__ import annotations

import uvicorn

from .app import create_app
from .core.config import settings

app = create_app()


def run():
    uvicorn.run(#启动一个asgi服务器，用来跑异步web服务框架，（打开一个服务端口，把app放进去
        "ux_speech_gateway.main:app",#类似于import路径，告诉uvicorn去哪里找app对象，将app对象放入端口运行
        host=settings.http_host,
        port=settings.http_port,
        reload=(settings.env == "dev"),#自动重启服务  区分开发环境和生产环境
    )


if __name__ == "__main__":
    run()
#调用关系：
#main.py → 导入 app → app.py 里组装好全部中间件和路由 → uvicorn 用它来处理 HTTP 请求。

# reload 参数的作用：
# reload=True：
# 监控代码文件变动；
# 一旦你改了 .py 文件，它会自动重启服务；
# 非常适合本地开发调试。
# reload=False：
# 不做文件监控；
# 性能更好，行为更稳定；
# 推荐用于生产环境。