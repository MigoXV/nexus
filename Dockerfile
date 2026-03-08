# 基于已经包含 Poetry 的基础镜像
FROM registry.cn-hangzhou.aliyuncs.com/migo-dl/python:3.10.18-poetry-0-4-1

# 设置工作目录
WORKDIR /app

# 拷贝必要的文件以安装依赖
COPY wheels/ ./wheels/
COPY pyproject.toml poetry.lock README.md ./

# 安装依赖
RUN mkdir -p src/nexus && \
    touch src/nexus/__init__.py && \
    poetry install --no-root
RUN poetry run pip install ./wheels/samplerate-0.2.3-cp310-cp310-linux_x86_64.whl
# 拷贝 pyproject.toml 和 poetry.lock 文件
COPY . .

# 安装依赖
RUN poetry install

# 暴露 gRPC 服务端口
EXPOSE 8000

# 默认入口
CMD ["poetry", "run", "python", "-m", "nexus.commands.app"]

