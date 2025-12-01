# 构建阶段
FROM python:3.10-slim AS builder

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 安装依赖（先复制依赖文件，利用Docker缓存）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 复制项目源代码
COPY README.md setup.py ./
COPY nsfwpy/ ./nsfwpy/

# 安装项目
RUN uv sync --frozen --no-dev

# 运行阶段
FROM python:3.10-slim

WORKDIR /app

# 设置环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_TYPE=d \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# 复制 uv 安装的虚拟环境和项目代码
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/nsfwpy ./nsfwpy

# 暴露端口
EXPOSE 8000

# 创建目录结构并设置权限
RUN mkdir -p /app/temp && \
    useradd -m appuser && \
    mkdir -p /home/appuser/.cache/nsfwpy && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

# 复制模型文件
COPY model/ /home/appuser/.cache/nsfwpy/

# 切换到非root用户
USER appuser

# 入口点
CMD ["python", "-m", "nsfwpy.server", "-w"]