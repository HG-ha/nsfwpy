# 构建阶段
FROM python:3.11-slim AS builder

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 安装编译工具和依赖（用于编译 numpy 等包）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 复制项目文件
COPY pyproject.toml README.md ./
COPY nsfwpy/ ./nsfwpy/

# 安装依赖
RUN uv pip install --system .

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 设置环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_TYPE=d \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制Python包和依赖
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/nsfwpy ./nsfwpy

# 暴露端口
EXPOSE 8000

# 创建目录结构并设置权限
RUN mkdir -p /app/temp && \
    useradd -m appuser && \
    mkdir -p /home/appuser/.cache/nsfwpy && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app/temp

# 复制模型文件
COPY model/ /home/appuser/.cache/nsfwpy/

# 切换到非root用户
USER appuser

# 入口点
CMD ["python", "-m", "nsfwpy.server", "-w"]