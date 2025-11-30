# 构建阶段
FROM python:3.11-slim AS builder

WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# 安装依赖并验证
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system . && \
    python -c "import nsfwpy; print(f'nsfwpy version: {nsfwpy.__version__}')"

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 设置环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    NSFWPY_MODEL_TYPE=d \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# 安装运行时依赖（opencv 需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 复制Python包和依赖
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/nsfwpy ./nsfwpy

# 创建非 root 用户和目录
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/temp /home/appuser/.cache/nsfwpy && \
    chown -R appuser:appuser /app /home/appuser

# 复制模型文件到缓存目录（预先加载模型，避免运行时下载）
COPY --chown=appuser:appuser model/*.onnx /home/appuser/.cache/nsfwpy/

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 入口点
CMD ["python", "-m", "nsfwpy.server", "--web", "--host", "0.0.0.0", "--port", "8000"]