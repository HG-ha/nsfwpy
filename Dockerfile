# 构建阶段
FROM python:3.10-slim AS builder

WORKDIR /app

# 设置pip不使用缓存并禁用版本检查，减少镜像大小
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 安装编译工具和依赖（用于编译 numpy 等包）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装依赖（先复制依赖文件，利用Docker缓存）
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 复制项目源代码
COPY README.md setup.py ./
COPY nsfwpy/ ./nsfwpy/

# 运行阶段
FROM python:3.10-slim

WORKDIR /app

# 设置环境变量
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_TYPE=d \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 复制Python包和依赖
COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
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