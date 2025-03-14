# 构建阶段
FROM python:3.10-slim AS builder

WORKDIR /app

# 首先复制setup相关文件
COPY README.md ./
COPY setup.py ./

# 复制项目源代码
COPY nsfwpy/ ./nsfwpy/

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# 运行阶段
FROM python:3.10-slim

WORKDIR /app

# 复制Python包和依赖
COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/nsfwpy ./nsfwpy

# 设置环境变量
ENV HOST=0.0.0.0
ENV PORT=8000
# 使用默认模型
ENV MODEL_TYPE=d

EXPOSE 8000

# 使用非root用户运行
RUN useradd -m appuser
USER appuser

RUN mkdir -p /home/appuser/.cache/nsfwpy && \
    chown -R appuser:appuser /home/appuser/.cache

COPY model/ /home/appuser/.cache/nsfwpy/

CMD ["python", "-m", "nsfwpy.server", "-w"]
