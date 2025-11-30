#!/bin/bash
# ============================================
# nsfwpy 构建脚本 (Linux/Mac)
# 支持: Nuitka 二进制构建 和 Docker 多架构镜像构建
# ============================================

set -e

show_help() {
    cat << EOF
用法: ./build.sh [命令]

命令:
  docker    - 构建 Docker 多架构镜像 (linux/amd64, linux/arm64)
  nuitka    - 使用 Nuitka 构建单文件可执行程序

示例:
  ./build.sh docker
  ./build.sh nuitka
EOF
}

docker_build() {
    echo "============================================"
    echo "构建 Docker 多架构镜像"
    echo "============================================"
    echo ""

    # 设置镜像信息
    IMAGE_NAME="nsfwpy"
    IMAGE_TAG="latest"

    # 检查 Docker buildx
    if ! command -v docker &> /dev/null; then
        echo "[错误] Docker 未安装"
        exit 1
    fi

    # 创建或使用 buildx builder
    echo "[1/3] 设置 buildx builder..."
    docker buildx create --name nsfwpy-builder --use --driver docker-container 2>/dev/null || true
    docker buildx inspect --bootstrap

    # 构建多架构镜像
    echo ""
    echo "[2/3] 构建多架构镜像..."
    echo "平台: linux/amd64, linux/arm64"
    echo "镜像: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""

    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --progress=plain \
        --load \
        .

    echo ""
    echo "[3/3] 构建完成！"
    echo ""
    echo "运行镜像:"
    echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "推送到仓库:"
    echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
    echo "  docker push your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
}

nuitka_build() {
    echo "============================================"
    echo "使用 Nuitka 构建单文件可执行程序"
    echo "============================================"
    echo ""

    # 使用 uv 安装依赖和构建工具
    # apt/dnf/yum install patchelf
    # curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv pip install --system -r pyproject.toml
    # uv pip install --system nuitka ordered-set zstandard

    uv run python -m nuitka --standalone \
    --onefile \
    --output-dir=dist \
    --output-filename=nsfwpy \
    --include-package=numpy \
    --include-package=PIL \
    --include-package=fastapi \
    --include-package=uvicorn \
    --include-package=onnxruntime \
    --include-package=multipart \
    --include-package=cv2 \
    --include-package=nsfwpy \
    --include-module=nsfwpy.api \
    --include-module=nsfwpy.server \
    --include-module=nsfwpy.nsfw \
    --include-module=nsfwpy.cli \
    --nofollow-import-to=tkinter,matplotlib,scipy,pandas,tests,distutils,setuptools,pip,wheel,sphinx,pytest \
    --assume-yes-for-downloads \
    --plugin-enable=numpy \
    --enable-plugin=anti-bloat \
    --python-flag=no_site \
    --python-flag=no_warnings \
    --noinclude-custom-mode=sklearn:error \
    --noinclude-custom-mode=tensorflow:error \
    --noinclude-custom-mode=torch:error \
    --lto=yes \
    --remove-output \
    --noinclude-data-files="*.py;*.c;*.h;*.txt;*.md;*.rst;*.css;*.html;*.js;*.git*;*.pkl" \
    --follow-imports \
    --show-progress \
    --show-memory \
    nsfwpy/server.py

    echo ""
    echo "============================================"
    echo "构建完成！"
    echo "可执行文件位于: dist/nsfwpy"
    echo "============================================"
}

# 主函数
case "${1:-}" in
    docker)
        docker_build
        ;;
    nuitka)
        nuitka_build
        ;;
    *)
        show_help
        ;;
esac
