# nsfwpy

[English](README_EN.md) | [简体中文](README.md)

# nsfwpy
一个轻量级Python库，使用深度学习模型进行图像内容分析，可以识别图像是否包含不适宜内容。支持常见图片格式以及GIF，支持常见视频格式。

## 特性

- 轻量级实现，依赖少，易于部署
- 支持多种图像格式输入（几乎所有常见格式）
- 支持GIF、视频输入
- 提供命令行工具、Python API和HTTP API接口
- 支持Windows和其他操作系统
- 自动下载和缓存模型文件
- 提供预编译版本
- **支持硬件加速**（CUDA、TensorRT、DirectML、CoreML、OpenVINO）
- **智能检测中国境内用户并自动使用镜像加速下载**

## 安装

### 📊 快速选择安装方式

| 使用场景 | 推荐方式 | 全局可用 | 需要Python |
|---------|---------|---------|-----------|
| **开发/测试** | pip 安装 | ❌ | ✅ |
| **个人日常使用** | pipx 安装 | ✅ | ✅ |
| **服务器部署** | Docker 或预编译版 | ✅ | ❌ |
| **无 Python 环境** | 预编译版本 | ✅ | ❌ |
| **多项目开发** | 虚拟环境 + pip | ❌ | ✅ |

### 方式1: pip 安装（推荐，适合开发使用）

```bash
pip install nsfwpy
```

**注意：** 这种方式安装后，`nsfwpy` 命令只在当前 Python 环境中可用。如果使用虚拟环境，需要先激活环境才能使用命令。

### 方式2: pipx 安装（推荐，全局使用）

pipx 会为 nsfwpy 创建独立的虚拟环境，同时让命令全局可用，是最佳的全局安装方案：

```bash
# 安装 pipx（如果还没有）
pip install pipx
pipx ensurepath

# 使用 pipx 安装 nsfwpy
pipx install nsfwpy

# 现在可以在任何地方使用 nsfwpy 命令
nsfwpy --help
```

**优点：**
- ✅ 全局可用，无需激活环境
- ✅ 依赖隔离，不影响其他项目
- ✅ 易于管理和升级

### 方式3: 系统级全局安装（不推荐）

```bash
# 直接安装到系统 Python（可能需要 sudo）
pip install --user nsfwpy  # 用户级安装
# 或
sudo pip install nsfwpy    # 系统级安装（不推荐）
```

**警告：** 可能与系统包管理器冲突，建议使用 pipx 代替。

### 方式4: 从源码安装

```bash
git clone https://github.com/HG-ha/nsfwpy.git
cd nsfwpy
pip install -e .
```

### 方式5: 预编译版本（推荐，无需 Python 环境）

预编译版本无需安装 Python，开箱即用，且可全局使用：

1. **下载**：前往 [Release](https://github.com/HG-ha/nsfwpy/releases) 下载对应平台的版本

2. **安装为全局命令**：

   **Linux/macOS:**
   ```bash
   # 下载后
   chmod +x nsfwpy-*-linux-x86_64
   sudo mv nsfwpy-*-linux-x86_64 /usr/local/bin/nsfwpy
   
   # 现在可以在任何地方使用
   nsfwpy --help
   ```

   **Windows:**
   ```powershell
   # 方法1: 添加到 PATH（推荐）
   # 1. 将 nsfwpy.exe 放到一个目录，如 C:\Program Files\nsfwpy\
   # 2. 将该目录添加到系统 PATH 环境变量
   # 3. 重启命令行，即可在任何地方使用 nsfwpy 命令

   # 方法2: 直接使用完整路径
   C:\path\to\nsfwpy.exe --help
   ```

**优点：**
- ✅ 无需安装 Python
- ✅ 无依赖冲突
- ✅ 下载即用
- ✅ 可全局使用

### 方式6: Docker（推荐，服务器部署）

```bash
# 运行 API 服务
docker run -p 8000:8000 yiminger/nsfwpy

# 使用指定模型
docker run -e NSFWPY_MODEL_TYPE=m2 -p 8000:8000 yiminger/nsfwpy
```

### 方式7: Termux (Android)

```bash
pkg install -y build-essential cmake ninja patchelf python3 git python-pip python-onnxruntime python-pillow rust
git clone https://github.com/HG-ha/nsfwpy.git && cd nsfwpy
pip install -e .
nsfwpy --help
```

### ✅ 验证安装

安装完成后，验证是否成功：

```bash
# 查看版本
nsfwpy --help

# 测试图片检测
nsfwpy --input test.jpg

# 启动 Web 服务测试
nsfwpy --web --port 8000
```

### 🔧 编译其他平台版本

参考 `build.bat | build.sh` 脚本自行编译


## 使用方法

### 硬件加速支持

**⚠️ 注意：** 默认安装只包含 CPU 支持。如需使用 GPU 等硬件加速，需要安装对应的 onnxruntime 版本：

```bash
# NVIDIA GPU (CUDA)
pip uninstall onnxruntime
pip install onnxruntime-gpu

# DirectML (Windows GPU)
pip uninstall onnxruntime
pip install onnxruntime-directml

# 其他加速后端请参考 ONNX Runtime 官方文档
```

nsfwpy 支持多种硬件加速选项以提升推理性能：

```python
from nsfwpy import NSFW

# 自动选择最佳可用设备（推荐）
detector = NSFW(device='auto')

# 使用特定设备
detector_cuda = NSFW(device='cuda')      # NVIDIA GPU (CUDA)
detector_tensorrt = NSFW(device='tensorrt')  # NVIDIA GPU (TensorRT)
detector_dml = NSFW(device='dml')        # Windows DirectML
detector_coreml = NSFW(device='coreml')  # Apple CoreML (macOS/iOS)
detector_openvino = NSFW(device='openvino')  # Intel OpenVINO
detector_cpu = NSFW(device='cpu')        # CPU only
```

**支持的加速后端：**
- `auto`: 自动选择最佳设备（默认）
- `cuda`: NVIDIA CUDA（需要安装 onnxruntime-gpu）
- `tensorrt`: NVIDIA TensorRT（需要安装 onnxruntime-gpu）
- `dml`: DirectML - Windows GPU（需要安装 onnxruntime-directml）
- `coreml`: Apple CoreML (macOS/iOS)
- `openvino`: Intel OpenVINO
- `cpu`: CPU（无需额外安装）

### 环境变量配置

```bash
# 模型路径配置
NSFWPY_ONNX_MODEL=/path/to/model.onnx  # 自定义模型路径
NSFWPY_MODEL_TYPE=d                     # 模型类型：d(默认)/m2/i3
NSFW_ONNX_MODEL=/path/to/model.onnx    # 备用环境变量

# 中国境内镜像加速（自动检测，也可手动配置）
NSFWPY_USE_CHINA_MIRROR=1              # 强制使用国内镜像（1/true/yes）
NSFWPY_GITHUB_MIRROR=https://ghproxy.cn  # 自定义镜像地址

# 内存管理配置
NSFWPY_CLEANUP_INTERVAL=100            # 自动垃圾回收间隔（推理次数），默认100，设为0禁用
NSFWPY_GPU_MEM_LIMIT=524288000         # GPU显存限制（字节），默认500MB
NSFWPY_INTRA_THREADS=1                 # ONNX Runtime内部并行线程数，默认自动检测（单核=1，多核=核心数/2，最多4）
NSFWPY_INTER_THREADS=1                 # ONNX Runtime跨操作并行线程数，默认自动检测（单核=1，多核=核心数/4，最多2）
```

### Python API

```python
from nsfwpy import NSFW

# 初始化检测器（首次运行会自动下载模型）
detector = NSFW()

# 预测单个图像
result = detector.predict_image("path/to/image.jpg")
print(result)

    # 预测PIL图像
    from PIL import Image
    img = Image.open("path/to/image.jpg")
    result = detector.predict_pil_image(img)
    print(result)

    # 批量预测目录中的图像
    results = detector.predict_batch("path/to/image/directory")
    print(results)

    # 预测视频文件
    result = detector.predict_video(
        "path/to/video.mp4",
        sample_rate=0.1,  # 采样率，表示每10帧取1帧
        max_frames=100    # 最大处理帧数
    )
    print(result)
    ```

- 命令行工具

    ```bash
    # 基本用法
    nsfwpy --input path/to/image.jpg

    # 指定自定义模型路径
    nsfwpy --model path/to/model.onnx --input path/to/image.jpg

    # 指定模型类型 (d: 默认模型, m2: mobilenet_v2, i3: inception_v3)
    nsfwpy --type m2 --input path/to/image.jpg

    # 启动Web API服务
    nsfwpy -w [--host 127.0.0.1] [--port 8080]
    ```

命令行参数说明：
- `--input`: 要检测的图像或视频文件路径
- `--model`: 自定义模型文件路径（指定此参数时将忽略--type）
- `--type`: 模型类型选择，可选值：d(默认), m2, i3
- `-w, --web`: 启用Web API服务
- `--host`: API服务器主机名（默认：0.0.0.0）
- `--port`: API服务器端口（默认：8000）
- `-s, --sample-rate`: 视频采样率，范围0-1（默认：0.1）
- `-f, --max-frames`: 视频最大处理帧数（默认：100）

### Web API服务（完全兼容 nsfwjs-api）

- 启动API服务器：

    ```bash
    # 基本用法
    nsfwpy -w

    # 指定主机和端口
    nsfwpy -w --host 127.0.0.1 --port 8080

    # 指定自定义模型
    nsfwpy -w --model path/to/model.onnx
    ```

- API端点：
    - `POST /classify`: 分析单张图片（支持图片和GIF）
    - `POST /classify-many`: 批量分析多张图片
    - `POST /classify-video`: 分析视频文件

- API文档：
    - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

- 请求：
    - /classify
        ```
        curl --location --request POST 'http://127.0.0.1:8000/classify' \
        --form 'image=@"image.jpeg"'
        ```
    - /classify-many
        ```
        curl --location --request POST 'http://127.0.0.1:8000/classify-many' \
        --form 'images=@"image.jpeg"' \
        --form 'images=@"image2.jpeg"'
        ```
    - /classify-video
        ```
        curl --location --request POST 'http://127.0.0.1:8000/classify-video' \
        --form 'video=@"video.mp4"' \
        --form 'sample_rate=0.1' \
        --form 'max_frames=100'
        ```

### 预测结果格式

返回包含以下类别概率值的字典：
```python
{
    "drawing": 0.1,    # 绘画/动画
    "hentai": 0.0,     # 动漫色情内容（変態）
    "neutral": 0.8,    # 中性/安全内容
    "porn": 0.0,       # 色情内容
    "sexy": 0.1        # 性感内容
}
```

## 致谢

本项目的模型基于 [nsfw_model](https://github.com/GantMan/nsfw_model) 以及 [nsfwjs](https://github.com/infinitered/nsfwjs)。感谢原作者的贡献。

### 推荐资源
1.  天狼星框架：<https://www.siriusbot.cn/>
2.  镜芯API：<https://api2.wer.plus/>
3.  林枫云_站长首选云服务器：<https://www.dkdun.cn/>
4.  ICP备案查询：<https://icp.show/>