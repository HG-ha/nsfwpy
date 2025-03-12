# nsfwpy

[English](README_EN.md) | 简体中文

图像敏感内容检测工具，提供简单易用的Python接口进行图像内容分析和过滤，并提供API和CLI工具。

## 简介

nsfwpy 是一个轻量级Python库，使用深度学习模型进行图像内容分析，可以识别图像是否包含不适宜内容。本项目基于[nsfw_model](https://github.com/GantMan/nsfw_model)提供的模型实现。

## 特性

- 轻量级实现，依赖少，易于部署
- 支持多种图像格式输入（JPG、PNG等）
- 提供命令行工具、Python API和HTTP API接口
- 使用TensorFlow Lite优化性能
- 支持Windows和其他操作系统
- 自动下载和缓存模型文件

## 安装要求
> 由依赖导致安装出错时，可以尝试移除版本号安装，但tflite-runtime版本必须>=2.5.0
- Python 3.7+
- NumPy <= 1.26.4
- Pillow <= 11.1.0
- FastAPI <= 0.115.11
- uvicorn <= 0.34.0
- python-multipart <= 0.0.20
- tflite-runtime = 2.13.0 (Windows) 或 >= 2.5.0 (其他系统)

## 安装

### 通过pip安装

```bash
pip install nsfwpy
```

### 从源码安装

```bash
git clone https://github.com/HG-ha/nsfwpy.git
cd nsfwpy
pip install -e .
```

## 使用方法

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
```

### 命令行工具

```bash
# 基本用法
nsfwpy --input path/to/image.jpg

# 指定自定义模型路径
nsfwpy --model path/to/model.tflite --input path/to/image.jpg

# 指定图像尺寸
nsfwpy --dim 299 --input path/to/image.jpg
```

### Web API服务（完全兼容 nsfwjs-api）

启动API服务器：

```bash
# 基本用法
nsfwpy -w

# 指定主机和端口
nsfwpy -w --host 127.0.0.1 --port 8080

# 指定自定义模型
nsfwpy -w --model path/to/model.tflite
```

API端点：
- `POST /classify`: 分析单张图片
- `POST /classify-many`: 批量分析多张图片

### 预测结果格式

返回包含以下类别概率值的字典：
```python
{
    "drawings": 0.1,    # 绘画/动画
    "hentai": 0.0,     # 动漫色情内容
    "neutral": 0.8,    # 中性/安全内容
    "porn": 0.0,       # 色情内容
    "sexy": 0.1        # 性感内容
}
```

## 开发说明

- 项目使用MIT许可证
- 欢迎提交Issue和Pull Request
- 自动发布到PyPI使用GitHub Actions

## 致谢

本项目的模型基于[nsfw_model](https://github.com/GantMan/nsfw_model)。感谢原作者的贡献。

## 许可证

[MIT License](LICENSE)
