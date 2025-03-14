# nsfwpy

[English](README_EN.md) | [简体中文](README.md)

# nsfwpy
一个轻量级Python库，使用深度学习模型进行图像内容分析，可以识别图像是否包含不适宜内容。

## 特性

- 轻量级实现，依赖少，易于部署
- 支持多种图像格式输入（几乎所有常见格式）
- 提供命令行工具、Python API和HTTP API接口
- 支持Windows和其他操作系统
- 自动下载和缓存模型文件
- 提供预编译版本

## 安装

- 通过pip安装

    ```bash
    pip install nsfwpy
    ```

- 从源码安装

    ```bash
    git clone https://github.com/HG-ha/nsfwpy.git
    cd nsfwpy
    pip install -e .
    ```
    
- Docker（默认使用模型：model.onnx）
    - `docker run -p 8000:8000 yiminger/nsfwpy`
    - 使用指定模型启动
        - 默认模型
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/model.onnx -p 8000:8000 nsfwpy
            ```
        - m2模型（NSFWJS mobilenet_v2）
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/m2model.onnx -p 8000:8000 nsfwpy
            ```
        - i3模型（NSFWJS inception_v3），速度是其他模型的一半
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/i3model.onnx -p 8000:8000 nsfwpy
            ```

- 使用预编译版本（开箱即用）
    - 请前往 [Release](https://github.com/HG-ha/nsfwpy/releases) 下载对应平台的预编译版本。
    - windows：在cmd中输入 `nsfwpy.exe`
    - linux：`chmod +x nsfwpy && ./nsfwpy`

- Termux
    ```bash
    pkg install python3 git python-pip python-onnxruntime rust -y
    git clone https://github.com/HG-ha/nsfwpy.git && cd nsfwpy
    pip install -e .
    nsfwpy --help
    ```

### 编译其他平台版本
- 参考 `build.bat | build.sh`


## 使用方法

- Python API

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

- 命令行工具

    ```bash
    # 基本用法
    nsfwpy --input path/to/image.jpg

    # 指定自定义模型路径
    nsfwpy --model path/to/model.onnx --input path/to/image.jpg

    # 指定模型需要的图像尺寸（通常用不到，因为目前模型只支持224，无需关注此参数）
    nsfwpy --dim 299 --input path/to/image.jpg
    ```

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
    - `POST /classify`: 分析单张图片
    - `POST /classify-many`: 批量分析多张图片

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
    

### 预测结果格式

返回包含以下类别概率值的字典：
```python
{
    "drawings": 0.1,    # 绘画/动画
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