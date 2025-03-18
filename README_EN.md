English | [简体中文](README.md)

# nsfwpy
A lightweight Python library for image content analysis using deep learning models to detect inappropriate content in images.

## Features

- Lightweight implementation with minimal dependencies, easy to deploy
- Support for multiple image formats (almost all common formats)
- Provides Command Line Tool, Python API, and HTTP API interfaces
- Supports Windows and other operating systems
- Automatic model download and caching
- Pre-compiled versions available

## Installation

- Via pip

    ```bash
    pip install nsfwpy
    ```

- From source

    ```bash
    git clone https://github.com/HG-ha/nsfwpy.git
    cd nsfwpy
    pip install -e .
    ```
    
- Docker (Default model: model.onnx)
    - `docker run -p 8000:8000 yiminger/nsfwpy`
    - Start with specific model:
        - `d` Default model
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/model.onnx -p 8000:8000 yiminger/nsfwpy
            ```
        - `m2` model (NSFWJS mobilenet_v2)
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/m2model.onnx -p 8000:8000 yiminger/nsfwpy
            ```
        - `i3` model (NSFWJS inception_v3), It takes twice as long as the others.
            ```
            docker run -e NSFWPY_ONNX_MODEL=/home/appuser/.cache/nsfwpy/i3model.onnx -p 8000:8000 yiminger/nsfwpy
            ```

- Using pre-compiled version (ready to use)
    - Please visit [Release](https://github.com/HG-ha/nsfwpy/releases) to download the pre-compiled version for your platform.
    - Windows: Enter `nsfwpy.exe` in cmd
    - Linux: `chmod +x nsfwpy && ./nsfwpy`

- Termux
    ```bash
    pkg install -y build-essential cmake ninja patchelf python3 git python-pip python-onnxruntime python-pillow rust
    git clone https://github.com/HG-ha/nsfwpy.git && cd nsfwpy
    pip install -e .
    nsfwpy --help
    ```
    
### Building for other platforms
- Refer to `build.bat | build.sh`

## Usage

- Python API

    ```python
    from nsfwpy import NSFW

    # Initialize detector (first run will automatically download the model)
    detector = NSFW()

    # Predict single image
    result = detector.predict_image("path/to/image.jpg")
    print(result)

    # Predict PIL image
    from PIL import Image
    img = Image.open("path/to/image.jpg")
    result = detector.predict_pil_image(img)
    print(result)

    # Batch predict images in directory
    results = detector.predict_batch("path/to/image/directory")
    print(results)
    
    # Predict video file
    result = detector.predict_video(
        "path/to/video.mp4",
        sample_rate=0.1,  # Sampling rate, process 1 frame every 10 frames
        max_frames=100    # Maximum number of frames to process
    )
    print(result)
    ```

- Command Line Tool

    ```bash
    # Basic usage
    nsfwpy --input path/to/image.jpg

    # Specify custom model path
    nsfwpy --model path/to/model.onnx --input path/to/image.jpg

    # Specify model type (d: default model, m2: mobilenet_v2, i3: inception_v3)
    nsfwpy --type m2 --input path/to/image.jpg

    # Start Web API service
    nsfwpy -w [--host 127.0.0.1] [--port 8080]
    ```

Command line arguments:
- `--input`: Path to image/video file or directory to analyze
- `--model`: Custom model file path (--type will be ignored when this is specified)
- `--type`: Model type selection, options: d(default), m2, i3
- `-w, --web`: Enable Web API service
- `--host`: API server hostname (default: 0.0.0.0)
- `--port`: API server port (default: 8000)
- `-s, --sample-rate`: Video sampling rate, range 0-1 (default: 0.1)
- `-f, --max-frames`: Maximum frames to process for video (default: 100)

### Web API Service (Fully compatible with nsfwjs-api)

- Start API server:

    ```bash
    # Basic usage
    nsfwpy -w

    # Specify host and port
    nsfwpy -w --host 127.0.0.1 --port 8080

    # Specify custom model
    nsfwpy -w --model path/to/model.onnx
    ```

- API endpoints:
    - `POST /classify`: Analyze single image (supports images and GIFs)
    - `POST /classify-many`: Batch analyze multiple images
    - `POST /classify-video`: Analyze video file

- API documentation:
    - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

- Requests:
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

### Prediction Result Format

Returns a dictionary with probability values for the following categories:
```python
{
    "drawing": 0.1,    # Drawings/Animation
    "hentai": 0.0,     # Anime pornographic content
    "neutral": 0.8,    # Neutral/Safe content
    "porn": 0.0,       # Pornographic content
    "sexy": 0.1        # Suggestive content
}
```

## Acknowledgements

This project's model is based on [nsfw_model](https://github.com/GantMan/nsfw_model) and [nsfwjs](https://github.com/infinitered/nsfwjs). Thanks to the original authors for their contribution.

### Recommended Resources
1. Sirius Framework: <https://www.siriusbot.cn/>
2. Mirror Core API: <https://api2.wer.plus/>
3. Linfeng Cloud - Top Choice for Webmasters: <https://www.dkdun.cn/>
4. ICP Registration Query: <https://icp.show/>