# nsfwpy

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
    
- Docker
    - `docker run -p 8000:8000 yiminger/nsfwpy`

- Using pre-compiled version (ready to use)
    - Please visit [Release](https://github.com/HG-ha/nsfwpy/releases) to download the pre-compiled version for your platform.
    - Windows: Enter `nsfwpy.exe` in cmd
    - Linux: `chmod +x nsfwpy && ./nsfwpy`

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
    ```

- Command Line Tool

    ```bash
    # Basic usage
    nsfwpy --input path/to/image.jpg

    # Specify custom model path
    nsfwpy --model path/to/model.onnx --input path/to/image.jpg

    # Specify model image dimensions (usually not needed as current model only supports 224)
    nsfwpy --dim 299 --input path/to/image.jpg
    ```

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
    - `POST /classify`: Analyze single image
    - `POST /classify-many`: Batch analyze multiple images

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

### Prediction Result Format

Returns a dictionary with probability values for the following categories:
```python
{
    "drawings": 0.1,    # Drawings/Animation
    "hentai": 0.0,     # Anime pornographic content
    "neutral": 0.8,    # Neutral/Safe content
    "porn": 0.0,       # Pornographic content
    "sexy": 0.1        # Suggestive content
}
```

## Acknowledgements

The model in this project is based on [nsfw_model](https://github.com/GantMan/nsfw_model). Thanks to the original authors for their contribution.

### Recommended Resources
1. Sirius Framework: <https://www.siriusbot.cn/>
2. Mirror Core API: <https://api2.wer.plus/>
3. Linfeng Cloud - Top Choice for Webmasters: <https://www.dkdun.cn/>
4. ICP Registration Query: <https://icp.show/>