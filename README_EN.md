# nsfwpy

English | [简体中文](README.md)

# nsfwpy
A lightweight Python library for image content analysis using deep learning models to identify potentially inappropriate content.

## Features

- Lightweight implementation with minimal dependencies
- Support for various image formats (almost all common formats)
- Command-line tool, Python API, and HTTP API interfaces
- Support for Windows and other operating systems
- Automatic model download and caching
- Pre-compiled versions available

## Installation

- Via pip

    ```bash
    pip install nsfwpy
    ```

- From Source

    ```bash
    git clone https://github.com/HG-ha/nsfwpy.git
    cd nsfwpy
    pip install -e .
    ```

### Using Pre-compiled Versions
- Download the pre-compiled version for your platform from [Release](https://github.com/HG-ha/nsfwpy/releases).
- Windows: Enter `nsfwpy.exe` in cmd
- Linux: `chmod +x nsfwpy && ./nsfwpy`

### Building for Other Platforms
- Refer to `build.bat | build.sh`

## Usage

- Python API

    ```python
    from nsfwpy import NSFW

    # Initialize detector (model will be automatically downloaded on first run)
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

    # Specify image dimension (usually not needed as current model only supports 224)
    nsfwpy --dim 299 --input path/to/image.jpg
    ```

### Web API Service (Fully compatible with nsfwjs-api)

- Start the API server:

    ```bash
    # Basic usage
    nsfwpy -w

    # Specify host and port
    nsfwpy -w --host 127.0.0.1 --port 8080

    # Specify custom model
    nsfwpy -w --model path/to/model.onnx
    ```

- API Endpoints:
    - `POST /classify`: Analyze single image
    - `POST /classify-many`: Batch analyze multiple images

- API Documentation:
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

Returns a dictionary containing probability values for each category:
```python
{
    "drawings": 0.1,    # Drawings/Animation
    "hentai": 0.0,     # Anime adult content
    "neutral": 0.8,    # Safe/Neutral content
    "porn": 0.0,       # Adult content
    "sexy": 0.1        # Suggestive content
}
```

## Acknowledgments

The model used in this project is based on [nsfw_model](https://github.com/GantMan/nsfw_model). Thanks to the original authors for their contribution.

### Recommended Resources
1. Sirius Framework: <https://www.siriusbot.cn/>
2. Mirror Core API: <https://api2.wer.plus/>
3. LinFengCloud - Best Choice for Webmasters: <https://www.dkdun.cn/>
4. ICP record inquiry: <https://icp.show/>