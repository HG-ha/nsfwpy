# nsfwpy

English | [简体中文](README.md)

A Python tool for NSFW (Not Safe For Work) content detection, providing an easy-to-use Python interface for image content analysis and filtering, with both API and CLI tools.

## Introduction

nsfwpy is a lightweight Python library that uses deep learning models for image content analysis to identify potentially inappropriate content. This project is based on the model provided by [nsfw_model](https://github.com/GantMan/nsfw_model).

## Features

- Lightweight implementation with minimal dependencies
- Support for various image formats (JPG, PNG, etc.)
- Command-line tool, Python API, and HTTP API interfaces
- Optimized performance using TensorFlow Lite
- Cross-platform support (Windows and other operating systems)
- Automatic model download and caching

## Requirements
> Method compatibility is typically stable across major versions
- Python 3.7+
- NumPy <= 1.26.4
- Pillow <= 11.1.0
- FastAPI <= 0.115.11
- uvicorn <= 0.34.0
- python-multipart <= 0.0.20
- tflite-runtime = 2.13.0 (Windows) or >= 2.5.0 (other systems)

## Installation

### Via pip

```bash
pip install nsfwpy
```

### From Source

```bash
git clone https://github.com/HG-ha/nsfwpy.git
cd nsfwpy
pip install -e .
```

## Usage

### Python API

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

### Command Line Tool

```bash
# Basic usage
nsfwpy --input path/to/image.jpg

# Specify custom model path
nsfwpy --model path/to/model.tflite --input path/to/image.jpg

# Specify image dimension
nsfwpy --dim 299 --input path/to/image.jpg
```

### Web API Service (Fully compatible with nsfwjs-api)

Start the API server:

```bash
# Basic usage
nsfwpy -w

# Specify host and port
nsfwpy -w --host 127.0.0.1 --port 8080

# Specify custom model
nsfwpy -w --model path/to/model.tflite
```

API Endpoints:
- `POST /classify`: Analyze single image
- `POST /classify-many`: Batch analyze multiple images

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

## Development

- Licensed under MIT
- Issues and Pull Requests are welcome
- Automatic PyPI deployment via GitHub Actions

## Acknowledgments

The model used in this project is based on [nsfw_model](https://github.com/GantMan/nsfw_model). Thanks to the original authors for their contribution.

## License

[MIT License](LICENSE)
