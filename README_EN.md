English | [简体中文](README.md)

# nsfwpy
A lightweight Python library that utilizes deep learning models for image content analysis, capable of identifying whether images contain inappropriate content. It supports common image formats as well as GIFs, and also supports common video formats.

## Features

- Lightweight implementation with minimal dependencies, easy to deploy
- Support for multiple image formats (almost all common formats)
- Support for GIFs
- Support for common video formats
- Provides Command Line Tool, Python API, and HTTP API interfaces
- Supports Windows and other operating systems
- Automatic model download and caching
- Pre-compiled versions available
- **Hardware acceleration support** (CUDA, TensorRT, DirectML, CoreML, OpenVINO)
- **Smart detection for users in China with automatic mirror acceleration**

## Installation

### 📊 Quick Installation Guide

| Use Case | Recommended Method | Globally Available | Requires Python |
|----------|-------------------|-------------------|-----------------|
| **Development/Testing** | pip install | ❌ | ✅ |
| **Personal Daily Use** | pipx install | ✅ | ✅ |
| **Server Deployment** | Docker or Pre-compiled | ✅ | ❌ |
| **No Python Environment** | Pre-compiled version | ✅ | ❌ |
| **Multi-project Development** | Virtual env + pip | ❌ | ✅ |

### Method 1: pip Install (Recommended for development)

```bash
pip install nsfwpy
```

**Note:** With this method, the `nsfwpy` command is only available in the current Python environment. If using a virtual environment, you need to activate it first.

### Method 2: pipx Install (Recommended for global use)

pipx creates an isolated environment for nsfwpy while making the command globally available - the best solution for global installation:

```bash
# Install pipx (if not already installed)
pip install pipx
pipx ensurepath

# Install nsfwpy with pipx
pipx install nsfwpy

# Now you can use nsfwpy anywhere
nsfwpy --help
```

**Advantages:**
- ✅ Globally available without activating environments
- ✅ Isolated dependencies, no conflicts
- ✅ Easy to manage and upgrade

### Method 3: System-wide Installation (Not recommended)

```bash
# Install directly to system Python (may require sudo)
pip install --user nsfwpy  # User-level installation
# or
sudo pip install nsfwpy    # System-level installation (not recommended)
```

**Warning:** May conflict with system package managers. Use pipx instead.

### Method 4: From Source

```bash
git clone https://github.com/HG-ha/nsfwpy.git
cd nsfwpy
pip install -e .
```

### Method 5: Pre-compiled Version (Recommended, no Python required)

Pre-compiled versions work out of the box without Python installation and can be used globally:

1. **Download**: Visit [Release](https://github.com/HG-ha/nsfwpy/releases) to download the version for your platform

2. **Install as global command**:

   **Linux/macOS:**
   ```bash
   # After downloading
   chmod +x nsfwpy-*-linux-x86_64
   sudo mv nsfwpy-*-linux-x86_64 /usr/local/bin/nsfwpy
   
   # Now you can use it anywhere
   nsfwpy --help
   ```

   **Windows:**
   ```powershell
   # Method 1: Add to PATH (Recommended)
   # 1. Place nsfwpy.exe in a directory, e.g., C:\Program Files\nsfwpy\
   # 2. Add that directory to system PATH environment variable
   # 3. Restart command prompt, then you can use nsfwpy command anywhere

   # Method 2: Use full path directly
   C:\path\to\nsfwpy.exe --help
   ```

**Advantages:**
- ✅ No Python installation required
- ✅ No dependency conflicts
- ✅ Download and use immediately
- ✅ Globally available

### Method 6: Docker (Recommended for server deployment)

```bash
# Run API service
docker run -p 8000:8000 yiminger/nsfwpy

# Use specific model
docker run -e NSFWPY_MODEL_TYPE=m2 -p 8000:8000 yiminger/nsfwpy
```

### Method 7: Termux (Android)

```bash
pkg install -y build-essential cmake ninja patchelf python3 git python-pip python-onnxruntime python-pillow rust
git clone https://github.com/HG-ha/nsfwpy.git && cd nsfwpy
pip install -e .
nsfwpy --help
```

### ✅ Verify Installation

After installation, verify it works:

```bash
# Check version
nsfwpy --help

# Test image detection
nsfwpy --input test.jpg

# Start Web service test
nsfwpy --web --port 8000
```

### 🔧 Building for Other Platforms

Refer to `build.bat | build.sh` scripts for custom compilation

## Usage

### Hardware Acceleration Support

**⚠️ Note:** The default installation only includes CPU support. For GPU or other hardware acceleration, you need to install the corresponding onnxruntime version:

```bash
# NVIDIA GPU (CUDA)
pip uninstall onnxruntime
pip install onnxruntime-gpu

# DirectML (Windows GPU)
pip uninstall onnxruntime
pip install onnxruntime-directml

# For other acceleration backends, refer to the official ONNX Runtime documentation
```

nsfwpy supports multiple hardware acceleration options to improve inference performance:

```python
from nsfwpy import NSFW

# Auto-select the best available device (recommended)
detector = NSFW(device='auto')

# Use specific device
detector_cuda = NSFW(device='cuda')      # NVIDIA GPU (CUDA)
detector_tensorrt = NSFW(device='tensorrt')  # NVIDIA GPU (TensorRT)
detector_dml = NSFW(device='dml')        # Windows DirectML
detector_coreml = NSFW(device='coreml')  # Apple CoreML (macOS/iOS)
detector_openvino = NSFW(device='openvino')  # Intel OpenVINO
detector_cpu = NSFW(device='cpu')        # CPU only
```

**Supported acceleration backends:**
- `auto`: Automatically select the best device (default)
- `cuda`: NVIDIA CUDA (requires onnxruntime-gpu)
- `tensorrt`: NVIDIA TensorRT (requires onnxruntime-gpu)
- `dml`: DirectML - Windows GPU (requires onnxruntime-directml)
- `coreml`: Apple CoreML (macOS/iOS)
- `openvino`: Intel OpenVINO
- `cpu`: CPU (no additional installation required)

### Environment Variables

```bash
# Model path configuration
NSFWPY_ONNX_MODEL=/path/to/model.onnx  # Custom model path
NSFWPY_MODEL_TYPE=d                     # Model type: d(default)/m2/i3
NSFW_ONNX_MODEL=/path/to/model.onnx    # Alternative environment variable

# China mirror acceleration (auto-detected, can be manually configured)
NSFWPY_USE_CHINA_MIRROR=1              # Force use of China mirror (1/true/yes)
NSFWPY_GITHUB_MIRROR=https://ghproxy.cn  # Custom mirror address

# Memory management configuration
NSFWPY_CLEANUP_INTERVAL=100            # Auto garbage collection interval (inference count), default 100, set to 0 to disable
NSFWPY_GPU_MEM_LIMIT=524288000         # GPU memory limit (bytes), default 500MB
NSFWPY_INTRA_THREADS=1                 # ONNX Runtime intra-op parallel threads, auto-detected (single-core=1, multi-core=cores/2, max 4)
NSFWPY_INTER_THREADS=1                 # ONNX Runtime inter-op parallel threads, auto-detected (single-core=1, multi-core=cores/4, max 2)
```

### Python API

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