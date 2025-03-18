import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

install_requires = [
    "onnxruntime<=1.21.0",
    "pillow<=11.1.0",
    "fastapi<=0.115.11",
    "uvicorn<=0.34.0",
    "python-multipart<=0.0.20",
    "numpy<=1.26.4",
]

def is_opencv_installed():
    try:
        import cv2
        return True
    except ImportError:
        return False
    
if not is_opencv_installed():
    install_requires.append("opencv-python-headless<=4.11.0.86")

setuptools.setup(
    name="nsfwpy",
    version="0.1.4.2",
    author="YiMing",
    author_email="1790233968@qq.com",
    description="基于OpenNSFW的图像内容检测工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HG-ha/nsfwpy",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    entry_points={
        'console_scripts': [
            'nsfwpy=nsfwpy.server:main',
        ],
    }
)