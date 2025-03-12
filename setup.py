import setuptools
from nsfwpy import __version__

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

install_requires = [
    "numpy<=1.26.4",
    "pillow<=11.1.0",
    "fastapi<=0.115.11",
    "uvicorn<=0.34.0",
    "python-multipart<=0.0.20",
    "onnxruntime<=1.21.0"
]

setuptools.setup(
    name="nsfwpy",
    version=__version__,
    author="YiMing",
    author_email="1790233968@qq.com",
    description="基于OpenNSFW的图像内容检测工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HG-ha/nsfwpy",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'nsfwpy=nsfwpy.server:main',
        ],
    },
)