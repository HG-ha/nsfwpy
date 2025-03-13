# pip install nuitka
# apt\dnf\yum install patchelf
# pip install -r requirements.txt

python -m nuitka --standalone \
    --onefile \
    --output-dir=dist \
    --output-filename=nsfwpy \
    --include-package=numpy \
    --include-package=PIL \
    --include-package=fastapi \
    --include-package=uvicorn \
    --include-package=onnxruntime \
    --include-package=multipart \
    --include-package=nsfwpy \
    --include-module=nsfwpy.api \
    --include-module=nsfwpy.server \
    --include-module=nsfwpy.nsfw \
    --include-module=nsfwpy.cli \
    --nofollow-import-to=tkinter,matplotlib,scipy,pandas,cv2 \
    --assume-yes-for-downloads \
    --plugin-enable=numpy \
    --enable-plugin=anti-bloat \
    --follow-imports \
    --show-progress \
    --show-memory \
    nsfwpy/server.py