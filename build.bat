# uv add nuitka
# uv sync
uv run python -m nuitka --standalone ^
    --onefile ^
    --output-dir=dist ^
    --output-filename=nsfwpy ^
    --include-package=numpy ^
    --include-package=PIL ^
    --include-package=fastapi ^
    --include-package=uvicorn ^
    --include-package=onnxruntime ^
    --include-package=multipart ^
    --include-package=cv2 ^
    --include-package=nsfwpy ^
    --include-module=nsfwpy.api ^
    --include-module=nsfwpy.server ^
    --include-module=nsfwpy.nsfw ^
    --include-module=nsfwpy.cli ^
    --nofollow-import-to=tkinter,matplotlib,scipy,pandas,tests,distutils,setuptools,pip,wheel,sphinx,pytest ^
    --assume-yes-for-downloads ^
    --plugin-enable=numpy ^
    --enable-plugin=anti-bloat ^
    --python-flag=no_site ^
    --python-flag=no_warnings ^
    --noinclude-custom-mode=sklearn:error ^
    --noinclude-custom-mode=tensorflow:error ^
    --noinclude-custom-mode=torch:error ^
    --lto=yes ^
    --remove-output ^
    --noinclude-data-files="*.py;*.c;*.h;*.txt;*.md;*.rst;*.css;*.html;*.js;*.git*;*.pkl" ^
    --follow-imports ^
    --show-progress ^
    --show-memory ^
    nsfwpy\server.py
