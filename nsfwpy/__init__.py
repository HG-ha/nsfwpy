__version__ = "0.1.4.3"

from .nsfw import NSFWDetectorONNX

# 为简化引用提供一个别名
NSFW = NSFWDetectorONNX

# 避免循环导入
def get_cli_main():
    from .server import main
    return main

cli_main = get_cli_main()