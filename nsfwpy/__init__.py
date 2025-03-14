from .nsfw import NSFWDetectorONNX

# 为简化引用提供一个别名
NSFW = NSFWDetectorONNX

# 导出用于命令行工具的主入口点
from .server import main as cli_main

__version__ = "0.1.3"
