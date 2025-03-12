import argparse
import os
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动NSFW检测API服务器或命令行工具")
    parser.add_argument("--host", default="0.0.0.0", help="API服务器主机名")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--model", help="模型文件路径")
    parser.add_argument("-w", "--web", action="store_true", help="启用Web API服务")
    parser.add_argument("--input", help="要检测的图像文件或目录")
    parser.add_argument("--dim", type=int, default=224, help="图像尺寸（默认：224）")
    
    args, unknown_args = parser.parse_known_args()
    
    # 如果指定了模型路径，设置环境变量
    if args.model:
        os.environ["NSFW_MODEL_PATH"] = str(Path(args.model).absolute())
    
    # 只在指定--web参数时启动API服务器
    if args.web:
        # 启动服务器
        # 移除上传文件大小限制
        from starlette.formparsers import MultiPartParser
        MultiPartParser.max_part_size = 1024 * 1024 * 1024
        MultiPartParser.max_file_size = 1024 * 1024 * 1024
        uvicorn.run("nsfwpy.api:app", host=args.host, port=args.port)
    else:
        # 运行命令行版本
        from nsfwpy.cli import main as cli_main
        import sys
        
        # 重建参数，传递给cli模块
        cli_args = ["--dim", str(args.dim)]
        if args.model:
            cli_args.extend(["--model", args.model])
        if args.input:
            cli_args.extend(["--input", args.input])
        
        # 添加未知参数
        sys.argv[1:] = cli_args + unknown_args
        cli_main()

if __name__ == "__main__":
    main()
