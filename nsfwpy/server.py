import argparse
import os
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="启动NSFW检测API服务器或命令行工具")
    parser.add_argument("--host", default="0.0.0.0", help="API服务器主机名")
    parser.add_argument("--port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--model", help="模型文件路径（指定此参数时将忽略--type）")
    parser.add_argument("--type", choices=['d', 'm2', 'i3'],
                       default=os.environ.get('NSFWPY_MODEL_TYPE', 'd'),
                       help='模型类型：d(默认), m2, i3。注意：当指定--model时此参数无效')
    parser.add_argument("-w", "--web", action="store_true", help="启用Web API服务")
    parser.add_argument("--input", help="要检测的图像或视频文件路径")
    # 添加视频处理相关参数
    parser.add_argument("-s", "--sample-rate", type=float, default=0.1,
                      help="视频采样率 (0-1), 默认0.1")
    parser.add_argument("-f", "--max-frames", type=int, default=100,
                      help="视频最大处理帧数，默认100")
    
    args, unknown_args = parser.parse_known_args()
    
    # 如果指定了模型路径，设置环境变量
    if args.model:
        os.environ["NSFWPY_ONNX_MODEL"] = str(Path(args.model).absolute())
    
    # 设置模型类型环境变量
    if args.type:
        os.environ["NSFWPY_MODEL_TYPE"] = args.type

    # 只在指定--web参数时启动API服务器
    if args.web:
        # 启动服务器
        uvicorn.run("nsfwpy.api:app", host=args.host, port=args.port)
    else:
        # 运行命令行版本
        from nsfwpy.cli import main as cli_main
        import sys
        
        # 重建参数，传递给cli模块
        cli_args = []
        if args.model:
            cli_args.extend(["--model", args.model])
        elif args.type:  # 只有在未指定model时才传递type参数
            cli_args.extend(["--type", args.type])
        if args.input:
            cli_args.extend(["--input", args.input])
            
        # 添加视频处理相关参数
        if args.sample_rate is not None:
            cli_args.extend(["--sample-rate", str(args.sample_rate)])
        if args.max_frames is not None:
            cli_args.extend(["--max-frames", str(args.max_frames)])
        
        # 添加未知参数
        sys.argv[1:] = cli_args + unknown_args
        cli_main()

if __name__ == "__main__":
    main()
