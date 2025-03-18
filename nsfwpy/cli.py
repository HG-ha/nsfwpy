#!/usr/bin/env python3
"""
NSFWpy 命令行工具 - 检测图像和视频中的NSFW内容
"""

import os
import sys
import argparse
import json
from .nsfw import NSFWDetectorONNX

def main():
    """命令行工具主函数"""
    parser = argparse.ArgumentParser(description="NSFWpy - 检测图像和视频中的NSFW内容")
    parser.add_argument("--input", required=True, help="要分析的图像或视频文件路径")
    parser.add_argument("-t", "--type", choices=["d", "m2", "i3"], default="d",
                        help="模型类型: d (默认), m2, i3")
    parser.add_argument("-m", "--model", help="自定义模型路径")
    parser.add_argument("-s", "--sample-rate", type=float, default=0.1,
                        help="视频采样率 (0-1), 默认0.1")
    parser.add_argument("-f", "--max-frames", type=int, default=100,
                        help="最大处理帧数，默认100")
    
    args = parser.parse_args()
    
    # 检查文件存在
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    try:
        # 初始化检测器
        detector = NSFWDetectorONNX(model_path=args.model, model_type=args.type)
        
        # 根据文件扩展名决定处理方式
        file_ext = os.path.splitext(args.input)[1].lower()
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        
        if file_ext in video_exts:
            # 处理视频
            result = detector.predict_video(args.input, args.sample_rate, args.max_frames)
        else:
            # 处理图像
            result = detector.predict_image(args.input)
        
        if not result:
            print("处理失败")
            sys.exit(1)
            
        # 输出
        print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()