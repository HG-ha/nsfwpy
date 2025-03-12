def main():
    import argparse
    import json
    from nsfwpy.nsfw import NSFWDetectorONNX

    parser = argparse.ArgumentParser(description='NSFW图像内容检测')
    parser.add_argument('--model', help='TFLite模型文件路径（可选）')
    parser.add_argument('--dim', type=int, default=224, help='图像尺寸（默认：224）')
    parser.add_argument('--input', required=True, help='要检测的图像文件或目录')

    args = parser.parse_args()

    # 创建检测器，如果未指定模型路径，则使用默认值
    detector = NSFWDetectorONNX(model_path=args.model, image_dim=args.dim)
    results = detector.predict_batch(args.input)

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()