def main():
    import argparse
    import json
    from nsfwpy.nsfw import NSFWDetectorONNX

    parser = argparse.ArgumentParser(description='NSFW图像内容检测')
    parser.add_argument('--model', help='模型文件路径（指定此参数时将忽略--type）')
    parser.add_argument('--type', choices=['d', 'm2', 'i3'], default='d', 
                       help='模型类型：d(默认), m2, i3。注意：当指定--model时此参数无效')
    parser.add_argument('--input', required=True, help='要检测的图像文件或目录')

    args = parser.parse_args()

    # 创建检测器，如果未指定模型路径，则使用默认值
    detector = NSFWDetectorONNX(model_path=args.model, model_type=args.type)
    results = detector.predict_batch(args.input)

    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()