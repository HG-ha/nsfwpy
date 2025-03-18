import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import platform
import urllib.request
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2  # 添加OpenCV库

class NSFWDetectorONNX:
    """NSFW内容检测器，基于MobileNet V2模型的ONNX版本"""
    
    CATEGORIES = ['drawing', 'hentai', 'neutral', 'porn', 'sexy']
    
    MODEL_CONFIGS = {
        'd': {
            'url': "https://github.com/HG-ha/nsfwpy/raw/main/model/model.onnx",
            'filename': "model.onnx",
            'dim': 224
        },
        'm2': {
            'url': "https://github.com/HG-ha/nsfwpy/raw/main/model/m2model.onnx",
            'filename': "m2model.onnx",
            'dim': 224
        },
        'i3': {
            'url': "https://github.com/HG-ha/nsfwpy/raw/main/model/i3model.onnx",
            'filename': "i3model.onnx",
            'dim': 299
        }
    }
    
    def __init__(self, model_path=None, model_type='d'):
        """
        初始化NSFW检测器(ONNX版本)
        
        参数:
            model_path: ONNX模型文件路径，若未提供则自动从缓存或网络获取
            model_type: 模型类型，可选值：'d'(默认), 'm2', 'i3'。注意：当提供model_path时此参数无效
        """
        # 当提供了model_path时，忽略model_type
        if model_path:
            self.model_type = 'd'  # 设置一个默认值
            self.model_path = model_path
            if not os.path.exists(model_path):
                raise ValueError(f"模型文件不存在: {model_path}")
        else:
            self.model_type = model_type.lower()
            if self.model_type not in self.MODEL_CONFIGS:
                raise ValueError(f"不支持的模型类型: {model_type}，可选值: {', '.join(self.MODEL_CONFIGS.keys())}")
                
            # 优先检查环境变量中是否设置了模型路径
            env_model_path = os.environ.get("NSFWPY_ONNX_MODEL")
            if env_model_path and os.path.exists(env_model_path):
                self.model_path = env_model_path
            else:
                self.model_path = self._get_model_path()

        # 根据模型文件名确定图像尺寸
        model_filename = os.path.basename(self.model_path)
        if model_filename == 'i3model.onnx':
            self.image_dim = 299
        else:
            self.image_dim = 224
        
        # 创建ONNX运行时会话
        self.session = ort.InferenceSession(self.model_path)
        
        # 获取输入名称
        self.input_name = self.session.get_inputs()[0].name

        # 获取输出名称
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _get_model_path(self):
        """根据平台获取缓存路径，检查模型文件是否存在，不存在则下载"""
        # 首先检查环境变量
        env_model_path = os.environ.get("NSFW_ONNX_MODEL")
        if env_model_path:
            # 如果环境变量指定的是目录而非文件，则在目录下查找model.onnx
            if os.path.isdir(env_model_path):
                model_path = os.path.join(env_model_path, self.MODEL_CONFIGS[self.model_type]['filename'])
            else:
                model_path = env_model_path
                
            if os.path.exists(model_path):
                return model_path
                
        # 确定平台相关的用户缓存目录
        system = platform.system()
        if system == "Windows":
            cache_dir = os.path.join(os.environ.get("LOCALAPPDATA"), "nsfwpy")
        elif system == "Darwin":  # macOS
            cache_dir = os.path.join(os.path.expanduser("~"), "Library", "Caches", "nsfwpy")
        else:  # Linux和其他系统
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "nsfwpy")
        
        # 确保目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, self.MODEL_CONFIGS[self.model_type]['filename'])
        # 检查模型文件是否存在，不存在则下载
        if not os.path.exists(model_path):
            print(f"ONNX模型文件不存在，正在下载到 {model_path}...")
            try:
                self._download_file(self.MODEL_CONFIGS[self.model_type]['url'], model_path)
                print("模型下载完成")
            except Exception as e:
                raise ValueError(f"模型下载失败: {e}")
        
        return model_path
    
    def _download_file(self, url, destination):
        """从指定URL下载文件到目标路径"""
        try:
            with urllib.request.urlopen(url) as response:
                with open(destination, "wb") as f:
                    f.write(response.read())
        except Exception as e:
            raise ValueError(f"下载失败: {e}")

    def _process_gif(self, gif_image):
        """处理GIF图像，对每一帧进行分析并返回平均值"""
        try:
            frame_count = getattr(gif_image, 'n_frames', 1)
            if frame_count == 1:
                # 不是动画GIF，按普通图像处理
                return self._process_pil_image(gif_image)
            all_predictions = []
            # 遍历每一帧
            for frame_idx in range(frame_count):
                gif_image.seek(frame_idx)
                # 对每一帧转换为RGB处理(GIF帧可能是P模式)
                frame = gif_image.convert('RGB')
                processed_frame = self._process_pil_image(frame)
                if processed_frame is not None:
                    # 直接对每一帧进行预测，避免维度问题
                    predictions = self._predict_single(processed_frame)
                    all_predictions.append(predictions)
            if not all_predictions:
                return None
            
            # 计算所有帧预测结果的平均值
            avg_predictions = np.mean(all_predictions, axis=0)
            return avg_predictions
        except Exception as ex:
            print(f"处理GIF图像时出错: {ex}")
            return None
    
    def _load_image(self, image_path):
        """加载并处理单个图像"""
        try:
            image = Image.open(image_path)
            
            # 检查是否为GIF文件
            if getattr(image, 'is_animated', False) or (image.format == 'GIF' and getattr(image, 'n_frames', 1) > 1):
                # 对GIF进行特殊处理，获得平均预测结果
                predictions = self._process_gif(image)
                if predictions is not None:
                    # 由于_process_gif直接返回预测结果，需要特殊处理
                    return predictions, True
                    
            # 常规图像处理
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            # 将图像转换为NumPy数组并归一化
            image = np.array(image, dtype=np.float32) / 255.0
            # 将维度从 (H,W,C) 调整为 (N,H,W,C)
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as ex:
            print(f"处理图像出错 {image_path}: {ex}")
            return None
    
    def _process_pil_image(self, pil_image):
        """处理PIL图像对象"""
        try:
            # 检查是否为GIF动画
            if hasattr(pil_image, 'is_animated') and pil_image.is_animated or \
               (pil_image.format == 'GIF' and getattr(pil_image, 'n_frames', 1) > 1):
                # 对GIF进行特殊处理，获得平均预测结果
                predictions = self._process_gif(pil_image)
                if predictions is not None:
                    return predictions, True
            
            # 常规图像处理
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            resized_image = pil_image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            # 将图像转换为NumPy数组并归一化
            image = np.array(resized_image, dtype=np.float32) / 255.0
            # 将维度从 (H,W,C) 调整为 (N,H,W,C)
            image = np.expand_dims(image, axis=0)
            return image
        except Exception as ex:
            print(f"处理PIL图像出错: {ex}")
            return None
    
    def _predict_single(self, image):
        """对单个图像进行预测"""
        # 使用ONNX运行时进行推理
        outputs = self.session.run(self.output_names, {self.input_name: image})
        return outputs[0][0]
    
    def _format_predictions(self, predictions):
        """将预测结果格式化为类别和概率"""
        result = {}
        for idx, probability in enumerate(predictions):
            category = self.CATEGORIES[idx]
            result[category] = format(probability, '.8f')
        return result
    
    def predict_image(self, image_path):
        """
        预测单个图像的NSFW内容
        
        参数:
            image_path: 图像文件路径
            
        返回:
            包含各类别预测概率的字典
        """
        if not os.path.exists(image_path):
            raise ValueError(f"图像文件不存在: {image_path}")
            
        result = self._load_image(image_path)
        if result is None:
            return None
            
        # 检查是否返回元组，元组表示是GIF并已经处理完成
        if isinstance(result, tuple) and len(result) == 2 and result[1] is True:
            # 已经是处理完的预测结果，直接格式化
            return self._format_predictions(result[0])
        else:
            # 普通图像，需要进行预测
            predictions = self._predict_single(result)
            return self._format_predictions(predictions)
    
    def predict_pil_image(self, pil_image):
        """
        从PIL图像对象预测NSFW内容
        
        参数:
            pil_image: PIL图像对象
            
        返回:
            包含各类别预测概率的字典
        """
        image = self._process_pil_image(pil_image)
        if image is None:
            return None
            
        # 检查是否返回元组，元组表示是GIF并已经处理完成
        if isinstance(image, tuple) and len(image) == 2 and image[1] is True:
            # 已经是处理完的预测结果，直接格式化
            return self._format_predictions(image[0])
        else:
            # 普通图像，需要进行预测
            predictions = self._predict_single(image)
            return self._format_predictions(predictions)
    
    def predict_from_bytes(self, image_bytes):
        """
        从字节流预测NSFW内容
        
        参数:
            image_bytes: 图像字节流
            
        返回:
            包含各类别预测概率的字典
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.predict_pil_image(image)
        except Exception as ex:
            print(f"从字节流处理图像出错: {ex}")
            return None
    
    def predict_batch(self, image_paths):
        """
        批量预测多个图像
        
        参数:
            image_paths: 单个图像路径或包含图像的目录
            
        返回:
            包含每个图像预测结果的列表
        """
        # 处理目录参数
        if os.path.isdir(image_paths):
            paths = [os.path.join(image_paths, f) for f in os.listdir(image_paths) 
                   if os.path.isfile(os.path.join(image_paths, f))]
        elif isinstance(image_paths, list):
            paths = image_paths
        else:
            paths = [image_paths]
        
        results = []
        for path in paths:
            prediction = self.predict_image(path)
            if prediction:
                results.append(prediction)
                
        return results

    async def _load_image_async(self, image_path):
        """异步加载并处理单个图像"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._load_image, image_path)
    
    async def _process_pil_image_async(self, pil_image):
        """异步处理PIL图像对象"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._process_pil_image, pil_image)
    
    async def _predict_single_async(self, image):
        """异步对单个图像进行预测"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._predict_single, image)
    
    async def predict_image_async(self, image_path):
        """
        异步预测单个图像的NSFW内容
        
        参数:
            image_path: 图像文件路径
            
        返回:
            包含各类别预测概率的字典
        """
        if not os.path.exists(image_path):
            raise ValueError(f"图像文件不存在: {image_path}")
            
        result = await self._load_image_async(image_path)
        if result is None:
            return None
            
        # 检查是否返回元组，元组表示是GIF并已经处理完成
        if isinstance(result, tuple) and len(result) == 2 and result[1] is True:
            # 已经是处理完的预测结果，直接格式化
            return self._format_predictions(result[0])
        else:
            # 普通图像，需要进行预测
            predictions = await self._predict_single_async(result)
            return self._format_predictions(predictions)
    
    async def predict_pil_image_async(self, pil_image):
        """
        异步从PIL图像对象预测NSFW内容
        
        参数:
            pil_image: PIL图像对象
            
        返回:
            包含各类别预测概率的字典
        """
        image = await self._process_pil_image_async(pil_image)
        if image is None:
            return None
            
        # 检查是否返回元组，元组表示是GIF并已经处理完成
        if isinstance(image, tuple) and len(image) == 2 and image[1] is True:
            # 已经是处理完的预测结果，直接格式化
            return self._format_predictions(image[0])
        else:
            # 普通图像，需要进行预测
            predictions = await self._predict_single_async(image)
            return self._format_predictions(predictions)
    
    async def predict_from_bytes_async(self, image_bytes):
        """
        异步从字节流预测NSFW内容
        
        参数:
            image_bytes: 图像字节流
            
        返回:
            包含各类别预测概率的字典
        """
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                image = await loop.run_in_executor(
                    pool, 
                    lambda: Image.open(io.BytesIO(image_bytes))
                )
            return await self.predict_pil_image_async(image)
        except Exception as ex:
            print(f"从字节流处理图像出错: {ex}")
            return None
    
    async def predict_batch_async(self, image_paths):
        """
        异步批量预测多个图像
        
        参数:
            image_paths: 单个图像路径或包含图像的目录
            
        返回:
            包含每个图像预测结果的列表
        """
        # 处理目录参数
        if os.path.isdir(image_paths):
            paths = [os.path.join(image_paths, f) for f in os.listdir(image_paths) 
                   if os.path.isfile(os.path.join(image_paths, f))]
        elif isinstance(image_paths, list):
            paths = image_paths
        else:
            paths = [image_paths]
        
        tasks = [self.predict_image_async(path) for path in paths]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    def _process_video_frames(self, video_path, sample_rate=1.0, max_frames=None):
        """
        处理视频文件，按采样率抽取帧并进行NSFW检测
        
        参数:
            video_path: 视频文件路径
            sample_rate: 采样率，范围0-1，例如0.1表示每10帧取1帧
            max_frames: 最大处理帧数，None表示不限制
            
        返回:
            包含各类别预测概率和每秒得分列表的字典
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
                
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames <= 0:
                raise ValueError(f"视频没有可处理的帧: {video_path}")
                
            # 计算采样间隔
            if sample_rate <= 0 or sample_rate > 1:
                sample_rate = 1.0
            frame_interval = int(1 / sample_rate)
            
            # 限制最大帧数
            frames_to_process = total_frames
            if max_frames and max_frames > 0:
                frames_to_process = min(total_frames, max_frames * frame_interval)
            
            all_predictions = []
            frame_scores = []  # 每一帧的得分
            
            # 处理视频帧
            for frame_idx in range(0, frames_to_process, frame_interval):
                # 设置读取位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 将OpenCV的BGR格式转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 创建PIL图像
                pil_image = Image.fromarray(frame_rgb)
                
                # 处理图像
                processed_frame = self._process_pil_image(pil_image)
                if processed_frame is not None:
                    # 进行预测
                    predictions = self._predict_single(processed_frame)
                    all_predictions.append(predictions)
                    
                    # 记录时间戳和对应的得分
                    timestamp = frame_idx / fps
                    frame_score = {
                        'time': round(timestamp, 2),
                        'predictions': self._format_predictions(predictions)
                    }
                    frame_scores.append(frame_score)
            
            # 释放视频资源
            cap.release()
            
            if not all_predictions:
                return None
                
            # 计算所有抽样帧的平均预测结果
            avg_predictions = np.mean(all_predictions, axis=0)
            
            # 返回结果
            result = {
                'average': self._format_predictions(avg_predictions),
                'frames': frame_scores,
                'metadata': {
                    'total_frames': total_frames,
                    'processed_frames': len(all_predictions),
                    'fps': fps,
                    'duration': duration,
                    'sample_rate': sample_rate
                }
            }
            return result
            
        except Exception as ex:
            print(f"处理视频时出错: {ex}")
            return None

    def predict_video(self, video_path, sample_rate=0.1, max_frames=100):
        """
        预测视频文件的NSFW内容
        
        参数:
            video_path: 视频文件路径
            sample_rate: 采样率，范围0-1，例如0.1表示每10帧取1帧
            max_frames: 最大处理帧数，None表示不限制
            
        返回:
            包含NSFW分析结果的字典
        """
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
            
        return self._process_video_frames(video_path, sample_rate, max_frames)

    async def _process_video_frames_async(self, video_path, sample_rate=1.0, max_frames=None):
        """异步处理视频帧"""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(
                pool, 
                lambda: self._process_video_frames(video_path, sample_rate, max_frames)
            )
    
    async def predict_video_async(self, video_path, sample_rate=0.1, max_frames=None):
        """
        异步预测视频文件的NSFW内容
        
        参数:
            video_path: 视频文件路径
            sample_rate: 采样率，范围0-1，例如0.1表示每10帧取1帧
            max_frames: 最大处理帧数，None表示不限制
            
        返回:
            包含NSFW分析结果的字典
        """
        if not os.path.exists(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
            
        return await self._process_video_frames_async(video_path, sample_rate, max_frames)