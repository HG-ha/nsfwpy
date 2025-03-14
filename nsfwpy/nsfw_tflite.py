
# 保留tflite模型方法

import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import io
import platform
import urllib.request

class NSFWDetector:
    """NSFW内容检测器，基于MobileNet V2模型"""
    
    CATEGORIES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    MODEL_URL = "https://github.com/HG-ha/nsfwpy/raw/main/model/model.tflite"
    
    def __init__(self, model_path=None, image_dim=224):
        """
        初始化NSFW检测器
        
        参数:
            model_path: TFLite模型文件路径，若未提供则自动从缓存或网络获取
            image_dim: 模型输入图像尺寸(默认224x224)
        """
        self.image_dim = image_dim
        
        # 优先检查环境变量中是否设置了模型路径
        env_model_path = os.environ.get("NSFWPY_ONNX_MODEL")
        if env_model_path and os.path.exists(env_model_path):
            model_path = env_model_path
        # 若未通过环境变量或参数提供模型路径，则自动获取
        elif model_path is None:
            model_path = self._get_model_path()
            
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
            
        self.model_path = model_path
        
        # 加载TFLite模型
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # 获取输入输出细节
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def _get_model_path(self):
        """根据平台获取缓存路径，检查模型文件是否存在，不存在则下载"""
        # 首先检查环境变量
        env_model_path = os.environ.get("NSFWPY_ONNX_MODEL")
        if env_model_path:
            # 如果环境变量指定的是目录而非文件，则在目录下查找model.tflite
            if os.path.isdir(env_model_path):
                model_path = os.path.join(env_model_path, "model.tflite")
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
        
        model_path = os.path.join(cache_dir, "model.tflite")
        # 检查模型文件是否存在，不存在则下载
        if not os.path.exists(model_path):
            print(f"模型文件不存在，正在下载到 {model_path}...")
            try:
                self._download_file(self.MODEL_URL, model_path)
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

    def _load_image(self, image_path):
        """加载并处理单个图像"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            image = np.array(image, dtype=np.float32) / 255.0
            return image
        except Exception as ex:
            print(f"处理图像出错 {image_path}: {ex}")
            return None
    
    def _process_pil_image(self, pil_image):
        """处理PIL图像对象"""
        try:
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            resized_image = pil_image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            image = np.array(resized_image, dtype=np.float32) / 255.0
            return image
        except Exception as ex:
            print(f"处理PIL图像出错: {ex}")
            return None
    
    def _predict_single(self, image):
        """对单个图像进行预测"""
        self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(image, axis=0))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    
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
            
        image = self._load_image(image_path)
        if image is None:
            return None
            
        predictions = self._predict_single(image)
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