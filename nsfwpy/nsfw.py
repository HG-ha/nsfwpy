import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import platform
import urllib.request

class NSFWDetectorONNX:
    """NSFW内容检测器，基于MobileNet V2模型的ONNX版本"""
    
    CATEGORIES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
    
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

    def _load_image(self, image_path):
        """加载并处理单个图像"""
        try:
            image = Image.open(image_path)
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