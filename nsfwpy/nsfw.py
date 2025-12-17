# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort
import io
import json
import platform
import urllib.request
import asyncio
import gc
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# 强制设置标准输出/错误输出为 UTF-8 编码
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import cv2  # 添加OpenCV库

class NSFWDetectorONNX:
    """
    NSFW内容检测器，基于MobileNet V2模型的ONNX版本
    
    支持的环境变量：
    - NSFWPY_ONNX_MODEL: 指定模型文件路径
    - NSFWPY_MODEL_TYPE: 模型类型 (d/m2/i3)
    - NSFWPY_USE_CHINA_MIRROR: 使用国内镜像 (1/true/yes)
    - NSFWPY_GITHUB_MIRROR: 自定义GitHub镜像地址
    - NSFWPY_CLEANUP_INTERVAL: 自动垃圾回收间隔（推理次数），默认100，设为0禁用
    - NSFWPY_INTRA_THREADS: ONNX Runtime内部并行线程数，默认4
    - NSFWPY_INTER_THREADS: ONNX Runtime跨操作并行线程数，默认2
    - NSFWPY_GPU_MEM_LIMIT: GPU显存限制（字节），默认500MB (524288000)
    - NSFWPY_TRT_MAX_WORKSPACE: TensorRT最大工作空间（字节），默认1GB
    """
    
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
    
    def __init__(self, model_path=None, model_type='d', device='auto'):
        """
        初始化NSFW检测器(ONNX版本)
        
        参数:
            model_path: ONNX模型文件路径，若未提供则自动从缓存或网络获取
            model_type: 模型类型，可选值：'d'(默认), 'm2', 'i3'。注意：当提供model_path时此参数无效
            device: 推理设备，可选值：
                - 'auto': 自动选择最佳设备（GPU优先）
                - 'cpu': 仅使用CPU
                - 'cuda': 使用CUDA (NVIDIA GPU)
                - 'tensorrt': 使用TensorRT (NVIDIA GPU优化)
                - 'dml': 使用DirectML (Windows GPU)
                - 'coreml': 使用CoreML (Apple设备)
                - 'openvino': 使用OpenVINO (Intel设备)
        
        内存管理：
            - 自动启用内存模式优化、内存Arena和内存复用
            - GPU默认限制500MB显存，可通过环境变量NSFWPY_GPU_MEM_LIMIT调整
            - 每100次推理自动触发垃圾回收，可通过NSFWPY_CLEANUP_INTERVAL调整
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
        
        # 配置推理设备
        self.device = device.lower()
        self.providers = self._get_execution_providers()
        
        # 创建ONNX运行时会话选项
        sess_options = ort.SessionOptions()
        
        # 图优化级别
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 内存管理优化
        # 启用内存模式优化（减少内存碎片）
        sess_options.enable_mem_pattern = True
        
        # 启用CPU内存Arena（内存池，提高性能并减少内存分配开销）
        sess_options.enable_cpu_mem_arena = True
        
        # 启用内存复用（减少内存分配）
        sess_options.enable_mem_reuse = True
        
        # 线程数优化（避免创建过多线程导致内存开销）
        # 自动检测 CPU 核心数并设置合理的默认值
        cpu_count = os.cpu_count() or 1
        
        # 单核 CPU 建议使用 1 个线程，多核 CPU 建议使用核心数的一半（最多4个）
        default_intra = min(max(cpu_count // 2, 1), 4) if cpu_count > 1 else 1
        default_inter = min(max(cpu_count // 4, 1), 2) if cpu_count > 2 else 1
        
        intra_threads = int(os.environ.get("NSFWPY_INTRA_THREADS", str(default_intra)))
        inter_threads = int(os.environ.get("NSFWPY_INTER_THREADS", str(default_inter)))
        sess_options.intra_op_num_threads = intra_threads
        sess_options.inter_op_num_threads = inter_threads
        
        # 配置执行提供程序选项（包括内存限制）
        provider_options = self._get_provider_options()
        
        # 创建会话
        self.session = ort.InferenceSession(
            self.model_path, 
            sess_options, 
            providers=self.providers,
            provider_options=provider_options
        )
        
        # 获取输入名称
        self.input_name = self.session.get_inputs()[0].name

        # 获取输出名称
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 推理计数器，用于定期清理
        self._inference_count = 0
        # 从环境变量读取清理间隔，默认每100次推理后清理一次
        self._cleanup_interval = int(os.environ.get("NSFWPY_CLEANUP_INTERVAL", "100"))

    def __del__(self):
        """析构函数，确保资源被释放"""
        try:
            if hasattr(self, 'session') and self.session is not None:
                # 清理会话资源
                del self.session
                self.session = None
            gc.collect()
        except:
            pass

    def cleanup_session_cache(self):
        """清理ONNX Runtime会话缓存，释放GPU显存和内存"""
        try:
            # 强制垃圾回收，清理Python对象
            gc.collect()
            
            # ONNX Runtime会自动管理GPU内存
            # 通过垃圾回收释放不再使用的tensor引用即可
            # 如果使用的是CUDAExecutionProvider，ONNX Runtime会自动管理显存
        except Exception as e:
            print(f"清理会话缓存时出错: {e}")
    
    def reset_inference_count(self):
        """重置推理计数器"""
        self._inference_count = 0
    
    def set_cleanup_interval(self, interval: int):
        """
        设置自动清理间隔
        
        参数:
            interval: 清理间隔（推理次数），设置为0表示禁用自动清理
        """
        if interval < 0:
            raise ValueError("清理间隔必须大于等于0")
        self._cleanup_interval = interval
    
    def get_inference_stats(self):
        """
        获取推理统计信息和内存配置
        
        返回:
            包含推理计数、清理间隔和内存配置的字典
        """
        # 收集内存配置信息
        memory_config = {
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": True,
            "enable_mem_reuse": True,
            "cpu_cores": os.cpu_count() or 1,
            "intra_op_threads": self.session.get_session_options().intra_op_num_threads if hasattr(self.session.get_session_options(), 'intra_op_num_threads') else "N/A",
            "inter_op_threads": self.session.get_session_options().inter_op_num_threads if hasattr(self.session.get_session_options(), 'inter_op_num_threads') else "N/A",
        }
        
        # 如果使用GPU，添加GPU内存配置
        if 'CUDAExecutionProvider' in self.providers:
            gpu_mem_limit = os.environ.get("NSFWPY_GPU_MEM_LIMIT", str(500 * 1024 * 1024))
            memory_config["gpu_mem_limit_mb"] = int(gpu_mem_limit) // (1024 * 1024)
            memory_config["arena_extend_strategy"] = "kSameAsRequested"
        
        return {
            "inference_count": self._inference_count,
            "cleanup_interval": self._cleanup_interval,
            "providers": self.providers,
            "device": self.device,
            "model_path": self.model_path,
            "image_dim": self.image_dim,
            "memory_config": memory_config
        }

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
                # 获取原始URL
                original_url = self.MODEL_CONFIGS[self.model_type]['url']
                # 如果在中国，使用代理URL
                download_url = self.get_proxied_github_url(original_url)
                print(f"下载地址: {download_url}")
                # 下载文件
                self._download_file(download_url, model_path)
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
    
    @staticmethod
    def is_user_in_china():
        """检测用户是否在中国大陆"""
        # 优先检查环境变量
        use_proxy = os.environ.get("NSFWPY_USE_CHINA_MIRROR")
        if use_proxy is not None:
            return use_proxy.lower() in ('1', 'true', 'yes')
        
        # 尝试通过IP检测
        try:
            # 设置超时时间避免长时间等待
            req = urllib.request.Request(
                "https://ipapi.co/json/",
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode())
                country = data.get('country_code') or data.get('country')
                return country == 'CN'
        except Exception:
            # IP检测失败，尝试检测时区
            try:
                import time
                tz_offset = -time.timezone / 3600
                # 中国使用 UTC+8
                if tz_offset == 8:
                    return True
            except Exception:
                pass
        
        return False

    @staticmethod
    def get_proxied_github_url(original_url):
        """返回代理或原始 URL
        
        环境变量说明:
        - NSFWPY_USE_CHINA_MIRROR: 设置为 1/true/yes 强制使用国内镜像
        - NSFWPY_GITHUB_MIRROR: 自定义镜像地址，例如 https://ghproxy.cn
        """
        # 检查是否需要使用代理
        if not NSFWDetectorONNX.is_user_in_china():
            return original_url
        
        # 检查自定义镜像地址
        custom_mirror = os.environ.get("NSFWPY_GITHUB_MIRROR")
        if custom_mirror:
            return f"{custom_mirror}/{original_url}"
        
        # 使用默认镜像服务
        return original_url.replace("https://github.com", "https://ghproxy.cn/https://github.com")
    
    def _get_execution_providers(self):
        """
        根据设备配置获取ONNX Runtime执行提供程序列表
        
        返回:
            按优先级排序的执行提供程序列表
        """
        available_providers = ort.get_available_providers()
        providers = []
        
        if self.device == 'auto':
            # 自动选择最佳可用设备，按性能优先级排序
            priority_providers = [
                'TensorrtExecutionProvider',
                'CUDAExecutionProvider',
                'DmlExecutionProvider',
                'CoreMLExecutionProvider',
                'OpenVINOExecutionProvider',
            ]
            for provider in priority_providers:
                if provider in available_providers:
                    providers.append(provider)
                    break  # 使用找到的第一个硬件加速提供程序
                    
        elif self.device == 'cpu':
            pass  # 只使用CPU
            
        elif self.device == 'cuda':
            if 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                
        elif self.device == 'tensorrt':
            if 'TensorrtExecutionProvider' in available_providers:
                providers.extend(['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
            elif 'CUDAExecutionProvider' in available_providers:
                providers.append('CUDAExecutionProvider')
                    
        elif self.device == 'dml':
            if 'DmlExecutionProvider' in available_providers:
                providers.append('DmlExecutionProvider')
                
        elif self.device == 'coreml':
            if 'CoreMLExecutionProvider' in available_providers:
                providers.append('CoreMLExecutionProvider')
                
        elif self.device == 'openvino':
            if 'OpenVINOExecutionProvider' in available_providers:
                providers.append('OpenVINOExecutionProvider')
                
        if 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')
        
        return providers
    
    def _get_provider_options(self):
        """
        获取执行提供程序的选项配置（包括内存限制）
        
        返回:
            提供程序选项列表
        """
        provider_options = []
        
        for provider in self.providers:
            options = {}
            
            if provider == 'CUDAExecutionProvider':
                # GPU内存限制（字节），从环境变量读取，默认500MB
                gpu_mem_limit = os.environ.get("NSFWPY_GPU_MEM_LIMIT")
                if gpu_mem_limit:
                    options['gpu_mem_limit'] = int(gpu_mem_limit)
                else:
                    # 默认500MB显存限制
                    options['gpu_mem_limit'] = 500 * 1024 * 1024
                
                # Arena扩展策略：kSameAsRequested（按需分配）或 kNextPowerOfTwo（2的幂次增长）
                options['arena_extend_strategy'] = 'kSameAsRequested'
                
                # 启用CUDA内存Arena
                options['enable_cuda_graph'] = False  # 禁用CUDA graph以节省内存
                
                # cuDNN卷积算法：EXHAUSTIVE(0), HEURISTIC(1), DEFAULT(2)
                # 使用HEURISTIC以节省初始化时的内存
                options['cudnn_conv_algo_search'] = 'HEURISTIC'
                
            elif provider == 'DmlExecutionProvider':
                # DirectML (Windows GPU) 内存限制
                # DML会自动管理内存，但我们可以设置一些选项
                options['disable_metacommands'] = False
                
            elif provider == 'TensorrtExecutionProvider':
                # TensorRT内存限制
                # 最大工作空间大小（字节），默认1GB
                trt_max_workspace = os.environ.get("NSFWPY_TRT_MAX_WORKSPACE")
                if trt_max_workspace:
                    options['trt_max_workspace_size'] = int(trt_max_workspace)
                else:
                    options['trt_max_workspace_size'] = 1 * 1024 * 1024 * 1024
                
                # 最大缓存引擎数量
                options['trt_max_cached_engines'] = 1
                
            elif provider == 'CPUExecutionProvider':
                # CPU执行器选项
                pass
            
            provider_options.append(options)
        
        return provider_options

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
                    # 清理临时变量
                    del processed_frame
                # 清理帧对象
                del frame
                
            if not all_predictions:
                return None
            
            # 计算所有帧预测结果的平均值
            avg_predictions = np.mean(all_predictions, axis=0)
            # 清理预测列表
            del all_predictions
            gc.collect()
            return avg_predictions
        except Exception as ex:
            print(f"处理GIF图像时出错: {ex}")
            gc.collect()
            return None
    
    def _load_image(self, image_path):
        """加载并处理单个图像"""
        image = None
        try:
            image = Image.open(image_path)
            
            # 检查是否为GIF文件
            if getattr(image, 'is_animated', False) or (image.format == 'GIF' and getattr(image, 'n_frames', 1) > 1):
                # 对GIF进行特殊处理，获得平均预测结果
                predictions = self._process_gif(image)
                image.close()
                if predictions is not None:
                    # 由于_process_gif直接返回预测结果，需要特殊处理
                    gc.collect()
                    return predictions, True
                gc.collect()
                return None
                    
            # 常规图像处理
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
                image.close()
                image = rgb_image
            resized = image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            # 将图像转换为NumPy数组并归一化
            img_array = np.array(resized, dtype=np.float32) / 255.0
            # 关闭图像对象
            image.close()
            del resized
            # 将维度从 (H,W,C) 调整为 (N,H,W,C)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as ex:
            print(f"处理图像出错 {image_path}: {ex}")
            if image:
                image.close()
            gc.collect()
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
                rgb_image = pil_image.convert('RGB')
                pil_image = rgb_image
            resized_image = pil_image.resize((self.image_dim, self.image_dim), Image.BICUBIC)
            # 将图像转换为NumPy数组并归一化
            img_array = np.array(resized_image, dtype=np.float32) / 255.0
            # 清理临时对象
            del resized_image
            # 将维度从 (H,W,C) 调整为 (N,H,W,C)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as ex:
            print(f"处理PIL图像出错: {ex}")
            gc.collect()
            return None
    
    def _predict_single(self, image):
        """对单个图像进行预测"""
        # 使用ONNX运行时进行推理
        outputs = self.session.run(self.output_names, {self.input_name: image})
        result = outputs[0][0].copy()  # 复制结果，避免持有ONNX Runtime的内部缓冲区
        
        # 清理输出引用
        del outputs
        
        # 增加推理计数
        self._inference_count += 1
        
        # 定期清理会话缓存（如果启用了自动清理）
        if self._cleanup_interval > 0 and self._inference_count % self._cleanup_interval == 0:
            gc.collect()
        
        return result
    
    def _format_predictions(self, predictions):
        """将预测结果格式化为类别和概率"""
        result = {}
        for idx, probability in enumerate(predictions):
            category = self.CATEGORIES[idx]
            # 转换为Python float并四舍五入到8位小数，避免科学计数格式
            result[category] = round(float(probability), 8)
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
            gc.collect()
            return None
            
        try:
            # 检查是否返回元组，元组表示是GIF并已经处理完成
            if isinstance(result, tuple) and len(result) == 2 and result[1] is True:
                # 已经是处理完的预测结果，直接格式化
                formatted = self._format_predictions(result[0])
                del result
                gc.collect()
                return formatted
            else:
                # 普通图像，需要进行预测
                predictions = self._predict_single(result)
                del result
                formatted = self._format_predictions(predictions)
                del predictions
                gc.collect()
                return formatted
        except Exception as ex:
            del result
            gc.collect()
            raise ex
    
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
            gc.collect()
            return None
            
        try:
            # 检查是否返回元组，元组表示是GIF并已经处理完成
            if isinstance(image, tuple) and len(image) == 2 and image[1] is True:
                # 已经是处理完的预测结果，直接格式化
                formatted = self._format_predictions(image[0])
                del image
                gc.collect()
                return formatted
            else:
                # 普通图像，需要进行预测
                predictions = self._predict_single(image)
                del image
                formatted = self._format_predictions(predictions)
                del predictions
                gc.collect()
                return formatted
        except Exception as ex:
            del image
            gc.collect()
            raise ex
    
    def predict_from_bytes(self, image_bytes):
        """
        从字节流预测NSFW内容
        
        参数:
            image_bytes: 图像字节流
            
        返回:
            包含各类别预测概率的字典
        """
        image = None
        try:
            image = Image.open(io.BytesIO(image_bytes))
            result = self.predict_pil_image(image)
            image.close()
            del image
            gc.collect()
            return result
        except Exception as ex:
            print(f"从字节流处理图像出错: {ex}")
            if image:
                image.close()
            gc.collect()
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
            # 每处理一张图像后主动回收
            gc.collect()
                
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
            gc.collect()
            return None
            
        try:
            # 检查是否返回元组，元组表示是GIF并已经处理完成
            if isinstance(result, tuple) and len(result) == 2 and result[1] is True:
                # 已经是处理完的预测结果，直接格式化
                formatted = self._format_predictions(result[0])
                del result
                gc.collect()
                return formatted
            else:
                # 普通图像，需要进行预测
                predictions = await self._predict_single_async(result)
                del result
                formatted = self._format_predictions(predictions)
                del predictions
                gc.collect()
                return formatted
        except Exception as ex:
            del result
            gc.collect()
            raise ex
    
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
            gc.collect()
            return None
            
        try:
            # 检查是否返回元组，元组表示是GIF并已经处理完成
            if isinstance(image, tuple) and len(image) == 2 and image[1] is True:
                # 已经是处理完的预测结果，直接格式化
                formatted = self._format_predictions(image[0])
                del image
                gc.collect()
                return formatted
            else:
                # 普通图像，需要进行预测
                predictions = await self._predict_single_async(image)
                del image
                formatted = self._format_predictions(predictions)
                del predictions
                gc.collect()
                return formatted
        except Exception as ex:
            del image
            gc.collect()
            raise ex
    
    async def predict_from_bytes_async(self, image_bytes):
        """
        异步从字节流预测NSFW内容
        
        参数:
            image_bytes: 图像字节流
            
        返回:
            包含各类别预测概率的字典
        """
        image = None
        try:
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor() as pool:
                image = await loop.run_in_executor(
                    pool, 
                    lambda: Image.open(io.BytesIO(image_bytes))
                )
            result = await self.predict_pil_image_async(image)
            image.close()
            del image
            gc.collect()
            return result
        except Exception as ex:
            print(f"从字节流处理图像出错: {ex}")
            if image:
                image.close()
            gc.collect()
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
        filtered_results = [r for r in results if r]
        # 清理临时变量
        del tasks
        del results
        gc.collect()
        return filtered_results

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
        cap = None
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
                # 清理原始帧
                del frame
                
                # 创建PIL图像
                pil_image = Image.fromarray(frame_rgb)
                # 清理numpy数组
                del frame_rgb
                
                # 处理图像
                processed_frame = self._process_pil_image(pil_image)
                # 关闭PIL图像
                pil_image.close()
                del pil_image
                
                if processed_frame is not None:
                    # 进行预测
                    predictions = self._predict_single(processed_frame)
                    del processed_frame
                    
                    all_predictions.append(predictions)
                    
                    # 记录时间戳和对应的得分
                    timestamp = frame_idx / fps
                    frame_score = {
                        'time': round(timestamp, 2),
                        'predictions': self._format_predictions(predictions)
                    }
                    frame_scores.append(frame_score)
                
                # 每处理一定数量帧后进行一次垃圾回收
                if frame_idx % (frame_interval * 10) == 0:
                    gc.collect()
            
            # 释放视频资源
            cap.release()
            cap = None
            
            if not all_predictions:
                gc.collect()
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
            
            # 清理临时变量
            del all_predictions
            del avg_predictions
            gc.collect()
            
            return result
            
        except Exception as ex:
            print(f"处理视频时出错: {ex}")
            if cap:
                cap.release()
            gc.collect()
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
            
        result = self._process_video_frames(video_path, sample_rate, max_frames)
        gc.collect()
        return result

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
            
        result = await self._process_video_frames_async(video_path, sample_rate, max_frames)
        gc.collect()
        return result