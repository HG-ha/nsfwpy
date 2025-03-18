from typing import List, Dict, Optional
import os
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from . import __version__ as version
from .nsfw import NSFWDetectorONNX

# 全局模型实例
global_detector = None

# 加载模型的辅助函数
def get_detector(model_path=None, model_type=None):
    global global_detector
    
    # 如果已经有全局模型实例，直接返回
    if (global_detector is not None):
        return global_detector
    
    # 如果未指定model_path，尝试从环境变量获取
    if model_path is None:
        model_path = os.environ.get("NSFWPY_ONNX_MODEL")
    
    # 如果未指定model_type，尝试从环境变量获取
    if model_type is None:
        model_type = os.environ.get("NSFWPY_MODEL_TYPE", "d")
    
    # 创建新的检测器实例
    detector = NSFWDetectorONNX(model_path=model_path, model_type=model_type)
    
    # 保存为全局实例
    global_detector = detector
    
    return detector

# 数据模型定义
class ClassifyItem(BaseModel):
    image: UploadFile = File(..., description="上传的图像文件")

class ClassifyManyItem(BaseModel):
    images: List[UploadFile] = File(..., description="上传的图像文件列表")


# 创建FastAPI应用
app = FastAPI(
    title="NSFW内容检测API",
    description="基于onnx的NSFW内容检测API，兼容nsfwjs接口",
    version=version
)


# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 在启动时预加载模型
@app.on_event("startup")
async def startup_event():
    # 在服务启动时预加载模型
    get_detector()

# 辅助函数：从上传文件读取图像
async def read_image_file(file: UploadFile):
    try:
        contents = await file.read()
        return contents
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取上传文件: {str(e)}")

@app.post("/classify", response_model=Dict[str, float])
async def classify_image(image: UploadFile = File(...)):
    """
    对单张上传的图像文件进行NSFW分类
    """
    try:
        detector = get_detector()
        image_bytes = await read_image_file(image)
        result = await detector.predict_from_bytes_async(image_bytes)
        
        # 立即关闭文件
        await image.close()
        # 清理内存
        del image_bytes

        if not result:
            raise HTTPException(status_code=500, detail="图像处理失败")
            
        return result
    except Exception as e:
        if image:
            await image.close()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-many", response_model=List[Dict[str, float]])
async def classify_many_images(images: List[UploadFile] = File(...)):
    """
    对多张上传的图像文件进行NSFW分类
    """
    try:
        detector = get_detector()
        results = []
        
        for image in images:
            try:
                image_bytes = await read_image_file(image)
                result = await detector.predict_from_bytes_async(image_bytes)

                # 立即关闭文件
                await image.close()
                # 清理内存
                del image_bytes

                if result:
                    results.append(result)
                else:
                    results.append({"error": "处理失败"})
            except Exception as e:
                if image:
                    await image.close()
                results.append({"error": str(e)})
                
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-video", response_model=Dict)
async def classify_video(
    video: UploadFile = File(...),
    sample_rate: Optional[float] = Form(0.1),
    max_frames: Optional[int] = Form(None)
):
    """
    对上传的视频文件进行NSFW分类
    
    - sample_rate: 采样率，0-1之间，如0.1表示每10帧取1帧
    - max_frames: 最大处理帧数，限制处理量
    """
    try:
        detector = get_detector()
        
        # 保存上传的视频到临时文件
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f"temp_video_{os.urandom(8).hex()}.mp4")
        
        try:
            # 读取并保存视频文件
            video_content = await video.read()
            with open(temp_file, "wb") as f:
                f.write(video_content)
            
            # 释放内存
            del video_content
            await video.close()
            
            # 处理视频
            result = await detector.predict_video_async(
                temp_file, 
                sample_rate=float(sample_rate) if sample_rate is not None else 0.1, 
                max_frames=int(max_frames) if max_frames is not None else None
            )
            
            if not result:
                raise HTTPException(status_code=500, detail="视频处理失败")
                
            return result
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    except Exception as e:
        if video:
            await video.close()
        raise HTTPException(status_code=500, detail=str(e))
