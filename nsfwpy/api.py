from typing import List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import __version__ as version
from .nsfw import NSFWDetectorONNX

# 全局模型实例
global_detector = None

# 加载模型的辅助函数
def get_detector(model_path=None):
    global global_detector
    
    # 如果已经有全局模型实例，直接返回
    if (global_detector is not None):
        return global_detector
    
    # 创建新的检测器实例
    detector = NSFWDetectorONNX(model_path=model_path)
    
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
        result = detector.predict_from_bytes(image_bytes)
        
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
                result = detector.predict_from_bytes(image_bytes)

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
