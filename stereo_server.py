from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import tempfile
import os
from pathlib import Path
import shutil
import numpy as np
from omegaconf import OmegaConf
from stereo_service import StereoService
import logging
import uvicorn
from loguru import logger
from Utils import timeit
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# 创建 FastAPI 应用
app = FastAPI(
    title="Foundation Stereo API",
    description="立体视觉服务 API",
    version="1.0.0"
)

# 添加连接日志中间件
class ConnectionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"收到来自 {client_host} 的连接 - {request.method} {request.url.path}")
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"处理请求时出错: {str(e)}")
            raise e

app.add_middleware(ConnectionLoggingMiddleware)

# 初始化立体视觉服务
stereo_service = None

@app.on_event("startup")
async def startup_event():
    global stereo_service
    # 加载配置文件
    config_path = "server_config.yaml"
    if not os.path.exists(config_path):
        raise RuntimeError(f"配置文件 {config_path} 不存在")
    stereo_service = StereoService(config_path=config_path)
    # 预热模型
    stereo_service.preheat()

@timeit
@app.post("/process_stereo_pair/")
async def process_stereo_pair(
    left_image: UploadFile = File(...),
    right_image: UploadFile = File(...),
):
    """
    处理立体图像对
    
    参数:
    - left_image: 左图像文件
    - right_image: 右图像文件
    
    返回:
    - 视差图数据（.npy格式）
    """
    # logging
    logger.info(f"Received request to process stereo pair")
    try:
        # 创建临时目录保存上传的图像
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Saving images to {temp_dir}")

            # 保存上传的图像
            left_path = os.path.join(temp_dir, "left.png")
            right_path = os.path.join(temp_dir, "right.png")
            
            with open(left_path, "wb") as f:
                shutil.copyfileobj(left_image.file, f)
            with open(right_path, "wb") as f:
                shutil.copyfileobj(right_image.file, f)

            logger.info(f"Left image saved to {left_path}")
            logger.info(f"Right image saved to {right_path}")
            
            # 创建自定义配置
            custom_args = OmegaConf.create({
                "output_dir": temp_dir,
                "scale": 1.0,
                "get_pc": 0  # 不生成点云
            })
            
            # 处理图像对
            output_dir = stereo_service.process_stereo_pair(left_path, right_path, custom_args)
            
            logger.info(f"Result saved to {output_dir}")
            
            # 返回视差文件
            disp_file = os.path.join(output_dir, "disp.npy")
            if not os.path.exists(disp_file):
                raise HTTPException(status_code=500, detail="视差文件生成失败")
            
            logger.info(f"Returning disp file {disp_file}")
                
            return FileResponse(
                disp_file,
                media_type="application/octet-stream",
                filename="disp.npy"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2000)
