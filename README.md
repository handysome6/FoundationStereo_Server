# FoundationStereo API 服务

这是一个基于 FastAPI 的立体视觉服务，用于处理立体图像对并生成视差图、深度图和点云。

## 模型权重

- 从[这里](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)下载基础模型用于零样本推理。将整个文件夹（例如 `23-51-11`）放在 `./pretrained_models/` 目录下。

## 安装

1. 克隆仓库：
```bash
git clone <repository_url>
cd FoundationStereo_Server
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 确保 `server_config.yaml` 配置文件存在并正确配置。

## 运行服务

启动服务器：
```bash
python stereo_server.py
```

服务器将在 http://localhost:2000 上运行。你可以访问 http://localhost:8000/docs 查看 API 文档。

## 服务端功能

服务端基于 `stereo_service.py` 实现，提供以下功能：

- 基于 FoundationStereo 模型的立体图像处理
- 支持图像缩放处理
- 自动生成视差图npy

## API 端点

### POST /process_stereo_pair/

处理立体图像对并返回处理结果。

**请求参数：**
- `left_image`: 左图像文件（文件上传）
- `right_image`: 右图像文件（文件上传）

**返回：**
- `disp.npy`: 视差图

## 示例

### 使用 curl 发送请求：
```bash
curl -X POST "http://localhost:2000/process_stereo_pair/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "left_image=@path/to/left.png" \
  -F "right_image=@path/to/right.png" \
  -F "intrinsic_file=@path/to/K.txt" \
  -F "scale=0.5" \
  --output stereo_results.zip
```

### 使用 Python 发送请求：
```python
import requests

url = "http://localhost:8000/process_stereo_pair/"
files = {
    'left_image': ('left.png', open('path/to/left.png', 'rb')),
    'right_image': ('right.png', open('path/to/right.png', 'rb')),
    'intrinsic_file': ('K.txt', open('path/to/K.txt', 'rb'))
}
data = {
    'scale': 0.5  # 可选，默认为1.0
}

response = requests.post(url, files=files, data=data)
with open('stereo_results.zip', 'wb') as f:
    f.write(response.content)
```
