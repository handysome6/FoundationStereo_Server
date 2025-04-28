# This file is for Nvidia Graphics Card below RTX 4000 series
# For RTX 5000 series, please use CUDA >=12.8, and cooresponding pytorch/flash-attn/xformers version

# 第一阶段：使用 NVIDIA CUDA 基础镜像
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as cuda-base

# 第二阶段：使用 Miniforge3 基础镜像
FROM condaforge/miniforge3:24.11.3-2 as conda-base

# 最终阶段：组合两个基础镜像
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# 复制 Miniforge3 环境
COPY --from=conda-base /opt/conda /opt/conda

# 设置环境变量
ENV PATH=/opt/conda/bin:$PATH

# 创建工作目录
WORKDIR /app/FoundationStereo_Server

# 复制当前目录内容到容器的 /app 目录
COPY . /app/FoundationStereo_Server

# 确保使用 bash 作为默认 shell
SHELL ["/bin/bash", "-c"]

RUN apt update && apt-get install libosmesa6-dev libglib2.0-0 -y

# 初始化 conda
RUN conda init bash

# RUN mamba install pytorch torchvision torchaudio xformers flash-attn cuda=12.4 -y

# RUN pip install -r requirements.txt

# 设置默认命令
CMD ["/bin/bash"] 
