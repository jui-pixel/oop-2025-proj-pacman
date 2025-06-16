# 使用 NVIDIA CUDA 基礎映像，支援 PyTorch GPU
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04

# 安裝 Python 和必要工具
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# 設置工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY docx/requirements.txt .

# 安裝 Python 依賴
RUN pip3 install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 設置環境變數
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# 默認命令：運行遊戲
CMD ["python3", "main.py"]