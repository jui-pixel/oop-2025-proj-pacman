# 使用官方 Python 3.13 基礎映像
FROM python:3.13-slim

# 設置工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY docx/requirements.txt .

# 安裝系統依賴（Pygame 需要 SDL 相關庫）
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 設置環境變數以避免 Pygame 的互動問題
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# 默認命令：運行遊戲
CMD ["python", "main.py"]