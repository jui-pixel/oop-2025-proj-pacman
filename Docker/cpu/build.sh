#!/bin/bash

# 腳本：Docker/cpu/build.sh
# 用途：構建 Pac-Man 遊戲的 Docker 映像（CPU 版本）

# 預設參數
IMAGE_NAME="pacman-game"
TAG="cpu"
DOCKERFILE="Docker/cpu/Dockerfile"

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "未知參數: $1"
            echo "用法: ./Docker/cpu/build.sh [--image-name <name>] [--tag <tag>]"
            exit 1
            ;;
    esac
done

# 檢查 Dockerfile 是否存在
if [[ ! -f "$DOCKERFILE" ]]; then
    echo "錯誤：$DOCKERFILE 不存在！請確保 Dockerfile 已正確創建。"
    exit 1
fi

# 構建 Docker 映像
echo "正在構建 CPU 版本的 Docker 映像：$IMAGE_NAME:$TAG..."
docker build -t "$IMAGE_NAME:$TAG" -f "$DOCKERFILE" .

# 檢查構建是否成功
if [[ $? -eq 0 ]]; then
    echo "映像 $IMAGE_NAME:$TAG 構建成功！"
    docker images | grep "$IMAGE_NAME"
else
    echo "映像構建失敗，請檢查 Dockerfile 或依賴。"
    exit 1
fi