#!/bin/bash

# 腳本：Docker/gpu/docker_run.sh
# 用途：運行 Pac-Man 遊戲或 DQN 訓練的 Docker 容器（GPU 版本）

# 預設參數
IMAGE_NAME="pacman-game"
TAG="gpu"
MODE="game"
DISPLAY_IP="172.25.208.1"  # 預設 WSL2 主機 IP，需根據實際情況調整

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            MODE="train"
            shift
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --display-ip)
            DISPLAY_IP="$2"
            shift 2
            ;;
        *)
            echo "未知參數: $1"
            echo "用法: ./Docker/gpu/docker_run.sh [--train] [--image-name <name>] [--tag <tag>] [--display-ip <ip>]"
            exit 1
            ;;
    esac
done

# 檢查映像是否存在
if ! docker image inspect "$IMAGE_NAME:$TAG" > /dev/null 2>&1; then
    echo "錯誤：映像 $IMAGE_NAME:$TAG 不存在！請先運行 Docker/gpu/build.sh 構建映像。"
    exit 1
fi

# 設置 DISPLAY 環境變數
DISPLAY="${DISPLAY_IP}:0.0"
echo "設置 DISPLAY 為 $DISPLAY"

# 根據模式運行容器
if [[ "$MODE" == "game" ]]; then
    echo "運行遊戲模式（需要 X11 顯示伺服器，例如 VcXsrv）..."
    docker run --rm -it --gpus all \
        -e DISPLAY="$DISPLAY" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$(pwd)":/app \
        "$IMAGE_NAME:$TAG" python3 main.py
elif [[ "$MODE" == "train" ]]; then
    echo "運行 DQN 訓練模式..."
    docker run --rm -it --gpus all \
        -v "$(pwd)":/app \
        "$IMAGE_NAME:$TAG" python3 ai/train.py
fi

# 檢查運行是否成功
if [[ $? -eq 0 ]]; then
    echo "容器運行成功！"
else
    echo "容器運行失敗，請檢查 DISPLAY 設置或依賴。"
    exit 1
fi