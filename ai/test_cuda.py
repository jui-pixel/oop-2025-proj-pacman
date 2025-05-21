# ai/test_cuda.py
"""
檢查 CUDA 是否可用，並顯示 GPU 相關資訊。
用於驗證 PyTorch 和 CUDA 環境設置是否正確。
"""
import torch

# 檢查 CUDA 是否可用
print("CUDA 可用: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))  # 顯示 GPU 名稱
    print("CUDA 版本:", torch.version.cuda)  # 顯示 CUDA 版本
    print("PyTorch CUDA 版本:", torch.version.cuda)  # 顯示 PyTorch 的 CUDA 版本
else:
    print("請檢查驅動、CUDA Toolkit 及 PyTorch 是否為 CUDA 版本")