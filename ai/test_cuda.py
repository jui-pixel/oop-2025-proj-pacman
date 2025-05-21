import torch
print("CUDA 可用: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 名稱:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
    print("PyTorch CUDA 版本:", torch.version.cuda)
else:
    print("請檢查驅動、CUDA Toolkit 及 PyTorch 是否為 CUDA 版本")