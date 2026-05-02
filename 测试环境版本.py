import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version (runtime):", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())