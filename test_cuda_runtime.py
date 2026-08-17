import ctypes

from runtime.cuda_runtime import setup_cuda_runtime


setup_cuda_runtime()


print("Testing cuBLAS...")

ctypes.CDLL("cublas64_12.dll")

print("cuBLAS: OK")


print("Testing cuDNN...")

ctypes.CDLL("cudnn64_9.dll")

print("cuDNN: OK")


print("Testing NVRTC...")

ctypes.CDLL("nvrtc64_120_0.dll")

print("NVRTC: OK")


print()
print("CUDA runtime is ready.")