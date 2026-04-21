import torch

import sys

print("Python location:", sys.executable)

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Print the CUDA version
print("CUDA Version:", torch.version.cuda)
print("PyTorch Version:", torch.__version__)

# Print the GPU name
if cuda_available:
    print("GPU Name:", torch.cuda.get_device_name(0))

# Create a tensor and move it to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(1000, 1000).to(device)  # Create a random tensor on the GPU
y = torch.rand(1000, 1000).to(device)  # Another random tensor on the GPU

# Perform a matrix multiplication
z = torch.matmul(x, y)
print("Matrix multiplication successful, result shape:", z.shape)

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))