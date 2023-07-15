import torch
import subprocess

# Print Torch version and CUDA version
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Print nvcc version
nvcc = subprocess.Popen(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = nvcc.communicate()
print(f"nvcc version: {out.decode('utf-8')}")

# Print nvidia-smi version
nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = nvidia_smi.communicate()
print(f"nvidia-smi version: {out.decode('utf-8')}")

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
else:
    print("CUDA is not available.")