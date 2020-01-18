import os
import numpy as np
import psutil
import torch
from tqdm import trange

if __name__ == "__main__":
    print("\n\n\n")

    n_samples = 100000

    pid = os.getpid()
    prev_memoryUse = 0.0

    py = psutil.Process(pid)
    init_mem = py.memory_info()[0] / 2. ** 30
    print(f"PyTorch Version {torch.__version__}")
    for i in trange(n_samples, desc="Torch Memory:", ncols=100):
        tmp = torch.matmul(torch.zeros((1, 256, 256)), torch.zeros((1, 256, 2)))

    memoryUse = py.memory_info()[0] / 2. ** 30
    print(f"Torch Memory: Memory = {memoryUse:.3e}Gb \t Delta Memory = {memoryUse - init_mem:+.3e}Gb")

    print(f"\nNumpy Version {np.__version__}")
    init_mem = py.memory_info()[0] / 2. ** 30
    for i in trange(n_samples, desc="Numpy Memory", ncols=100):
        tmp = np.matmul(np.zeros((1, 256, 256)), np.zeros((1, 256, 2)))

    memoryUse = py.memory_info()[0] / 2. ** 30
    print(f"Numpy Memory: Memory = {memoryUse:.3e}Gb \t Delta Memory = {memoryUse - init_mem:+.3e}Gb")



