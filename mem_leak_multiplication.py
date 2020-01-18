import os
import numpy as np
import psutil
import torch
import argparse
import time

from tqdm import trange

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--iter", type=int, required=False, default=100000)

    args = arg_parser.parse_args()
    n_samples = args.iter

    pid = os.getpid()
    prev_memoryUse = 0.0

    py = psutil.Process(pid)
    init_mem = py.memory_info()[0] / 2. ** 30
    print(f"PyTorch Version {torch.__version__}")
    t0 = time.perf_counter()
    for i in range(n_samples):
        tmp = torch.matmul(torch.zeros((1, 256, 256)), torch.zeros((1, 256, 2)))

    t_torch = time.perf_counter() - t0
    memoryUse = py.memory_info()[0] / 2. ** 30
    print(f"Computation Time = {t_torch:.2}s \tMemory = {memoryUse:.3e}Gb \t Delta Memory = {memoryUse - init_mem:+.3e}Gb")

    print(f"\nNumpy Version {np.__version__}")
    time.sleep(0.1)

    init_mem = py.memory_info()[0] / 2. ** 30
    t0 = time.perf_counter()
    for i in range(n_samples):
        tmp = np.matmul(np.zeros((1, 256, 256)), np.zeros((1, 256, 2)))

    t_numpy = time.perf_counter() - t0
    memoryUse = py.memory_info()[0] / 2. ** 30
    print(f"Computation Time = {t_numpy:.2}s \tMemory = {memoryUse:.3e}Gb \t Delta Memory = {memoryUse - init_mem:+.3e}Gb")

    print("\n\n\n")



