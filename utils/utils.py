import os
import time
import numpy as np
import cupy as cp
from itertools import permutations
import psutil


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def load_matrices(file_path: str):
    """Load matrices from .npy or .npz file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    if file_path.endswith(".npy"):
        return list(np.load(file_path, allow_pickle=True))
    elif file_path.endswith(".npz"):
        data = np.load(file_path)
        return [data[key] for key in sorted(data.files)]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def merge_and_permute_sets(*matrix_lists):
    """Merge multiple sets and generate all valid permutations (A×B and B×A)"""
    all_matrices = []
    for matrix_list in matrix_lists:
        all_matrices.extend(matrix_list)

    print(f"Total matrices merged: {len(all_matrices)}")
    valid_pairs = []

    for a, b in permutations(all_matrices, 2):
        if a.shape[1] == b.shape[0]:
            valid_pairs.append((a, b))

    print(f"Total valid permutations: {len(valid_pairs)}")
    return valid_pairs

def get_cpu_memory_usage_gb():
    """Return used RAM in GB"""
    return psutil.virtual_memory().used / 1e9

def get_gpu_memory_usage_gb():
    """Return used GPU memory in GB (first device)"""
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        used = (total - free) / 1e9
        return used
    except cp.cuda.runtime.CUDARuntimeError:
        return 0.0