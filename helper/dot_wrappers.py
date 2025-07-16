import numpy as np
import cupy as cp

def safe_cpu_dot(a, b):
    """NumPy dot product"""
    return np.dot(a, b)

def safe_gpu_dot(a, b):
    """CuPy dot product, returns CPU result and clears memory."""
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    result_gpu = cp.dot(a_gpu, b_gpu)
    result_cpu = cp.asnumpy(result_gpu)
    cp.get_default_memory_pool().free_all_blocks()
    return result_cpu
