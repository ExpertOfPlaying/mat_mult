"""Dot product wrappers for CPU (NumPy) and GPU (CuPy).

Provides safe matrix multiplication functions with memory management and error handling.
"""

import logging
import numpy as np
import cupy as cp
from typing import Any

logger = logging.getLogger(__name__)

def safe_cpu_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the dot product using NumPy on CPU.

    Args:
        a (np.ndarray): First matrix.
        b (np.ndarray): Second matrix.

    Returns:
        np.ndarray: The result of the matrix multiplication.

    Raises:
        ValueError: If the shapes are not aligned for matrix multiplication.
    """
    try:
        return np.dot(a, b)
    except ValueError as e:
        logger.error(f"CPU dot product shape mismatch: {a.shape} x {b.shape}. Error: {e}")
        raise

def safe_gpu_dot(a: Any, b: Any, memory_pool: cp.cuda.MemoryPool = None) -> np.ndarray:
    """
    Compute the dot product using CuPy on GPU, transfer the result to CPU, and clear memory.

    Args:
        a (array-like): First matrix (NumPy or CuPy array).
        b (array-like): Second matrix (NumPy or CuPy array).
        memory_pool (cp.cuda.MemoryPool, optional): Custom memory pool for CuPy. Defaults to global pool.

    Returns:
        np.ndarray: The result of the matrix multiplication as a NumPy array.

    Raises:
        ValueError: If the shapes are not aligned for matrix multiplication.
        cp.cuda.runtime.CUDARuntimeError: If a GPU memory or computation error occurs.
    """
    if memory_pool is None:
        memory_pool = cp.get_default_memory_pool()

    try:
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result_gpu = cp.dot(a_gpu, b_gpu)
        result_cpu = cp.asnumpy(result_gpu)
        memory_pool.free_all_blocks()
        return result_cpu
    except ValueError as e:
        logger.error(f"GPU dot product shape mismatch: {a.shape} x {b.shape}. Error: {e}")
        raise
    except cp.cuda.runtime.CUDARuntimeError as e:
        logger.error(f"GPU error during dot product: {e}")
        raise

