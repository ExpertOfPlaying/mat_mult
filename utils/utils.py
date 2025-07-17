"""Utility functions for matrix loading, permutations, memory tracking, and timing."""

import os
import time
import logging
import threading
from typing import Callable, List, Tuple
import numpy as np
import cupy as cp
from itertools import permutations
import psutil
from functools import wraps

logger = logging.getLogger(__name__)

def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func (Callable): Function to time.

    Returns:
        Callable: Wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__}-pipeline took {end - start:.2f} seconds")
        return result
    return wrapper

def load_matrices(file_path: str) -> List[np.ndarray]:
    """
    Load matrices from a .npy or .npz file.

    Args:
        file_path (str): Path to .npy or .npz file.

    Returns:
        List[np.ndarray]: List of loaded matrices.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If unsupported file type.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    if file_path.endswith(".npy"):
        return list(np.load(file_path, allow_pickle=True))
    elif file_path.endswith(".npz"):
        data = np.load(file_path)
        return [data[key] for key in sorted(data.files)]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def merge_and_permute_sets(*matrix_lists: List[np.ndarray], verbose: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Merge multiple sets and generate all valid permutations (A×B and B×A).

    Args:
        *matrix_lists: Lists of matrices to merge.
        verbose (bool): Whether to log summary info.

    Returns:
        List[Tuple]: Valid (a, b) pairs for multiplication.
    """
    all_matrices: List[np.ndarray] = []
    for matrix_list in matrix_lists:
        all_matrices.extend(matrix_list)

    if verbose:
        logger.info(f"Total matrices merged: {len(all_matrices)}")
    valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

    for a, b in permutations(all_matrices, 2):
        if a.shape[1] == b.shape[0]:
            valid_pairs.append((a, b))

    if verbose:
        logger.info(f"Total valid permutations: {len(valid_pairs)}")
    return valid_pairs

def get_cpu_memory_usage_gb() -> float:
    """Return used RAM in GB."""
    return psutil.virtual_memory().used / 1e9

def get_gpu_memory_usage_gb() -> float:
    """Return used GPU memory in GB (first device)."""
    try:
        free, total = cp.cuda.runtime.memGetInfo()
        used = (total - free) / 1e9
        return used
    except cp.cuda.runtime.CUDARuntimeError:
        return 0.0

def start_hybrid_memory_logger(
    interval: float = 1.0
) -> Tuple[List[float], List[float], threading.Thread, threading.Thread, threading.Event]:
    """
    Starts threads to log CPU and GPU memory usage every interval seconds.

    Returns:
        cpu_log (List[float]): Logged CPU memory usage.
        gpu_log (List[float]): Logged GPU memory usage.
        cpu_thread (Thread): CPU logger thread.
        gpu_thread (Thread): GPU logger thread.
        stop_flag (Event): Event for stopping both loggers.
    """
    cpu_log = []
    gpu_log = []
    stop_flag = threading.Event()

    def cpu_logger():
        while not stop_flag.is_set():
            cpu_log.append(get_cpu_memory_usage_gb())
            time.sleep(interval)

    def gpu_logger():
        while not stop_flag.is_set():
            gpu_log.append(get_gpu_memory_usage_gb())
            time.sleep(interval)

    cpu_thread = threading.Thread(target=cpu_logger, daemon=True)
    gpu_thread = threading.Thread(target=gpu_logger, daemon=True)
    cpu_thread.start()
    gpu_thread.start()

    return cpu_log, gpu_log, cpu_thread, gpu_thread, stop_flag

def stop_hybrid_memory_logger(cpu_thread, gpu_thread, stop_flag):
    """Stops both CPU and GPU memory logger threads."""
    stop_flag.set()
    cpu_thread.join()
    gpu_thread.join()