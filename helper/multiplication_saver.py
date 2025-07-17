"""Multiplication result saver and memory logger utilities.

Handles batch saving of results (with threading), memory logging, and result merging for
hybrid or batched matrix multiplication workflows.
"""

import os
import threading
import time
import logging
from typing import List, Tuple, Optional
from queue import Queue
import numpy as np

from utils.utils import get_cpu_memory_usage_gb, get_gpu_memory_usage_gb

SAVE_QUEUE_LIMIT = 100
DEFAULT_RESULTS_DIR = "results"

logger = logging.getLogger(__name__)

def start_saver(
    name_prefix: str = "result",
    batch_size: int = 50,
    out_dir: str = DEFAULT_RESULTS_DIR,
    memory_logging_mode: str = 'cpu'
) -> Tuple[Queue, threading.Thread, List[float], List[str], Optional[List[float]]]:
    """
    Starts a saver thread, logs memory usage according to mode.
    memory_logging_mode: 'cpu', 'gpu', or 'both'
    Returns both cpu_memory_log and (optionally) gpu_memory_log.
    """
    os.makedirs(out_dir, exist_ok=True)
    queue = Queue(maxsize=SAVE_QUEUE_LIMIT)
    cpu_memory_log: List[float] = []
    gpu_memory_log: Optional[List[float]] = [] if memory_logging_mode in ('gpu', 'both') else None
    batch_file_paths: List[str] = []

    def saver():
        batch = []
        batch_counter = 0
        while True:
            item = queue.get()
            if item is None:
                queue.task_done()
                break
            result, _ = item
            batch.append(result)

            # Always log CPU memory
            if memory_logging_mode in ('cpu', 'both'):
                cpu_memory_log.append(get_cpu_memory_usage_gb())
            # Optionally log GPU memory
            if memory_logging_mode in ('gpu', 'both') and gpu_memory_log is not None:
                gpu_memory_log.append(get_gpu_memory_usage_gb())
            queue.task_done()

            if len(batch) >= batch_size:
                file_path = _save_batch(batch, name_prefix, batch_counter, out_dir)
                batch_file_paths.append(file_path)
                batch.clear()
                batch_counter += 1
        if batch:
            file_path = _save_batch(batch, name_prefix, batch_counter, out_dir)
            batch_file_paths.append(file_path)

    def _save_batch(batch, prefix, batch_idx, out_dir_saver):
        filename = f"{prefix}_batch_{batch_idx}.npy"
        full_path = os.path.join(out_dir_saver, filename)
        try:
            if all(r.shape == batch[0].shape for r in batch):
                np.save(full_path, np.stack(batch))
            else:
                full_path = full_path.replace(".npy", ".npz")
                np.savez(full_path, **{f"m{i}": m for i, m in enumerate(batch)})
            logger.info(f"[Hybrid-Saver] Saved batch {batch_idx} with {len(batch)} results to {full_path}")
            return full_path
        except Exception as e:
            logger.error(f"[Hybrid-Saver] Failed to save batch {batch_idx}: {e}")
            return full_path

    thread = threading.Thread(target=saver, daemon=True)
    thread.start()
    return queue, thread, cpu_memory_log, batch_file_paths, gpu_memory_log

def stop_saver(queue: Queue, thread: threading.Thread) -> None:
    """
    Stops the saver thread and waits for cleanup.

    Args:
        queue (Queue): The saver queue.
        thread (Thread): The saver thread.
    """
    queue.put(None)
    thread.join()
    queue.join()

def save_results_to_disk(filename: str, results: list, out_dir: str = DEFAULT_RESULTS_DIR) -> None:
    """
    Saves a list of matrix results to a .npy or .npz file in the results/ folder.

    Args:
        filename (str): Output file name.
        results (list): List of numpy arrays.
        out_dir (str): Output directory.
    """
    if not results:
        logger.warning(f"[Saver] No results to save for {filename}")
        return

    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)
    try:
        if all(r.shape == results[0].shape for r in results):
            stacked = np.stack(results)
            np.save(full_path, stacked)
            logger.info(f"[Saver] Saved {full_path} (stacked) with shape {stacked.shape}")
        else:
            save_dict = {f"matrix_{i}": r for i, r in enumerate(results)}
            np.savez(full_path.replace(".npy", ".npz"), **save_dict)
            logger.info(f"[Saver] Saved {full_path.replace('.npy', '.npz')} with {len(results)} variable-shaped results")
    except Exception as e:
        logger.error(f"[Saver] Error saving to disk: {e}")

def save_merged_results(output_file: str, batch_files: List[str], out_dir: str = DEFAULT_RESULTS_DIR) -> None:
    """
    Loads batch files, merges them, saves a single .npy or .npz file, and deletes the batch files.

    Args:
        output_file (str): Final merged result file.
        batch_files (List[str]): Paths to batch files.
        out_dir (str): Output directory.
    """
    results = []
    for file_path in batch_files:
        try:
            if file_path.endswith(".npy"):
                results.append(np.load(file_path))
            elif file_path.endswith(".npz"):
                with np.load(file_path) as npz:
                    results.extend([npz[key] for key in npz])
        except Exception as e:
            logger.error(f"[Saver] Error loading {file_path}: {e}")

    # Flatten all loaded batches
    flat_results = [
        item
        for sublist in results
        for item in (sublist if isinstance(sublist, np.ndarray) and sublist.ndim > 2 else [sublist])
    ]

    try:
        full_path = os.path.join(out_dir, output_file)
        if all(r.shape == flat_results[0].shape for r in flat_results):
            np.save(full_path, np.stack(flat_results))
            logger.info(f"[Saver] Merged and saved {full_path} (stacked) with shape {np.stack(flat_results).shape}")
        else:
            np.savez(full_path.replace(".npy", ".npz"), **{f"m{i}": m for i, m in enumerate(flat_results)})
            logger.info(f"[Saver] Merged and saved {full_path.replace('.npy', '.npz')} with {len(flat_results)} variable-shaped results")
    except Exception as e:
        logger.error(f"[Saver] Failed to merge results: {e}")

    # Clean up temp files
    for file_path in batch_files:
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"[Saver] Warning: could not delete {file_path}: {e}")

def start_memory_logger(
    interval: float = 1.0,
    mode: str = 'cpu'
) -> Tuple[List[float], Optional[List[float]], threading.Thread, Optional[threading.Thread], threading.Event]:
    """
    Starts memory logging for CPU, GPU, or both.

    Returns:
        cpu_log (List[float])
        gpu_log (List[float]) or None
        cpu_thread (Thread)
        gpu_thread (Thread) or None
        stop_flag (Event)
    """
    cpu_log = []
    gpu_log = [] if mode in ('gpu', 'both') else None
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
    cpu_thread.start()

    gpu_thread = None
    if mode in ('gpu', 'both'):
        gpu_thread = threading.Thread(target=gpu_logger, daemon=True)
        gpu_thread.start()

    return cpu_log, gpu_log, cpu_thread, gpu_thread, stop_flag

def stop_memory_logger(cpu_thread, gpu_thread, stop_flag):
    stop_flag.set()
    cpu_thread.join()
    if gpu_thread is not None:
        gpu_thread.join()