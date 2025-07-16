from queue import Queue
import threading
from threading import Thread
from utils.utils import get_cpu_memory_usage_gb
import os
import numpy as np
import time

os.makedirs("results", exist_ok=True)
SAVE_QUEUE_LIMIT = 100

def start_saver(name_prefix="result", batch_size=50):
    """
    Starts a CPU saver thread that writes batches to disk.
    Returns: (queue, thread, memory_log)
    """
    queue = Queue(maxsize=SAVE_QUEUE_LIMIT)
    memory_log = []
    batch_file_paths = []

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
            memory_log.append(get_cpu_memory_usage_gb())
            queue.task_done()

            if len(batch) >= batch_size:
                file_path = _save_batch(batch, name_prefix, batch_counter)
                batch_file_paths.append(file_path)
                batch.clear()
                batch_counter += 1

        if batch:
            file_path = _save_batch(batch, name_prefix, batch_counter)
            batch_file_paths.append(file_path)

    def _save_batch(batch, prefix, batch_idx):
        filename = f"{prefix}_batch_{batch_idx}.npy"
        full_path = filename if os.path.dirname(filename) else os.path.join("results", filename)

        if all(r.shape == batch[0].shape for r in batch):
            np.save(full_path, np.stack(batch))
        else:
            full_path = full_path.replace(".npy", ".npz")
            np.savez(full_path, **{f"m{i}": m for i, m in enumerate(batch)})
        print(f"[Hybrid-Saver] Saved batch {batch_idx} with {len(batch)} results")
        return full_path

    thread = Thread(target=saver, daemon=True)
    thread.start()
    return queue, thread, memory_log, batch_file_paths

def stop_saver(queue, thread):
    """Stops the saver thread and waits for final cleanup."""
    queue.put(None)
    thread.join()
    queue.join()

def save_results_to_disk(filename: str, results: list):
    """
    Saves a list of matrix results to a .npy or .npz file in the results/ folder.
    """
    if not results:
        print(f"[Saver] No results to save for {filename}")
        return

    full_path = os.path.join("results", filename)
    try:
        if all(r.shape == results[0].shape for r in results):
            stacked = np.stack(results)
            np.save(full_path, stacked)
            print(f"[Saver] Saved {full_path} (stacked) with shape {stacked.shape}")
        else:
            save_dict = {f"matrix_{i}": r for i, r in enumerate(results)}
            np.savez(full_path.replace(".npy", ".npz"), **save_dict)
            print(f"[Saver] Saved {full_path.replace('.npy', '.npz')} with {len(results)} variable-shaped results")
    except Exception as e:
        print(f"[Saver] Error saving to disk: {e}")

# def save_merged_results(output_file, batch_files):
#     """Loads batch files, merges them, saves a single .npy or .npz file, and deletes the batch files."""
#     results = []
#
#     for file_path in batch_files:
#         try:
#             if file_path.endswith(".npy"):
#                 results.append(np.load(file_path))
#             elif file_path.endswith(".npz"):
#                 with np.load(file_path) as npz:
#                     results.extend([npz[key] for key in npz])
#         except Exception as e:
#             print(f"[Saver] Error loading {file_path}: {e}")
#
#     # Flatten all loaded batches
#     flat_results = [item for sublist in results for item in (sublist if isinstance(sublist, np.ndarray) and sublist.ndim > 2 else [sublist])]
#
#     try:
#         full_path = os.path.join("results", output_file)
#         if all(r.shape == flat_results[0].shape for r in flat_results):
#             np.save(full_path, np.stack(flat_results))
#             print(f"[Saver] Merged and saved {full_path} (stacked) with shape {np.stack(flat_results).shape}")
#         else:
#             np.savez(full_path.replace(".npy", ".npz"), **{f"m{i}": m for i, m in enumerate(flat_results)})
#             print(f"[Saver] Merged and saved {full_path.replace('.npy', '.npz')} with {len(flat_results)} variable-shaped results")
#     except Exception as e:
#         print(f"[Saver] Failed to merge results: {e}")
#
#     # Clean up temp files
#     for file_path in batch_files:
#         try:
#             os.remove(file_path)
#         except Exception as e:
#             print(f"[Saver] Warning: could not delete {file_path}: {e}")


def start_memory_logger(interval=1.0):
    """
    Starts a thread that logs CPU memory usage every `interval` second.
    Returns: (memory_log, thread, stop_flag)
    """
    memory_log = []
    stop_flag = threading.Event()

    def log_memory():
        while not stop_flag.is_set():
            memory_log.append(get_cpu_memory_usage_gb())
            time.sleep(interval)

    thread = threading.Thread(target=log_memory, daemon=True)
    thread.start()

    return memory_log, thread, stop_flag

def stop_memory_logger(thread, stop_flag):
    """
    Stops the memory logger thread.
    """
    stop_flag.set()
    thread.join()