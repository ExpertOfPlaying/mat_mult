"""Hybrid GPU-CPU matrix multiplication pipeline.

Performs GPU matrix multiplications, saves results to disk asynchronously via a CPU saver thread.
"""

import logging
from typing import List, Tuple, Any, Optional
import cupy as cp
from time import perf_counter

from helper.dot_wrappers import safe_gpu_dot
from helper.multiplication_saver import start_saver, stop_saver, save_merged_results
from data.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)

def multiply_hybrid(
    pairs: List[Tuple[Any, Any]],
    stats_tracker: Optional[StatsTracker] = None,
    batch_size: int = 50,
    out_filename: str = "hybrid_results.npy"
) -> Tuple[List[str], StatsTracker]:
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    logger.info("[Hybrid] Starting GPU multiplication with CPU saving pipeline...")

    queue, saver_thread, cpu_memory_log, batch_file_paths, gpu_memory_log = start_saver(
        name_prefix="hybrid_result",
        batch_size=batch_size,
        memory_logging_mode='both'
    )

    try:
        for idx, (a, b) in enumerate(pairs):
            start = perf_counter()
            try:
                a_gpu = cp.asarray(a)
                b_gpu = cp.asarray(b)
                result_gpu = safe_gpu_dot(a_gpu, b_gpu)
                result_cpu = cp.asnumpy(result_gpu)
                end = perf_counter()

                stats_tracker.record_timing("hybrid", end - start)
                queue.put((result_cpu, idx))
            except Exception as e:
                logger.error(f"[Hybrid] Error during multiplication (pair {idx}): {e}")

    except Exception as fatal:
        logger.exception(f"[Hybrid] Fatal error in main hybrid pipeline: {fatal}")

    else:
        # Batch saving time: time it takes to drain and finish saver thread
        batch_save_start = perf_counter()
        stop_saver(queue, saver_thread)
        batch_save_end = perf_counter()
        stats_tracker.record_saving_time("hybrid", batch_save_end - batch_save_start)

        # Merging time: only the actual merging
        merge_start = perf_counter()
        save_merged_results(out_filename, batch_file_paths)
        merge_end = perf_counter()
        stats_tracker.record_merging_time("hybrid", merge_end - merge_start)

    return batch_file_paths, stats_tracker