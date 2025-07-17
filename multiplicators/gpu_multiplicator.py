"""GPU-based matrix multiplication.

Performs matrix multiplications on the GPU, tracks timings, handles errors, and saves results.
"""

import logging
from typing import List, Tuple, Any, Optional
from time import perf_counter

from helper.dot_wrappers import safe_gpu_dot
from utils.utils import timeit
from helper.multiplication_saver import save_results_to_disk
from data.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)

def process_pair_and_time(pair: Tuple[Any, Any]) -> Tuple[Optional[Any], Optional[str], float]:
    """
    Multiply two matrices (GPU), timing the operation and catching exceptions.

    Args:
        pair (Tuple): Tuple of matrices (a, b)

    Returns:
        result: The result matrix, or None on error
        error:  None or error message
        duration: Time taken (float, seconds)
    """
    a, b = pair
    try:
        start = perf_counter()
        result = safe_gpu_dot(a, b)
        end = perf_counter()
        duration = end - start
        return result, None, duration
    except Exception as e:
        logger.error(f"[GPU] Exception during multiplication: {e}")
        return None, str(e), 0.0

@timeit
def multiply_on_gpu(
    pairs: List[Tuple[Any, Any]],
    stats_tracker: Optional[StatsTracker] = None,
    out_filename: str = "gpu_results.npy"
) -> Tuple[List[Any], StatsTracker]:
    """
    Multiply all matrix pairs on the GPU, tracking timings and saving results.

    Args:
        pairs (List[Tuple]): List of (a, b) matrix pairs to multiply.
        stats_tracker (StatsTracker, optional): Stats tracker for timing.
        out_filename (str): Output file name.

    Returns:
        results (List): All result matrices.
        stats_tracker (StatsTracker): Updated tracker.
    """
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    results: List[Any] = []

    for result, error, duration in map(process_pair_and_time, pairs):
        if error is None and result is not None:
            results.append(result)
            stats_tracker.record_timing("gpu", duration)
        else:
            logger.error(f"[GPU] Error during multiplication: {error}")

    save_start = perf_counter()
    save_results_to_disk(out_filename, results)
    save_end = perf_counter()
    stats_tracker.record_saving_time("gpu", save_end - save_start)

    return results, stats_tracker
