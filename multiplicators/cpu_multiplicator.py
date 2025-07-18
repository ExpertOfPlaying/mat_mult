"""CPU-based matrix multiplication using parallel processing.

Handles multiplication of matrix pairs with timing, error reporting, and batch saving.
"""

import logging
from typing import List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

from helper.dot_wrappers import safe_cpu_dot
from utils.utils import timeit
from data.stats_tracker import StatsTracker
from helper.multiplication_saver import save_results_to_disk

logger = logging.getLogger(__name__)

def process_pair_and_time(pair: Tuple[Any, Any]) -> Tuple[Optional[Any], Optional[str], float]:
    """
    Multiply two matrices (CPU), timing the operation and catching exceptions.

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
        result = safe_cpu_dot(a, b)
        end = perf_counter()
        duration = end - start
        return result, None, duration
    except Exception as e:
        logger.error(f"[CPU] Exception during multiplication: {e}")
        return None, str(e), 0.0

@timeit
def multiply_on_cpu(
    pairs: List[Tuple[Any, Any]],
    stats_tracker: Optional[StatsTracker] = None
) -> Tuple[List[Any], StatsTracker]:
    """
    Multiply all matrix pairs on CPU in parallel, tracking timings and saving results.

    Args:
        pairs (List[Tuple]): List of (a, b) matrix pairs to multiply.
        stats_tracker (StatsTracker, optional): Tracker for timing/statistics.

    Returns:
        results (List): All result matrices.
        stats_tracker (StatsTracker): Updated tracker.
    """
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    results = []

    with ProcessPoolExecutor() as executor:
        for result, error, duration in executor.map(process_pair_and_time, pairs):
            if error is None and result is not None:
                results.append(result)
                stats_tracker.record_timing("cpu", duration)
            else:
                logger.error(f"[CPU] Error during multiplication: {error}")

    # Time and record the saving step
    save_start = perf_counter()
    save_results_to_disk("cpu_results.npy", results)
    save_end = perf_counter()
    stats_tracker.record_saving_time("cpu", save_end - save_start)

    return results, stats_tracker
