from concurrent.futures import ProcessPoolExecutor
from helper.dot_wrappers import safe_cpu_dot
from utils.utils import timeit
from data.stats_tracker import StatsTracker
from time import perf_counter
from helper.multiplication_saver import save_results_to_disk

def process_pair_and_time(pair):
    """Used by ProcessPoolExecutor, with timing and error handling."""
    a, b = pair
    try:
        start = perf_counter()
        result = safe_cpu_dot(a, b)
        end = perf_counter()
        duration = end - start
        return result, None, duration
    except Exception as e:
        return None, f"Unexpected: {e}", 0.0

@timeit
def multiply_on_cpu(pairs, stats_tracker: StatsTracker = None):
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    results = []

    with ProcessPoolExecutor() as executor:
        for result, error, duration in executor.map(process_pair_and_time, pairs):
            if error is None and result is not None:
                results.append(result)
                stats_tracker.record_timing("cpu", duration)
            else:
                print(f"[CPU] Error during multiplication: {error}")

    # Time and record the saving step
    save_start = perf_counter()
    save_results_to_disk("cpu_results.npy", results)
    save_end = perf_counter()
    stats_tracker.record_saving_time("cpu", save_end - save_start)

    return results, stats_tracker
