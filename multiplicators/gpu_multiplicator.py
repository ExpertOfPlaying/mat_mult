from time import perf_counter
from helper.dot_wrappers import safe_gpu_dot
from utils.utils import timeit
from helper.multiplication_saver import save_results_to_disk
from data.stats_tracker import StatsTracker

@timeit
def multiply_on_gpu(pairs, stats_tracker: StatsTracker = None):
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    results = []

    for a, b in pairs:
        try:
            start = perf_counter()
            result = safe_gpu_dot(a, b)
            end = perf_counter()

            results.append(result)
            stats_tracker.record_timing("gpu", end - start)
        except Exception as e:
            print(f"[GPU] Error during multiplication: {e}")

    save_start = perf_counter()
    save_results_to_disk("gpu_results.npy", results)
    save_end = perf_counter()
    stats_tracker.record_saving_time("gpu", save_end - save_start)

    stats_tracker.stop()
    return stats_tracker