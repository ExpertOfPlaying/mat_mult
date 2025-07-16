import cupy as cp
from helper.dot_wrappers import safe_gpu_dot
from helper.multiplication_saver import start_saver, stop_saver #, save_merged_results
from time import perf_counter
from data.stats_tracker import StatsTracker


def multiply_hybrid(pairs, stats_tracker: StatsTracker = None, batch_size=50):
    if stats_tracker is None:
        stats_tracker = StatsTracker()

    print("[Hybrid] Starting GPU multiplication with CPU saving pipeline...")

    # Start the CPU saver thread to collect results
    queue, saver_thread, memory_log, batch_file_paths = start_saver(
        name_prefix="hybrid_result", batch_size=batch_size
    )

    # Start multiplying on GPU and send to CPU queue
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
            print(f"[Hybrid] Error during multiplication: {e}")

    stop_saver(queue, saver_thread)
    save_start = perf_counter()
    #save_merged_results("hybrid_results.npy", batch_file_paths)
    save_end = perf_counter()
    stats_tracker.record_saving_time("hybrid", save_end - save_start)

    return stats_tracker
