import argparse
from helper.multiplication_saver import start_memory_logger, stop_memory_logger
from utils.utils import load_matrices, merge_and_permute_sets
from multiplicators.cpu_multiplicator import multiply_on_cpu
from multiplicators.gpu_multiplicator import multiply_on_gpu
from multiplicators.hybrid_pipeline import multiply_hybrid
from data.stats_tracker import StatsTracker
from data.matrix_result_plotter import plot_avg_times, plot_memory

def main():
    parser = argparse.ArgumentParser(description="Matrix Multiplication App")
    parser.add_argument("--mode", choices=["cpu", "gpu", "hybrid"], required=True, help="Multiplication strategy")
    parser.add_argument("--files", nargs="+", required=True, help="Paths to .npy or .npz files")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size (hybrid only)")
    args = parser.parse_args()

    print("🔢 Loading matrix sets...")
    sets = [load_matrices(path) for path in args.files]
    pairs = merge_and_permute_sets(*sets)

    stats = StatsTracker()

    if args.mode == "cpu":
        cpu_memory_log, cpu_mem_thread, cpu_stop_mem_flag = start_memory_logger(interval=0.5)
        results, stats = multiply_on_cpu(pairs, stats)
        stats.stop()
        stop_memory_logger(cpu_mem_thread, cpu_stop_mem_flag)
        plot_memory(cpu_memory_log, [], filename="results/cpu_memory.png")

    elif args.mode == "gpu":
        gpu_memory_log, gpu_mem_thread, gpu_stop_mem_flag = start_memory_logger(interval=0.5)
        stats = multiply_on_gpu(pairs, stats)
        stats.stop()
        stop_memory_logger(gpu_mem_thread, gpu_stop_mem_flag)
        plot_memory([], gpu_memory_log, filename="results/gpu_memory.png")
    elif args.mode == "hybrid":
        cpu_memory_log, cpu_mem_thread, cpu_stop_mem_flag = start_memory_logger(interval=0.5)
        gpu_memory_log, gpu_mem_thread, gpu_stop_mem_flag = start_memory_logger(interval=0.5)
        stats = multiply_hybrid(pairs, stats)
        stats.stop()
        stop_memory_logger(cpu_mem_thread, cpu_stop_mem_flag)
        stop_memory_logger(gpu_mem_thread, gpu_stop_mem_flag)
        plot_memory(cpu_memory_log, gpu_memory_log, filename="results/hybrid_memory.png")

    stats.print_summary()
    stats.export_csv()
    plot_avg_times(stats.get_data_for_plot())

if __name__ == "__main__":
    main()
