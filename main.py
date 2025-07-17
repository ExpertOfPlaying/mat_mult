"""Main entry point for MatrixMultiplicator.

Supports CPU, GPU, or hybrid matrix multiplication. Tracks stats and memory usage,
and plots results.

Usage:
    python main.py --mode [cpu|gpu|hybrid] --files set1.npy set2.npy ...
"""

import argparse
import logging
import sys
from pathlib import Path

from helper.multiplication_saver import start_memory_logger, stop_memory_logger
from utils.utils import load_matrices, merge_and_permute_sets
from multiplicators.cpu_multiplicator import multiply_on_cpu
from multiplicators.gpu_multiplicator import multiply_on_gpu
from multiplicators.hybrid_pipeline import multiply_hybrid
from data.stats_tracker import StatsTracker
from data.matrix_result_plotter import plot_avg_times, plot_memory

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main() -> None:
    """
    Orchestrate the matrix multiplication workflow.
    Parses command-line arguments, loads matrices, runs the chosen multiplication strategy,
    tracks stats, and generates result plots.
    """
    parser = argparse.ArgumentParser(description="Matrix Multiplication App")
    parser.add_argument(
        "--mode", choices=["cpu", "gpu", "hybrid"], required=True,
        help="Multiplication strategy to use"
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Paths to .npy or .npz files (matrix sets)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Batch size for hybrid mode (ignored for cpu/gpu)"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    logging.info("🔢 Loading matrix sets...")
    try:
        sets = [load_matrices(path) for path in args.files]
    except Exception as e:
        logging.error(f"Error loading matrices: {e}")
        sys.exit(1)

    try:
        pairs = merge_and_permute_sets(*sets)
    except Exception as e:
        logging.error(f"Error merging and permuting sets: {e}")
        sys.exit(1)

    stats = StatsTracker()

    try:
        if args.mode == "cpu":
            cpu_log, _, cpu_thread, _, stop_flag = start_memory_logger(interval=0.5, mode='cpu')
            results, stats = multiply_on_cpu(pairs, stats)
            stats.stop()
            stop_memory_logger(cpu_thread, None, stop_flag)
            plot_memory(cpu_log, [], filename=str(output_dir / "cpu_memory.png"))
            stats.print_summary()

        elif args.mode == "gpu":
            _, gpu_log, cpu_thread, gpu_thread, stop_flag = start_memory_logger(interval=0.5, mode='gpu')
            results, stats = multiply_on_gpu(pairs, stats)
            stats.stop()
            stop_memory_logger(cpu_thread, gpu_thread, stop_flag)
            plot_memory([], gpu_log, filename=str(output_dir / "gpu_memory.png"))
            stats.print_summary()

        elif args.mode == "hybrid":
            cpu_log, gpu_log, cpu_thread, gpu_thread, stop_flag = start_memory_logger(interval=0.5, mode='both')
            results, stats = multiply_hybrid(pairs, stats, batch_size=args.batch_size)
            stats.stop()
            stop_memory_logger(cpu_thread, gpu_thread, stop_flag)
            plot_memory(cpu_log, gpu_log, filename=str(output_dir / "hybrid_memory.png"))
            stats.print_summary()

        else:
            logging.error("Unknown mode selected!")
            sys.exit(1)

        # Plot average times for all modes
        plot_avg_times(stats.get_data_for_plot(), filename=str(output_dir / f"{args.mode}_avg_times.png"))
        stats.export_csv(str(output_dir / f"{args.mode}_stats.csv"))
        logging.info(f"✅ {args.mode.upper()} run complete. Results and plots saved in '{output_dir}'.")

    except Exception as e:
        logging.exception(f"An error occurred during matrix multiplication workflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()