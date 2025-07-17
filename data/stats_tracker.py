"""Statistics tracker for matrix multiplication runs.

Tracks timings, saving times, and exports summaries for CPU, GPU, and hybrid strategies.
"""

import time
import logging
from typing import Optional, Dict, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class StatsTracker:
    """
    Tracks and reports statistics for different matrix multiplication strategies.

    Attributes:
        start_time (float): Time when tracking started.
        end_time (Optional[float]): Time when tracking stopped.
        timings (List[Tuple[str, float]]): List of (strategy, duration) for each multiplication.
        saving_times (Dict[str, float]): Total saving time per strategy.
    """
    def __init__(self) -> None:
        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None
        self.timings: List[Tuple[str, float]] = []
        self.saving_times: Dict[str, float] = {}
        self.merging_times: dict = {}

    def record_timing(self, strategy: str, duration: float) -> None:
        """Record the time taken for one multiplication."""
        self.timings.append((strategy, duration))

    def record_saving_time(self, strategy: str, duration: float) -> None:
        """Add saving time for the given strategy."""
        self.saving_times[strategy] = self.saving_times.get(strategy, 0.0) + duration

    def record_merging_time(self, strategy: str, duration: float) -> None:
        self.merging_times[strategy] = self.merging_times.get(strategy, 0.0) + duration

    def stop(self) -> None:
        """Mark the end of a run."""
        self.end_time = time.perf_counter()

    def export_csv(self, filename: str = "results/stats.csv") -> None:
        """
        Export all statistics as a CSV file.

        Args:
            filename (str): Output CSV file path.
        """
        strategies = set(s for s, _ in self.timings)
        data = []
        for strategy in strategies:
            multiplications = [d for s, d in self.timings if s == strategy]
            count = len(multiplications)
            total_mult_time = sum(multiplications)
            avg_mult_time = total_mult_time / count if count else 0.0
            save_time = self.saving_times.get(strategy, 0.0)
            merge_time = self.merging_times.get(strategy, 0.0)
            total_time = total_mult_time + save_time + merge_time

            data.append({
                "strategy": strategy,
                "count": count,
                "avg_time_per_multiplication": round(avg_mult_time, 6),
                "total_multiplication_time": round(total_mult_time, 6),
                "total_saving_time": round(save_time, 6),
                "total_merging_time": round(merge_time, 6),
                "total_runtime": round(total_time, 6)
            })

        if not data:
            logger.warning("[Stats] No data to export to CSV.")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"[Stats] Exported stats to {filename}")

    def print_summary(self) -> None:
        """
        Print a human-readable summary of run statistics.
        """
        logger.info("\n" + "=" * 40)
        logger.info(f"=== Multiplication Summary ===")
        for strategy in set(s for s, _ in self.timings):
            mults = [d for s, d in self.timings if s == strategy]
            save_time = self.saving_times.get(strategy, 0.0)
            merge_time = self.merging_times.get(strategy, 0.0)
            logger.info(f"{strategy.upper()}: {len(mults)} multiplications, "
                        f"{save_time:.2f}s saving, "
                        f"{merge_time:.2f}s merging, "
                        f"{sum(mults):.2f}s multiplying")
        logger.info("=" * 40 + "\n")

    def get_data_for_plot(self) -> Dict[str, List]:
        """
        Prepare data for plotting (compatible with plot_avg_times).

        Returns:
            dict: { "strategies": [...], "multiplication": [...], "saving": [...], "merging": [...], "total": [...] }
        """
        strategies = list(set(s for s, _ in self.timings))
        data = {
            "strategies": [],
            "multiplication": [],
            "saving": [],
            "merging": [],
            "total": []
        }
        for strategy in strategies:
            mult = sum(d for s, d in self.timings if s == strategy)
            save = self.saving_times.get(strategy, 0.0)
            merge = self.merging_times.get(strategy, 0.0)
            total = mult + save
            data["strategies"].append(strategy)
            data["multiplication"].append(mult)
            data["saving"].append(save)
            data["merging"].append(merge)
            data["total"].append(total)
        return data

    def reset(self) -> None:
        """Reset all statistics for a new run."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.timings.clear()
        self.saving_times.clear()
