"""Plotting utilities for matrix multiplication results and resource usage.

Generates bar charts for average timings and line plots for CPU/GPU memory usage over time.
"""

import os
import logging
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def plot_avg_times(
    data_dict: Dict[str, List[float]],
    filename: str = "results/avg_times.png"
) -> None:
    """
    Plot average multiplication, saving, and total times for each strategy.

    Args:
        data_dict (dict): Keys: 'strategies', 'multiplication', 'saving', 'total'
        filename (str): Output image file path.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    strategies = data_dict["strategies"]
    width = 0.25
    x = np.arange(len(strategies))

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, data_dict["multiplication"], width, label="Multiplication")
    plt.bar(x, data_dict["saving"], width, label="Saving")
    plt.bar(x + width, data_dict["total"], width, label="Total")

    plt.xticks(x, strategies)
    plt.ylabel("Time (s)")
    plt.title("Total Time per Strategy (Multiplication vs Saving)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"[Plot] Saved average time chart as {filename}")

def plot_memory(
    cpu_log: Optional[List[float]],
    gpu_log: Optional[List[float]],
    filename: str = "results/memory_usage.png"
) -> None:
    """
    Plot CPU and/or GPU memory usage over time.

    Args:
        cpu_log (list): CPU memory usage history (GB).
        gpu_log (list): GPU memory usage history (GB).
        filename (str): Output image file path.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(10, 5))
    have_any = False
    if cpu_log:
        plt.plot(cpu_log, label="CPU Memory (GB)", color="blue")
        have_any = True
    if gpu_log:
        plt.plot(gpu_log, label="GPU Memory (GB)", color="orange")
        have_any = True

    if not have_any:
        logger.warning("[Plot] No memory logs to plot.")
        plt.close()
        return

    plt.xlabel("Checkpoint")
    plt.ylabel("Memory Usage (GB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logger.info(f"[Plot] Saved memory usage chart as {filename}")
