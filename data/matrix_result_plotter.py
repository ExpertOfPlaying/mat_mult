import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)

def plot_avg_times(data_dict, filename="results/avg_times.png"):
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
    print(f"[Plot] Saved average time chart as {filename}")

def plot_memory(cpu_log, gpu_log, filename="results/memory_usage.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(cpu_log, label="CPU Memory (GB)", color="blue")
    plt.plot(gpu_log, label="GPU Memory (GB)", color="orange")
    plt.xlabel("Checkpoint")
    plt.ylabel("Memory Usage (GB)")
    plt.title("Memory Usage Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[Plot] Saved memory usage chart as {filename}")
