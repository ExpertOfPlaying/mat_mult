import time
import pandas as pd

class StatsTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.timings = []
        self.saving_times = {}

    def record_timing(self, strategy: str, duration: float):
        self.timings.append((strategy, duration))

    def record_saving_time(self, strategy: str, duration: float):
        self.saving_times[strategy] = self.saving_times.get(strategy, 0.0) + duration

    def stop(self):
        self.end_time = time.perf_counter()

    def export_csv(self, filename="results/stats.csv"):
        data = []
        strategies = set(s for s, _ in self.timings)

        for strategy in strategies:
            multiplications = [d for s, d in self.timings if s == strategy]
            count = len(multiplications)
            total_mult_time = sum(multiplications)
            avg_mult_time = total_mult_time / count if count else 0.0
            save_time = self.saving_times.get(strategy, 0.0)
            total_time = total_mult_time + save_time

            data.append({
                "strategy": strategy,
                "count": count,
                "avg_time_per_multiplication": round(avg_mult_time, 6),
                "total_multiplication_time": round(total_mult_time, 6),
                "total_saving_time": round(save_time, 6),
                "total_runtime": round(total_time, 6)
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"[Stats] Exported stats to {filename}")

    def print_summary(self):
        total_time = (self.end_time or time.perf_counter()) - self.start_time
        print(f"\n=== Multiplication Summary ===")
        print(f"Total runtime: {total_time:.2f} seconds")
        for strategy in set(s for s, _ in self.timings):
            mults = [d for s, d in self.timings if s == strategy]
            save_time = self.saving_times.get(strategy, 0.0)
            print(f"{strategy.upper()}: {len(mults)} multiplications, "
                  f"{save_time:.2f}s saving, {sum(mults):.2f}s multiplying")

    def get_data_for_plot(self):
        strategies = list(set(s for s, _ in self.timings))
        data = {
            "strategies": [],
            "multiplication": [],
            "saving": [],
            "total": []
        }
        for strategy in strategies:
            mult = sum(d for s, d in self.timings if s == strategy)
            save = self.saving_times.get(strategy, 0.0)
            total = mult + save
            data["strategies"].append(strategy)
            data["multiplication"].append(mult)
            data["saving"].append(save)
            data["total"].append(total)
        return data