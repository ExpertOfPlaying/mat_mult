from typing import Literal, List
import cupy as cp
import numpy as np
import threading
import os

MatrixType = Literal["random", "identity", "triangular"]

def generate_matrices(matrix_type: MatrixType, count: int, min_rows: int, max_rows: int, min_cols: int = None,
                      max_cols: int = None, low: float = 0.0, high: float = 1.0, upper: bool = True) \
        -> List[cp.ndarray]:
    """
    Generate matrices of a specific type with random shape and values.
    - 'random': random values, random shape
    - 'identity': identity matrices with random size (square only)
    - 'triangular': upper or lower triangular with random size (square only)
    """
    matrices = []

    for _ in range(count):
        if matrix_type == "random":
            rows = np.random.randint(min_rows, max_rows + 1)
            cols = np.random.randint(min_cols or min_rows, max_cols or max_rows + 1)
            mat = cp.random.uniform(low, high, size=(rows, cols)).astype(cp.float32)
            matrices.append(mat)

        elif matrix_type == "identity":
            size = np.random.randint(min_rows, max_rows + 1)
            matrices.append(cp.eye(size, dtype=cp.float32))

        elif matrix_type == "triangular":
            size = np.random.randint(min_rows, max_rows + 1)
            mat = cp.random.uniform(low, high, size=(size, size)).astype(cp.float32)
            triangular = cp.triu(mat) if upper else cp.tril(mat)
            matrices.append(triangular)

        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")

    return matrices


def convert_to_numpy(cupy_matrices: List[cp.ndarray]) -> List[np.ndarray]:
    return [cp.asnumpy(m) for m in cupy_matrices]


def save_as_npy(filename: str, matrices: List[np.ndarray]):
    os.makedirs("", exist_ok=True)
    full_path = os.path.join("", filename)

    if not matrices:
        print(f"No matrices to save for {filename}")
        return
    if all(m.shape == matrices[0].shape for m in matrices):
        # Uniform shape → stack and save
        stacked = np.stack(matrices, axis=0)
        np.save(full_path, stacked)
        print(f"Saved {full_path} (stacked) with shape {stacked.shape}")
    else:
        # Variable shapes → save as dict with numbered keys
        save_dict = {f"matrix_{i}": m for i, m in enumerate(matrices)}
        np.savez(full_path.replace(".npy", ".npz"), **save_dict)
        print(f"Saved {full_path.replace('.npy', '.npz')} with {len(matrices)} variable-shaped matrices")

def async_save_as_npy(filename: str, matrices: List[np.ndarray]):
    matrices_copy = matrices.copy()
    thread = threading.Thread(target=save_as_npy, args=(filename, matrices_copy))
    thread.start()
    return thread

def generate_and_save_all(per_type_count=1000, batch_size=250, min_rows=1000, max_rows=1200, min_cols=1000,
                          max_cols=1200, low=-10, high=10):
    all_matrices_cpu = []
    save_threads = []

    def generate_in_batches(matrix_type: Literal["random", "identity", "triangular"], filename: str, **kwargs):
        matrices_cpu = []

        for i in range(0, per_type_count, batch_size):
            count = min(batch_size, per_type_count - i)
            print(f"Generating batch: {matrix_type} [{i}–{i+count}]")

            matrices_gpu = generate_matrices(
                matrix_type,
                count=count,
                min_rows=min_rows,
                max_rows=max_rows,
                min_cols=min_cols,
                max_cols=max_cols,
                low=low,
                high=high,
                **kwargs
            )
            batch_cpu = convert_to_numpy(matrices_gpu)
            matrices_cpu.extend(batch_cpu)
            all_matrices_cpu.extend(batch_cpu)
            cp.get_default_memory_pool().free_all_blocks()

        save_threads.append(async_save_as_npy(filename, matrices_cpu))
        matrices_cpu.clear()

    # Individual sets
    generate_in_batches("random", "random_matrices.npy")
    generate_in_batches("identity", "identity_matrices.npy")
    generate_in_batches("triangular", "triangular_matrices.npy", upper=True)

    for thread in save_threads:
        thread.join()

    # Save all combined
    print("Saving all matrices together...")
    save_as_npy("all_matrices.npy", all_matrices_cpu)


if __name__ == "__main__":
    generate_and_save_all(
        per_type_count=500,
        batch_size=250,
        min_rows=1000,
        max_rows=1500,
        min_cols=1000,
        max_cols=1500,
        low=-50,
        high=50
    )