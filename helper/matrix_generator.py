"""Matrix generator utilities for random, identity, and triangular matrices (CPU & GPU).

Supports batch generation, async saving, and output in .npy/.npz format.
"""

import os
import threading
import logging
from typing import Literal, List, Optional

import numpy as np
import cupy as cp

MatrixType = Literal["random", "identity", "triangular"]

logger = logging.getLogger(__name__)

def generate_matrices(
    matrix_type: MatrixType,
    count: int,
    min_rows: int,
    max_rows: int,
    min_cols: Optional[int] = None,
    max_cols: Optional[int] = None,
    low: float = 0.0,
    high: float = 1.0,
    upper: bool = True
) -> List[cp.ndarray]:
    """
    Generate a list of matrices of the specified type, shape, and value range.

    Args:
        matrix_type (MatrixType): Type ('random', 'identity', or 'triangular').
        count (int): Number of matrices to generate.
        min_rows (int): Minimum rows (for all types).
        max_rows (int): Maximum rows (for all types).
        min_cols (int, optional): Minimum columns (random only). Defaults to min_rows.
        max_cols (int, optional): Maximum columns (random only). Defaults to max_rows.
        low (float, optional): Lower bound for random values. Defaults to 0.0.
        high (float, optional): Upper bound for random values. Defaults to 1.0.
        upper (bool, optional): Use upper or lower triangle for 'triangular'. Defaults to True.

    Returns:
        List[cp.ndarray]: List of generated matrices.
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
            logger.error(f"Unsupported matrix type: {matrix_type}")
            raise ValueError(f"Unsupported matrix type: {matrix_type}")

    return matrices


def convert_to_numpy(cupy_matrices: List[cp.ndarray]) -> List[np.ndarray]:
    """Convert a list of CuPy matrices to NumPy arrays (CPU)."""
    return [cp.asnumpy(m) for m in cupy_matrices]


def save_as_npy(filename: str, matrices: List[np.ndarray], out_dir: str = "") -> None:
    """
    Save matrices as .npy (if shapes uniform) or .npz (if variable shapes).

    Args:
        filename (str): Output file name.
        matrices (List[np.ndarray]): List of matrices to save.
        out_dir (str): Output directory (optional).
    """
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)

    if not matrices:
        logger.warning(f"No matrices to save for {filename}")
        return

    if all(m.shape == matrices[0].shape for m in matrices):
        # Uniform shape → stack and save
        stacked = np.stack(matrices, axis=0)
        np.save(full_path, stacked)
        logger.info(f"Saved {full_path} (stacked) with shape {stacked.shape}")
    else:
        # Variable shapes → save as dict with numbered keys
        save_dict = {f"matrix_{i}": m for i, m in enumerate(matrices)}
        np.savez(full_path.replace(".npy", ".npz"), **save_dict)
        logger.info(f"Saved {full_path.replace('.npy', '.npz')} with {len(matrices)} variable-shaped matrices")

def async_save_as_npy(filename: str, matrices: List[np.ndarray], out_dir: str = "") -> threading.Thread:
    """
    Save matrices asynchronously in a background thread.

    Args:
        filename (str): Output file name.
        matrices (List[np.ndarray]): List of matrices to save.
        out_dir (str): Output directory (optional).

    Returns:
        threading.Thread: The thread object for joining/waiting.
    """
    matrices_copy = matrices.copy()
    thread = threading.Thread(target=save_as_npy, args=(filename, matrices_copy, out_dir))
    thread.start()
    return thread

def generate_and_save_all(
    per_type_count: int = 1000,
    batch_size: int = 250,
    min_rows: int = 1000,
    max_rows: int = 1200,
    min_cols: int = 1000,
    max_cols: int = 1200,
    low: float = -10,
    high: float = 10,
    out_dir: str = "matrix_sets"
) -> None:
    """
    Generate and save random, identity, and triangular matrices in batches, saving to disk.

    Args:
        per_type_count (int): Total count per matrix type.
        batch_size (int): Matrices per batch (for memory efficiency).
        min_rows (int): Minimum rows (for all types).
        max_rows (int): Maximum rows (for all types).
        min_cols (int, optional): Minimum columns (random only). Defaults to min_rows.
        max_cols (int, optional): Maximum columns (random only). Defaults to max_rows.
        low (float, optional): Lower bound for random values. Defaults to 0.0.
        high (float, optional): Upper bound for random values. Defaults to 1.0.
        out_dir (str): Directory to save all matrix files.
    """
    all_matrices_cpu: List[np.ndarray] = []
    save_threads: List[threading.Thread] = []

    def generate_in_batches(matrix_type: MatrixType, filename: str, **kwargs) -> None:
        matrices_cpu: List[np.ndarray] = []

        for i in range(0, per_type_count, batch_size):
            count = min(batch_size, per_type_count - i)
            logger.info(f"Generating batch: {matrix_type} [{i}–{i+count}]")

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

        save_threads.append(async_save_as_npy(filename, matrices_cpu, out_dir))
        matrices_cpu.clear()

    os.makedirs(out_dir, exist_ok=True)

    # Individual sets
    generate_in_batches("random", "random_matrices.npy")
    generate_in_batches("identity", "identity_matrices.npy")
    generate_in_batches("triangular", "triangular_matrices.npy", upper=True)

    for thread in save_threads:
        thread.join()

    # Save all combined
    logger.info("Saving all matrices together...")
    save_as_npy("all_matrices.npy", all_matrices_cpu, out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_and_save_all(
        per_type_count=500,
        batch_size=250,
        min_rows=1000,
        max_rows=1500,
        min_cols=1000,
        max_cols=1500,
        low=-50,
        high=50,
        out_dir="matrix_sets"
    )
