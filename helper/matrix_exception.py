"""Custom exception for matrix multiplication errors."""

from typing import Tuple

class IncompatibleMatrixShape(Exception):
    """
    Raised when two matrices cannot be multiplied due to incompatible shapes.

    Attributes:
        shape_a (Tuple[int, ...]): Shape of the first matrix.
        shape_b (Tuple[int, ...]): Shape of the second matrix.
    """
    def __init__(self, shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> None:
        self.shape_a = shape_a
        self.shape_b = shape_b
        super().__init__(self.message)

    @property
    def message(self) -> str:
        """Detailed error message."""
        return f"Incompatible shapes for multiplication: {self.shape_a} x {self.shape_b}"

    def __str__(self) -> str:
        return self.message
