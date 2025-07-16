class IncompatibleMatrixShape(Exception):
    """Raised when two matrices cannot be multiplied due to incompatible shapes."""
    def __init__(self, shape_a, shape_b):
        self.shape_a = shape_a
        self.shape_b = shape_b
        super().__init__(f"Incompatible shapes for multiplication: {self.shape_a} x {self.shape_b}")