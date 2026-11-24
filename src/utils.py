import numpy as np

class Result:
    def __init__(
        self,
        x: np.array,
        k: int,
        residual: float,
        success: bool,
        message: str,
        size: int = None,
        q: float = None
    ):
        self.x = x
        self.k = k
        self.size = size
        self.q = q
        self.residual = residual
        self.success = success
        self.message = message