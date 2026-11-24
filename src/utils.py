from dataclasses import dataclass, field
import numpy as np


@dataclass
class Result:
    x: np.ndarray
    k: int
    residual: float
    success: bool
    message: str
    size: int | None = None
    q: float | None = None
    status: str = ""
    seed: int | None = None
    q_actual: float | None = None
    rho: float | None = None
    final_error: float | None = None
    history: list[tuple[int, float, float | None]] = field(default_factory=list)
    family: str = "dense"
    neighbors: int | None = None
    density: float | None = None
    sparsity: float | None = None


def real_array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real numbers.")
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers.")
    return array


def square_matrix(A) -> np.ndarray:
    A = real_array(A, "A")
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a nonempty square matrix.")
    return A
