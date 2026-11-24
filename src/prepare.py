from dataclasses import dataclass
from typing import Iterator
from utils import square_matrix
import numpy as np


DEFAULT_SIZES = [3, 16, 24, 50, 100]
DEFAULT_Q_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0, 1.05, 1.2, 1.8]
DEFAULT_SEEDS = list(range(42, 62))
DEFAULT_NEIGHBORS = [2, 4, 8]


@dataclass
class ExperimentCase:
    size: int
    seed: int
    q: float
    A: np.ndarray
    b: np.ndarray
    x_exact: np.ndarray
    x0: np.ndarray
    family: str = "dense"
    neighbors: int = 0
    density: float = 1.0
    sparsity: float = 0.0


def apply_diagonal(A: np.ndarray, q: float) -> np.ndarray:
    """Return a new matrix with positive diagonal and row ratios exactly q."""
    A = square_matrix(A)
    if not np.isscalar(q) or not np.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and strictly positive.")
    matrix = A.copy()
    np.fill_diagonal(matrix, 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        row_sums = np.abs(matrix).sum(axis=1)
        diagonal = row_sums / q
    if np.any(row_sums == 0):
        raise ValueError("Every row must have a nonzero off-diagonal entry.")
    if np.any(diagonal == 0) or not np.all(np.isfinite(diagonal)):
        raise ValueError("The requested diagonal is not representable as finite nonzero floats.")
    np.fill_diagonal(matrix, diagonal)
    return matrix


def diagonal_dominance_index(A: np.ndarray) -> float:
    A = square_matrix(A)
    diagonal = np.abs(np.diag(A))
    if np.any(diagonal == 0):
        raise ValueError("A must not have zero diagonal entries.")
    off_diagonal = A.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return float(np.max(np.abs(off_diagonal).sum(axis=1) / diagonal))


def prepare_experiments(
    size: int, *, q_values=None, seeds=None, neighbors=None,
) -> Iterator[ExperimentCase]:
    """Yield input systems, without solving or plotting them.

    Always include the original dense baseline. Optional neighbors adds
    row-sparse families with exactly k off-diagonal nonzeros per row.
    Supports are nested across k and fixed across q; retained weights are
    exactly those of the dense baseline. Each case owns its arrays.
    """
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 2:
        raise ValueError("Experiment size must be an integer >= 2.")
    q_values = list(DEFAULT_Q_VALUES if q_values is None else q_values)
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    neighbors = list([] if neighbors is None else neighbors)
    if not q_values or not seeds:
        raise ValueError("Provide at least one q value and one seed.")
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in neighbors):
        raise ValueError("neighbors must contain positive integers.")
    if len(set(neighbors)) != len(neighbors):
        raise ValueError("neighbors must not contain duplicates.")
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Seeds must be nonnegative integers.")
        rng = np.random.default_rng(seed)
        base = rng.uniform(-1.0, 1.0, size=(size, size))
        np.fill_diagonal(base, 0.0)
        # Separate RNG stream: adding sparse families cannot change dense data.
        # One permutation per row makes supports nested when k increases.
        mask_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(size), 9173]))
        order = np.empty((size, size - 1), dtype=int)
        for i in range(size):
            order[i] = mask_rng.permutation(np.flatnonzero(np.arange(size) != i))
        families = [("dense", size - 1)]
        families.extend((f"sparse_k{k}", int(k)) for k in neighbors if k < size - 1)
        for family, k in families:
            if family == "dense":
                off_diagonal = base
            else:
                off_diagonal = np.zeros_like(base)
                rows = np.arange(size)[:, None]
                columns = order[:, :k]
                off_diagonal[rows, columns] = base[rows, columns]
            counts = np.count_nonzero(off_diagonal, axis=1)
            if not np.all(counts == k):
                raise ValueError("A sampled weight is exactly zero; choose another seed to preserve exact row degree.")
            density = k / (size - 1)
            for q in q_values:
                A = apply_diagonal(off_diagonal, q)
                x_exact, x0 = np.ones(size), np.zeros(size)
                yield ExperimentCase(
                    size, int(seed), float(q), A, A @ x_exact, x_exact, x0,
                    family, k, density, 1.0 - density,
                )
