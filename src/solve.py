from utils import Result, real_array, square_matrix
import numpy as np


def iteration_matrix(A: np.ndarray) -> np.ndarray:
    """GS iteration matrix associated with A (B_GS), from equation B_GS = -(D + L)^-1 * U."""
    A = square_matrix(A)
    if np.any(np.diag(A) == 0):
        raise ValueError("A must not have zero diagonal entries.")
    return np.linalg.solve(np.tril(A), -np.triu(A, k=1))


def spectral_radius(A: np.ndarray) -> float:
    """Spectral radius of the GS iteration matrix associated with A."""
    return float(np.max(np.abs(np.linalg.eigvals(iteration_matrix(A)))))


def error_norm(x_exact: np.ndarray, x_approx: np.ndarray) -> float:
    """Infinity norm of the error vector, ||x_exact - x_approx||_inf."""
    exact = real_array(x_exact, "x_exact")
    approx = real_array(x_approx, "x_approx")
    if exact.ndim != 1 or exact.size == 0 or exact.shape != approx.shape:
        raise ValueError("Solution vectors must have the same nonempty 1D shape.")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.linalg.norm(exact - approx, ord=np.inf))


def solve_gauss_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iteration: int = 10000,
    *,
    x_exact: np.ndarray | None = None,
) -> Result:
    """Forward GS, stopped by ||b-Ax||_inf / max(1, ||b||_inf) <= tol.

    History includes iteration zero. x_exact is used only to measure error,
    never to update the iterate or decide when to stop.
    """
    A = square_matrix(A)
    b = real_array(b, "b")
    x0 = real_array(x0, "x0")
    n = A.shape[0]
    if b.shape != (n,) or x0.shape != (n,):
        raise ValueError("b and x0 must be 1D vectors of length n.")
    if np.any(np.diag(A) == 0):
        raise ValueError("A must not have zero diagonal entries.")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive.")
    if isinstance(max_iteration, (bool, np.bool_)) or not isinstance(max_iteration, (int, np.integer)) or max_iteration < 1:
        raise ValueError("max_iteration must be a positive integer.")
    if x_exact is not None:
        x_exact = real_array(x_exact, "x_exact")
        if x_exact.shape != (n,):
            raise ValueError("x_exact must be a 1D vector of length n.")

    x = x0.copy()
    scale = max(1.0, float(np.linalg.norm(b, ord=np.inf)))
    history = []

    def measure():
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = float(np.linalg.norm(b - A @ x, ord=np.inf) / scale)
            error = None if x_exact is None else float(np.linalg.norm(x - x_exact, ord=np.inf))
        return residual, error

    def finish(k, residual, error, status, message):
        return Result(
            x=x.copy(), k=k, residual=residual, success=status == "converged",
            message=message, size=n, status=status, final_error=error, history=history,
        )

    residual, error = measure()
    history.append((0, residual, error))
    if not np.isfinite(residual):
        return finish(0, residual, error, "numerical_error", "Non-finite initial residual.")
    if residual <= tol:
        return finish(0, residual, error, "converged", "Initial guess meets the residual tolerance.")

    for k in range(max_iteration):
        x_old = x.copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i in range(n):
                # Same row-wise GS formula as the pseudocode; use dot products
                # for the two sums to avoid slow nested Python loops.
                sum_new = A[i, :i] @ x[:i]
                sum_old = A[i, i + 1:] @ x_old[i + 1:]
                x[i] = (b[i] - sum_new - sum_old) / A[i, i]

        steps = k + 1
        if not np.all(np.isfinite(x)):
            history.append((steps, float("nan"), None))
            return finish(steps, float("nan"), None, "numerical_error", "Non-finite iterate.")
        residual, error = measure()
        history.append((steps, residual, error))
        if not np.isfinite(residual):
            return finish(steps, residual, error, "numerical_error", "Non-finite residual.")
        if residual <= tol:
            return finish(steps, residual, error, "converged", "Residual tolerance reached.")

    return finish(max_iteration, residual, error, "max_iterations", "Iteration budget exhausted; this alone does not prove divergence.")
