import numpy as np
import pandas as pd
from utils import Result


sizes = [3, 5, 10, 20, 50, 100, 131, 277, 512, 1000, 1331]
q_experiments = np.linspace(0.1, 1.0, 10)


def apply_diagonal(A: np.ndarray, q: float) -> np.ndarray:
    matrix = A.copy()
    np.fill_diagonal(matrix, 0)
    s = np.sum(np.abs(matrix), axis=1)
    diag = np.diag([s_i / q for s_i in s])
    return matrix + diag


def accuracy(x_exact: np.array, x_approx: np.array) -> float:
    return np.linalg.norm(x_exact - x_approx)


def solve_gauss_seidel(
    A: np.ndarray,
    b: np.array,
    x0: np.array,
    tol: float = 1e-8,
    max_iteration: int = 10000
) -> np.array:
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    
    if A.shape[0] != x0.shape[0]:
        raise ValueError("The size of initial guess x0 must match the number of rows in matrix A.")

    if A.shape[0] != b.shape[0]:
        raise ValueError("The size of vector b must match the number of rows in matrix A.")
    
    if any(np.diag(A) == 0):
        raise ValueError("Matrix A must not have zero on its diagonal.")

    n = len(x0)
    x = x0.copy()
    scale = max(1, np.linalg.norm(b))
    residual = np.linalg.norm(b - A @ x) / scale
    
    if residual < tol:
        return Result(x, 0, residual, True, "Initial guess is already within tolerance.")
    
    for k in range(max_iteration):
        x_old = x.copy()
        
        for i in range(n):
            sum_new = 0
            for j in range(i):
                sum_new += A[i, j] * x[j]
                
            sum_old = 0
            for j in range(i + 1, n):
                sum_old += A[i, j] * x_old[j]
                
            x[i] = (b[i] - sum_new - sum_old) / A[i, i]
            
        steps = k + 1
        if not all(np.isfinite(x)):
            return Result(x, steps, np.nan, False, "Numeric error: Non-finite values encountered.")
        
        residual = np.linalg.norm(b - A @ x) / scale
        
        if not np.isfinite(residual):
            return Result(x, steps, np.nan, False, "Numeric error: Non-finite residual encountered.")
        
        if residual < tol:
            return Result(x, steps, residual, True, "Converged successfully.")
        
    return Result(x, max_iteration, residual, False, "Maximum iterations reached without convergence.")



def run_experiments_on_size(size: int) -> list[Result]:
    x_exact = np.ones(size)
    x0 = np.zeros(size)
    
    results = []
    
    for q in q_experiments:
        print(f"   Running experiment for size {size} with q={q:.2f}...")
        rng = np.random.default_rng(seed=42)
        A = rng.uniform(-1, 1, size=(size, size))
        A = apply_diagonal(A, q)
        b = A @ x_exact
        
        result = solve_gauss_seidel(A, b, x0)
        result.size = size
        result.q = q
        results.append(result)
        
    return results


def main():
    all_results = []
    
    for size in sizes:
        print(f"Running experiments for size: {size}...")
        results = run_experiments_on_size(size)
        all_results.extend(results)
    
    
if __name__ == "__main__":
    main()