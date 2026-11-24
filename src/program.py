import argparse
import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import (
    DEFAULT_Q_VALUES, DEFAULT_SEEDS, DEFAULT_SIZES, DEFAULT_NEIGHBORS,
    diagonal_dominance_index, prepare_experiments,
)
from solve import solve_gauss_seidel, spectral_radius
from utils import Result
from visualization import save_plots, summarize


sizes = DEFAULT_SIZES
q_experiments = DEFAULT_Q_VALUES
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
RESULT_COLUMNS = [
    "n", "seed", "q_target", "q_actual", "rho", "iterations", "status",
    "success", "final_residual", "final_error", "message",
    "family", "neighbors", "density", "sparsity",
]
HISTORY_COLUMNS = ["n", "seed", "q_target", "iteration", "residual", "error", "family", "neighbors", "density", "sparsity"]


def run_experiments_on_size(
    size: int,
    *,
    q_values=None,
    seeds=None,
    neighbors=None,
    tol: float = 1e-8,
    max_iteration: int = 10000,
    verbose: bool = True,
) -> list[Result]:
    """Connect the prepared cases to the solver and attach run metadata."""
    cases = prepare_experiments(size, q_values=q_values, seeds=seeds, neighbors=neighbors)
    results = []
    for case in cases:
        result = solve_gauss_seidel(
            case.A, case.b, case.x0, tol, max_iteration, x_exact=case.x_exact,
        )
        result.q = case.q
        result.seed = case.seed
        result.q_actual = diagonal_dominance_index(case.A)
        result.rho = spectral_radius(case.A)
        result.family = case.family
        result.neighbors = case.neighbors
        result.density = case.density
        result.sparsity = case.sparsity
        results.append(result)
        if verbose:
            print(f"{case.family:10s} n={size:4d} seed={case.seed:3d} k={case.neighbors:3d} q={case.q:.3g} rho={result.rho:.6g} steps={result.k:5d} {result.status}", flush=True)
    return results


def result_record(result: Result) -> dict:
    return dict(zip(RESULT_COLUMNS, [
        result.size, result.seed, result.q, result.q_actual, result.rho,
        result.k, result.status, result.success, result.residual,
        result.final_error, result.message,
        result.family, result.neighbors, result.density, result.sparsity,
    ]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=sizes)
    parser.add_argument("--q-values", nargs="+", type=float, default=q_experiments)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--neighbors", nargs="+", type=int, default=DEFAULT_NEIGHBORS,
                        help="Additional row-sparse families (off-diagonal nonzeros per row); dense is always included.")
    parser.add_argument("--dense-only", action="store_true", help="Run only the original dense baseline.")
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if any(k < 1 for k in args.neighbors):
        parser.error("--neighbors must all be >= 1")
    if len(set(args.neighbors)) != len(args.neighbors):
        parser.error("Duplicate --neighbors would repeat experiment cases")
    neighbors = [] if args.dense_only else args.neighbors
    if any(n < 2 for n in args.sizes):
        parser.error("--sizes must all be >= 2")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must all be nonnegative")
    if any(not np.isfinite(q) or q <= 0 for q in args.q_values):
        parser.error("--q-values must all be finite and > 0")
    if not np.isfinite(args.tol) or args.tol <= 0 or args.max_iter < 1:
        parser.error("--tol must be finite and > 0; --max-iter must be >= 1")
    for name, values in [("sizes", args.sizes), ("seeds", args.seeds), ("q-values", args.q_values)]:
        if len(set(values)) != len(values):
            parser.error(f"Duplicate --{name} would repeat experiment cases")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sizes": args.sizes, "q_values": args.q_values, "seeds": args.seeds,
        "tol": args.tol, "max_iter": args.max_iter,
        "neighbors": neighbors,
        "expected_cases": sum(1 + sum(k < n - 1 for k in neighbors) for n in args.sizes) * len(args.seeds) * len(args.q_values),
        "skipped_neighbors_by_size": {str(n): [k for k in neighbors if k >= n - 1] for n in args.sizes},
        "families": ["dense"] + [f"sparse_k{k}" for k in neighbors if any(k < n - 1 for n in args.sizes)],
        "sparsity_definition": "Off-diagonal density = k/(n-1); sparsity = 1-density. Diagonal excluded.",
        "sparse_generation": "Nested row permutations from independent SeedSequence([seed,n,9173]); retained weights match dense. Supports and weights fixed across q. k>=n-1 skipped (dense already included).",
        "norm": "infinity", "stopping_rule": "||b-Ax||_inf / max(1, ||b||_inf) <= tol",
        "x_exact": "ones(n)", "x0": "zeros(n)",
        "generation": "default_rng(seed).uniform(-1, 1, (n,n)); off-diagonal fixed across q; a_ii=sum_j!=i|a_ij|/q",
        "history": "Includes iteration 0; NaN means non-finite arithmetic, not a measured zero.",
        "interpretation": "q>=1 alone does not imply divergence. max_iterations is an iteration-budget outcome. rho near 1 is sensitive to floating-point rounding.",
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "matplotlib": version("matplotlib"),
    }
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_path = args.output_dir / "results.csv"
    history_path = args.output_dir / "convergence_history.csv"
    records = []
    # Write after each seed so completed runs remain available if interrupted.
    with results_path.open("w", newline="", encoding="utf-8") as result_file, history_path.open("w", newline="", encoding="utf-8") as history_file:
        result_writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        history_writer = csv.writer(history_file)
        result_writer.writeheader()
        history_writer.writerow(HISTORY_COLUMNS)
        for n in args.sizes:
            for seed in args.seeds:
                results = run_experiments_on_size(n, q_values=args.q_values, seeds=[seed], neighbors=neighbors, tol=args.tol, max_iteration=args.max_iter, verbose=not args.quiet)
                for result in results:
                    record = result_record(result)
                    records.append(record)
                    result_writer.writerow(record)
                    for k, residual, error in result.history:
                        history_writer.writerow([n, seed, result.q, k, residual, error, result.family, result.neighbors, result.density, result.sparsity])
                result_file.flush()
                history_file.flush()
    frame = pd.DataFrame(records, columns=RESULT_COLUMNS)
    summary = summarize(frame)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    save_plots(summary, history_path, args.output_dir, example_seed=args.seeds[0])
    print(f"Saved {len(frame)} cases to {args.output_dir.resolve()}")
    print(frame["status"].value_counts().to_string())
    return frame


if __name__ == "__main__":
    main()
