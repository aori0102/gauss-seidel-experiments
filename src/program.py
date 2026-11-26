"""Run both experiment sets in order, then build the three report figures."""

import csv
import json
import platform
import sys
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
from report import build_report_bundle


sizes = DEFAULT_SIZES
q_experiments = DEFAULT_Q_VALUES
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "output"
DEFAULT_TOL = 1e-8
DEFAULT_MAX_ITERATION = 10000
DEFAULT_VERBOSE = False
MODES = ("q-variants", "non-diag-dominance")
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


def select_q_values(mode: str, q_values=None) -> list[float]:
    """Select by configured q, without classifying rounded q_actual values."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Choose one of {MODES}.")
    values = list(q_experiments if q_values is None else q_values)
    if not values or any(not np.isscalar(q) or not np.isfinite(q) or q <= 0 for q in values):
        raise ValueError("Configured q values must be nonempty, finite and strictly positive.")
    if len(set(values)) != len(values):
        raise ValueError("Configured q values must not contain duplicates.")
    selected = values if mode == "q-variants" else [q for q in values if q >= 1]
    if not selected:
        raise ValueError("non-diag-dominance needs at least one configured q >= 1.")
    return selected


def run_pipeline(
    mode: str,
    *,
    experiment_sizes=None,
    q_values=None,
    experiment_seeds=None,
    neighbors=None,
    tol: float = DEFAULT_TOL,
    max_iteration: int = DEFAULT_MAX_ITERATION,
    output_dir: Path | None = None,
    verbose: bool = True,
):
    """Execute one mode. Keyword settings support programmatic use and tests.

    Normal experiment settings live in prepare.py and this module's constants.
    main() runs both modes automatically; this helper supports focused tests.
    """
    configured_q = list(q_experiments if q_values is None else q_values)
    selected_q = select_q_values(mode, configured_q)
    experiment_sizes = list(sizes if experiment_sizes is None else experiment_sizes)
    experiment_seeds = list(DEFAULT_SEEDS if experiment_seeds is None else experiment_seeds)
    neighbors = list(DEFAULT_NEIGHBORS if neighbors is None else neighbors)
    if not experiment_sizes or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2 for n in experiment_sizes):
        raise ValueError("Configured sizes must contain integers >= 2.")
    if not experiment_seeds or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0 for seed in experiment_seeds):
        raise ValueError("Configured seeds must contain nonnegative integers.")
    if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1 for k in neighbors):
        raise ValueError("Configured neighbors must contain positive integers.")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0.")
    if isinstance(max_iteration, (bool, np.bool_)) or not isinstance(max_iteration, (int, np.integer)) or max_iteration < 1:
        raise ValueError("max_iteration must be a positive integer.")
    for name, values in [("sizes", experiment_sizes), ("seeds", experiment_seeds), ("neighbors", neighbors)]:
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate configured {name} would repeat experiment cases.")
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": mode,
        "configured_q_values": configured_q,
        "q_selection": "all configured q values" if mode == "q-variants" else "configured q_target >= 1 (including boundary q=1)",
        "sizes": experiment_sizes, "q_values": selected_q, "seeds": experiment_seeds,
        "tol": tol, "max_iter": max_iteration,
        "neighbors": neighbors,
        "expected_cases": sum(1 + sum(k < n - 1 for k in neighbors) for n in experiment_sizes) * len(experiment_seeds) * len(selected_q),
        "skipped_neighbors_by_size": {str(n): [k for k in neighbors if k >= n - 1] for n in experiment_sizes},
        "families": ["dense"] + [f"sparse_k{k}" for k in neighbors if any(k < n - 1 for n in experiment_sizes)],
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
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_path = output_dir / "results.csv"
    history_path = output_dir / "convergence_history.csv"
    records = []
    # Write after each seed so completed runs remain available if interrupted.
    with results_path.open("w", newline="", encoding="utf-8") as result_file, history_path.open("w", newline="", encoding="utf-8") as history_file:
        result_writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS)
        history_writer = csv.writer(history_file)
        result_writer.writeheader()
        history_writer.writerow(HISTORY_COLUMNS)
        for n in experiment_sizes:
            for seed in experiment_seeds:
                results = run_experiments_on_size(n, q_values=selected_q, seeds=[seed], neighbors=neighbors, tol=tol, max_iteration=max_iteration, verbose=verbose)
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
    summary.to_csv(output_dir / "summary.csv", index=False)
    save_plots(summary, history_path, output_dir, example_seed=experiment_seeds[0])
    print(f"Saved {len(frame)} cases to {output_dir.resolve()}")
    print(frame["status"].value_counts().to_string())
    return frame


def main():
    # Validate the second selection before starting an expensive first run.
    select_q_values("non-diag-dominance")
    results = {}
    for step, mode in enumerate(MODES, start=1):
        print(f"[{step}/3] Running {mode}...", flush=True)
        results[mode] = run_pipeline(mode, verbose=DEFAULT_VERBOSE)
    print("[3/3] Building compact report figures and tables...", flush=True)
    results["final"] = build_report_bundle(
        DEFAULT_OUTPUT / "q-variants",
        DEFAULT_OUTPUT / "non-diag-dominance",
        DEFAULT_OUTPUT / "final",
    )
    print(f"Report files saved to {results['final'].resolve()}", flush=True)
    return results



if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Run program.py without arguments. Edit experiment settings in prepare.py or program.py.")
    main()
