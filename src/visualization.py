from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Save plots without opening a GUI window.
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


def with_family(data: pd.DataFrame) -> pd.DataFrame:
    """Old dense-only CSVs remain readable; never modify the input frame."""
    data = data.copy()
    if "family" not in data:
        data["family"] = "dense"
    if "neighbors" not in data:
        data["neighbors"] = data["n"] - 1
    if "density" not in data:
        data["density"] = data["neighbors"] / (data["n"] - 1)
    if "sparsity" not in data:
        data["sparsity"] = 1.0 - data["density"]
    return data


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    results = with_family(results)
    rows = []
    for (n, q, family), group in results.groupby(["n", "q_target", "family"], sort=True):
        successful = group.loc[group["status"] == "converged", "iterations"]
        rows.append({
            "n": n, "q_target": q, "runs": len(group),
            "family": family, "neighbors": int(group["neighbors"].iloc[0]),
            "density": float(group["density"].iloc[0]),
            "sparsity": float(group["sparsity"].iloc[0]),
            "converged_runs": len(successful),
            "max_iterations_runs": int((group["status"] == "max_iterations").sum()),
            "numerical_error_runs": int((group["status"] == "numerical_error").sum()),
            "convergence_rate": len(successful) / len(group),
            "iterations_median": successful.median(),
            "iterations_p25": successful.quantile(0.25),
            "iterations_p75": successful.quantile(0.75),
            "rho_median": group["rho"].median(),
            "rho_p25": group["rho"].quantile(0.25),
            "rho_p75": group["rho"].quantile(0.75),
            "final_error_median_converged": group.loc[group["status"] == "converged", "final_error"].median(),
        })
    return pd.DataFrame(rows)


def save_plots(summary: pd.DataFrame, history_path: Path, output_dir: Path, example_seed: int):
    summary = with_family(summary)
    output_dir = Path(output_dir)
    for family, family_summary in summary.groupby("family", sort=True):
        folder = output_dir if family == "dense" else output_dir / family
        folder.mkdir(parents=True, exist_ok=True)
        save_iteration_plot(family_summary, folder, family=family)
        save_spectral_plot(family_summary, folder, family=family)
        save_error_plot(history_path, folder, example_seed, family=family)
    save_sparsity_comparisons(summary, output_dir)


def save_spectral_plot(summary: pd.DataFrame, output_dir: Path, *, family: str = "dense"):
    summary = with_family(summary)
    summary = summary.loc[summary["family"] == family]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for n, group in summary.groupby("n", sort=True):
        q = group["q_target"].to_numpy()
        line, = ax.plot(q, group["rho_median"], "o-", label=f"n={n}")
        ax.fill_between(q, group["rho_p25"], group["rho_p75"], color=line.get_color(), alpha=0.15)
    ax.axhline(1.0, color="black", linestyle="--", label="rho = 1")
    ax.set(xlabel="q target", ylabel="Spectral radius of B_GS", title=f"{family}: spectral radius, median and IQR across seeds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(Path(output_dir) / "q_vs_spectral_radius.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_iteration_plot(summary: pd.DataFrame, output_dir: Path, *, family: str = "dense"):
    """Spread the upper plot on a log scale; keep the lower plot linear.

    Symlog is linear between -1 and 1 so an initial-guess success at zero
    completed sweeps remains representable rather than being hidden.
    """
    summary = with_family(summary)
    summary = summary.loc[summary["family"] == family]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")
    for n, group in summary.groupby("n", sort=True):
        q = group["q_target"].to_numpy()
        line, = axes[0].plot(q, group["iterations_median"], "o-", label=f"n={n}")
        axes[0].fill_between(q, group["iterations_p25"], group["iterations_p75"], color=line.get_color(), alpha=0.15)
        axes[1].plot(q, group["convergence_rate"], "o-", label=f"n={n}")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set(ylabel="Completed sweeps (log scale above 1)", title=f"{family}: median and IQR among runs meeting tolerance")
    axes[1].set(xlabel="q target", ylabel="Fraction meeting tolerance", ylim=(-0.05, 1.05))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(Path(output_dir) / "q_vs_iterations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_error_plot(
    history_path: Path,
    output_dir: Path,
    example_seed: int,
    *,
    error_ceiling: float = 10.0,
    error_floor: float = 1e-12,
    family: str = "dense",
):
    """Plot existing history within a readable display range; never edit CSV.

    Stop each displayed curve at its first upper-limit crossing and mark it
    with a triangle, rather than drawing a misleading flat convergence tail.
    Values below error_floor are clipped for the logarithmic display only.
    """
    if not (np.isfinite(error_floor) and np.isfinite(error_ceiling)
            and 0 < error_floor < error_ceiling):
        raise ValueError("Display limits must be finite and 0 < floor < ceiling.")
    selected = []
    for chunk in pd.read_csv(history_path, chunksize=100000):
        if "family" not in chunk:
            chunk["family"] = "dense"
        selected.append(chunk.loc[(chunk["seed"] == example_seed) & (chunk["family"] == family)])
    history = pd.concat(selected, ignore_index=True)
    if history.empty:
        raise ValueError(f"No history for family={family}, seed={example_seed}.")
    ns = sorted(history["n"].unique())
    fig, axes = plt.subplots(len(ns), 1, figsize=(10, 3.8 * len(ns)), squeeze=False, layout="constrained")
    for ax, n in zip(axes[:, 0], ns):
        group = history.loc[history["n"] == n]
        crossing_labels = 0
        q_values = sorted(group["q_target"].unique())
        indices = np.unique(np.linspace(0, len(q_values) - 1, min(5, len(q_values)), dtype=int))
        for idx in indices:
            q = q_values[idx]
            curve = group.loc[group["q_target"] == q].sort_values("iteration")
            error = curve["error"].to_numpy(dtype=float)
            iterations = curve["iteration"].to_numpy()
            crossings = np.flatnonzero(error > error_ceiling)
            # Positive infinity is also a crossing; NaN remains a data gap.
            stop = int(crossings[0]) + 1 if crossings.size else len(error)
            error, iterations = error[:stop], iterations[:stop]
            valid = (np.isfinite(error) & (error >= 0)) | np.isposinf(error)
            displayed = np.where(valid, np.clip(error, error_floor, error_ceiling), np.nan)
            line, = ax.semilogy(iterations, displayed, label=f"q={q:g}")
            if crossings.size:
                ax.plot(iterations[-1], error_ceiling, marker="^",
                        color=line.get_color(), markersize=7, linestyle="none")
                ax.annotate(f"q={q:g}: error > {error_ceiling:g}; curve truncated",
                            xy=(iterations[-1], error_ceiling),
                            xytext=(0.02, 0.96 - 0.07 * crossing_labels),
                            textcoords="axes fraction", va="top",
                            color=line.get_color(), fontsize=9)
                crossing_labels += 1
        ax.set(title=f"{family}: same off-diagonal matrix, n={n}, seed={example_seed}", xlabel="Completed sweeps", ylabel="Infinity-norm solution error")
        ax.set_ylim(error_floor, error_ceiling * 3)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        f"Display clipped to [{error_floor:g}, {error_ceiling:g}]; "
        "triangles mark upper-limit crossings. CSV values unchanged.",
        fontsize=10,
    )
    fig.savefig(Path(output_dir) / "error_vs_iterations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sparsity_comparisons(summary: pd.DataFrame, output_dir: Path):
    """At fixed n compare q curves across row degrees, without mixing seeds."""
    summary = with_family(summary)
    for n, size_summary in summary.groupby("n", sort=True):
        if size_summary["family"].nunique() < 2:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
        for family, group in size_summary.groupby("family", sort=True):
            group = group.sort_values("q_target")
            q = group["q_target"].to_numpy()
            k = int(group["neighbors"].iloc[0])
            density = float(group["density"].iloc[0])
            label = f"{family}, k={k}, density={density:.1%}"
            line, = axes[0].plot(q, group["iterations_median"], "o-", label=label)
            color = line.get_color()
            axes[0].fill_between(q, group["iterations_p25"], group["iterations_p75"], color=color, alpha=0.12)
            axes[1].plot(q, group["rho_median"], "o-", color=color, label=label)
            axes[1].fill_between(q, group["rho_p25"], group["rho_p75"], color=color, alpha=0.12)
            axes[2].plot(q, group["convergence_rate"], "o-", color=color, label=label)
        axes[0].set_yscale("symlog", linthresh=1)
        axes[0].yaxis.set_major_formatter(ScalarFormatter())
        bounds = size_summary[["iterations_p25", "iterations_median", "iterations_p75"]].to_numpy(dtype=float)
        finite_bounds = bounds[np.isfinite(bounds) & (bounds >= 0)]
        axes[0].set_ylim(bottom=float(finite_bounds.min()) * 0.8 if finite_bounds.size else 0)
        axes[0].set(ylabel="Completed sweeps (log above 1)", title=f"n={n}: effect of off-diagonal sparsity")
        axes[1].axhline(1, color="black", linestyle="--", label="rho = 1")
        axes[1].set(ylabel="Spectral radius of B_GS")
        axes[2].set(xlabel="q target", ylabel="Fraction meeting tolerance", ylim=(-0.05, 1.05))
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.savefig(Path(output_dir) / f"sparsity_comparison_n{n}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
