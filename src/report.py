"""Build three report figures and compact tables from the two saved runs.

Read-only with respect to experiment data. Overlapping q>=1 cases are checked
for agreement and counted once, never treated as independent repetitions.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from visualization import plt, ScalarFormatter, summarize


CASE_KEY = ["n", "seed", "family", "q_target"]
PREFERRED_FAMILY = "sparse_k2"
ERROR_FLOOR = 1e-12
ERROR_CEILING = 10.0


def load_consistent_runs(q_dir: Path, non_dir: Path):
    """Check that the second run really is a matching q>=1 subset."""
    q_dir, non_dir = Path(q_dir), Path(non_dir)
    all_cases = pd.read_csv(q_dir / "results.csv")
    non_cases = pd.read_csv(non_dir / "results.csv")
    configs = [json.loads((folder / "config.json").read_text(encoding="utf-8")) for folder in [q_dir, non_dir]]
    for frame in [all_cases, non_cases]:
        if frame.empty or frame.duplicated(CASE_KEY).any():
            raise ValueError("Report inputs must contain nonempty, unique experiment cases.")
    if configs[0].get("mode") != "q-variants" or configs[1].get("mode") != "non-diag-dominance":
        raise ValueError("Report inputs have incorrect experiment modes.")
    for setting in ["sizes", "seeds", "neighbors", "tol", "max_iter", "configured_q_values"]:
        if configs[0].get(setting) != configs[1].get(setting):
            raise ValueError(f"Cannot combine runs with different {setting}.")
    if not non_cases["q_target"].ge(1).all():
        raise ValueError("non-diag-dominance contains q < 1.")
    expected = all_cases.loc[all_cases["q_target"] >= 1].sort_values(CASE_KEY).reset_index(drop=True)
    actual = non_cases.sort_values(CASE_KEY).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_exact=False, rtol=1e-12, atol=1e-14)
    except AssertionError as error:
        raise ValueError("The q>=1 cases do not agree between the two runs.") from error
    # All rows of non_cases have just been checked against this unique set.
    return all_cases, non_cases, configs


def family_label(family, group):
    if family == "dense":
        return "Dense"
    return f"Sparse k={int(group['neighbors'].iloc[0])}"


def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_count_axis(ax, summary):
    ax.set_yscale("symlog", linthresh=1)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    values = summary[["iterations_p25", "iterations_median", "iterations_p75"]].to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    ax.set_ylim(bottom=float(values.min()) * 0.8 if values.size else 0)


def plot_q_variants(summary, n, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for family, group in summary.groupby("family", sort=True):
        group = group.sort_values("q_target")
        q = group["q_target"].to_numpy()
        label = family_label(family, group)
        line, = axes[0].plot(q, group["iterations_median"], "o-", markersize=4, label=label)
        axes[0].fill_between(q, group["iterations_p25"], group["iterations_p75"], color=line.get_color(), alpha=.13)
        axes[1].plot(q, group["convergence_rate"], "o-", markersize=4, color=line.get_color(), label=label)
    set_count_axis(axes[0], summary)
    axes[0].set(xlabel="q", ylabel="Completed sweeps (log above 1)", title="(a) Median and IQR of successful runs")
    axes[1].set(xlabel="q", ylabel="Fraction meeting tolerance", ylim=(-.04, 1.06), title="(b) Success fraction across seeds")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Effect of diagonal dominance at n={n}")
    save_figure(fig, output / "01_q_vs_iterations.png")


def plot_error_history(history_path, n, seed, family, q_values, output):
    pieces = []
    for chunk in pd.read_csv(history_path, chunksize=100000):
        pieces.append(chunk.loc[(chunk["n"] == n) & (chunk["seed"] == seed) & (chunk["family"] == family) & chunk["q_target"].isin(q_values)])
    history = pd.concat(pieces, ignore_index=True)
    if history.empty:
        raise ValueError("No history for the report's selected case.")
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    crossings = 0
    for q in q_values:
        curve = history.loc[history["q_target"] == q].sort_values("iteration")
        if curve.empty:
            raise ValueError(f"Missing error history for q={q}.")
        x, y = curve["iteration"].to_numpy(), curve["error"].to_numpy(dtype=float)
        crossed = np.flatnonzero(y > ERROR_CEILING)
        stop = int(crossed[0]) + 1 if crossed.size else len(y)
        x, y = x[:stop], y[:stop]
        valid = (np.isfinite(y) & (y >= 0)) | np.isposinf(y)
        displayed = np.where(valid, np.clip(y, ERROR_FLOOR, ERROR_CEILING), np.nan)
        line, = ax.semilogy(x, displayed, label=f"q={q:g}")
        if crossed.size:
            ax.plot(x[-1], ERROR_CEILING, marker="^", color=line.get_color(), linestyle="none")
            ax.text(.02, .95 - .075 * crossings, f"q={q:g}: error > 10; curve truncated", transform=ax.transAxes,
                    va="top", fontsize=8, color=line.get_color(), bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
            crossings += 1
    ax.set(xlabel="Completed sweeps", ylabel="Infinity-norm solution error", ylim=(ERROR_FLOOR, ERROR_CEILING * 3),
           title=f"Same off-diagonal matrix: n={n}, {family}, seed={seed}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    save_figure(fig, output / "02_error_vs_iterations.png")


def plot_non_diagonal(summary, n, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for family, group in summary.groupby("family", sort=True):
        group = group.sort_values("q_target")
        q = group["q_target"].to_numpy()
        label = family_label(family, group)
        line, = axes[0].plot(q, group["rho_median"], "o-", markersize=4, label=label)
        axes[0].fill_between(q, group["rho_p25"], group["rho_p75"], color=line.get_color(), alpha=.13)
        axes[1].plot(q, group["convergence_rate"], "o-", markersize=4, color=line.get_color(), label=label)
    axes[0].axhline(1, color="black", linestyle="--", label="rho = 1")
    axes[0].set(xlabel="q", ylabel="Spectral radius of B_GS", title="(a) Median and IQR across seeds")
    axes[1].set(xlabel="q", ylabel="Fraction meeting tolerance", ylim=(-.04, 1.06), title="(b) Success fraction across seeds")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Cases with q >= 1 at n={n} (q=1 is the non-strict boundary)")
    save_figure(fig, output / "03_non_diag_dominance.png")


def build_report_bundle(q_dir: Path, non_dir: Path, output_dir: Path) -> Path:
    q_dir, non_dir, output_dir = Path(q_dir), Path(non_dir), Path(output_dir)
    all_cases, non_cases, configs = load_consistent_runs(q_dir, non_dir)
    # Select a common size; prefer a size offering both dense and sparse data.
    common_sizes = sorted(set(all_cases["n"]) & set(non_cases["n"]))
    if not common_sizes:
        raise ValueError("The two runs have no common matrix size.")
    n = int(max(common_sizes))
    at_n = all_cases.loc[all_cases["n"] == n]
    family = PREFERRED_FAMILY if PREFERRED_FAMILY in set(at_n["family"]) else sorted(at_n["family"].unique())[0]
    seeds = sorted(at_n.loc[at_n["family"] == family, "seed"].unique())
    seed = int(configs[0]["seeds"][0]) if configs[0]["seeds"][0] in seeds else int(seeds[0])
    available_q = sorted(at_n["q_target"].unique())
    curve_q = sorted({float(min(available_q, key=lambda q: abs(q-target))) for target in [.1, .9, 1.2, 1.8]})
    all_summary = summarize(all_cases)
    q_summary = all_summary.loc[all_summary["n"] == n].copy()
    non_summary = summarize(non_cases)
    non_summary = non_summary.loc[non_summary["n"] == n]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_q_variants(q_summary, n, output_dir)
    plot_error_history(q_dir / "convergence_history.csv", n, seed, family, curve_q, output_dir)
    plot_non_diagonal(non_summary, n, output_dir)

    # Compact summary: one chosen size, keeping each family and q separate.
    q_summary.to_csv(output_dir / "q_summary.csv", index=False)
    regimes = []
    for family_name, group in all_cases.groupby("family", sort=True):
        for name, mask in [("q<1", group["q_target"] < 1), ("q=1", group["q_target"] == 1), ("q>1", group["q_target"] > 1)]:
            subset = group.loc[mask]
            if subset.empty:
                continue
            regimes.append(dict(family=family_name, regime=name, runs=len(subset), converged=int(subset["success"].sum()),
                                convergence_rate=float(subset["success"].mean()),
                                max_iterations=int(subset["status"].eq("max_iterations").sum()),
                                numerical_error=int(subset["status"].eq("numerical_error").sum())))
    pd.DataFrame(regimes).to_csv(output_dir / "convergence_by_regime.csv", index=False)

    # At most six measured examples: one success and one failure per family,
    # from q>1 first; at most three families (preferred sparse, dense, other).
    family_order = list(dict.fromkeys([family, "dense"] + sorted(non_cases["family"].unique())))[:3]
    examples = []
    for family_name in family_order:
        pool = non_cases.loc[(non_cases["n"] == n) & (non_cases["family"] == family_name)]
        for success in [True, False]:
            candidates = pool.loc[pool["success"] == success]
            if candidates.empty:
                continue
            candidates = candidates.assign(boundary=candidates["q_target"].eq(1), other_seed=candidates["seed"].ne(seed))
            chosen = candidates.sort_values(["boundary", "other_seed", "q_target", "seed"]).iloc[0]
            examples.append(chosen[non_cases.columns].to_dict())
    pd.DataFrame(examples, columns=non_cases.columns).to_csv(output_dir / "non_diagonal_examples.csv", index=False)
    provenance = {
        # Source paths are relative to the directory containing provenance.json.
        "sources": {"q-variants": Path(os.path.relpath(q_dir.resolve(), output_dir.resolve())).as_posix(),
                    "non-diag-dominance": Path(os.path.relpath(non_dir.resolve(), output_dir.resolve())).as_posix()},
        "source_cases": {"q-variants": len(all_cases), "non-diag-dominance": len(non_cases)},
        "total_executions": len(all_cases) + len(non_cases), "unique_cases": len(all_cases),
        "overlapping_cases_verified_and_counted_once": len(non_cases),
        "report_size": n, "error_history_family": family, "error_history_seed": seed,
        "error_history_q_values": curve_q, "error_display_limits": [ERROR_FLOOR, ERROR_CEILING],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    notes = f'''BỘ HÌNH VÀ SỐ LIỆU GỌN CHO BÁO CÁO

01_q_vs_iterations.png: câu hỏi 1. So sánh q và số vòng lặp tại n={n}; dải màu là khoảng tứ phân vị. Số vòng chỉ tổng hợp các lần đạt ngưỡng; đọc cùng ô tỷ lệ đạt ngưỡng.
02_error_vs_iterations.png: quá trình giảm/tăng sai số tại n={n}, bộ {family}, seed={seed}. Các đường dùng cùng phần ngoài đường chéo. Sai số hiển thị trong [1e-12,10]; tam giác đánh dấu vượt ngưỡng trên, không sửa dữ liệu thật. Ngưỡng dừng dựa vào phần dư, không phải sai số nghiệm.
03_non_diag_dominance.png: câu hỏi 2. Bán kính phổ và tỷ lệ đạt ngưỡng cho q>=1 tại n={n}; q=1 chỉ là biên không chéo trội nghiêm ngặt. Trung vị rho dưới 1 không bảo đảm mọi seed có rho dưới 1.

q_summary.csv: bảng thống kê ở kích thước được chọn, không gộp các bộ thưa với bộ đặc.
convergence_by_regime.csv: số lần đạt ngưỡng theo q<1, q=1, q>1, dùng toàn bộ kích thước.
non_diagonal_examples.csv: tối đa sáu lần chạy thực tế minh họa đạt/không đạt ngưỡng khi q>=1, ưu tiên q>1. numerical_error và max_iterations là trạng thái tính toán; max_iterations không tự chứng minh phân kỳ.

Hai bộ có {len(non_cases)} trường hợp q>=1 trùng nhau. Đã đối chiếu toàn bộ kết quả trùng, chỉ đếm một lần trong bảng tổng hợp: {len(all_cases)} trường hợp duy nhất, không phải {len(all_cases)+len(non_cases)} mẫu độc lập.
provenance.json ghi nguồn và quy tắc chọn hình. Hình dùng kích thước lớn nhất có ở cả hai bộ; ưu tiên bộ sparse_k2 và seed đầu tiên trong cấu hình, không chọn lại seed theo kết quả đẹp.
'''
    (output_dir / "HUONG-DAN.txt").write_text(notes, encoding="utf-8")
    return output_dir
