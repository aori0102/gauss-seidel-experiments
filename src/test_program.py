from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import program as gs
import prepare
import solve
import visualization


class GaussSeidelTests(unittest.TestCase):
    def test_mode_selection_preserves_grid_and_includes_boundary(self):
        configured = [.5, 1., 1.05, 1.2, 1.8]
        self.assertEqual(gs.select_q_values("q-variants", configured), configured)
        self.assertEqual(gs.select_q_values("non-diag-dominance", configured), [1., 1.05, 1.2, 1.8])
        self.assertEqual(configured, [.5, 1., 1.05, 1.2, 1.8])
        self.assertEqual(len(gs.select_q_values("q-variants")), 11)
        self.assertEqual(len(gs.select_q_values("non-diag-dominance")), 4)

    def test_invalid_mode_or_empty_selection_fails_before_output(self):
        for mode, grid in [("invalid", [.5, 1.]), ("non-diag-dominance", [.1, .9]), ("q-variants", []), ("q-variants", [1., 1.]), ("q-variants", [np.nan])]:
            with self.subTest(mode=mode, grid=grid), self.assertRaises(ValueError):
                gs.select_q_values(mode, grid)

    def test_main_runs_both_modes_then_report_without_arguments(self):
        events = []
        def run(mode, **kwargs):
            events.append(mode)
            return mode
        def report(q_dir, non_dir, final_dir):
            events.append("final")
            self.assertEqual(q_dir.name, "q-variants")
            self.assertEqual(non_dir.name, "non-diag-dominance")
            self.assertEqual(final_dir.name, "final")
            return final_dir
        with patch.object(gs, "run_pipeline", side_effect=run), patch.object(gs, "build_report_bundle", side_effect=report):
            result = gs.main()
        self.assertEqual(events, ["q-variants", "non-diag-dominance", "final"])
        self.assertEqual(result["final"], gs.DEFAULT_OUTPUT/"final")
        with patch.object(gs, "run_pipeline", side_effect=RuntimeError("failed")), patch.object(gs, "build_report_bundle") as report:
            with self.assertRaises(RuntimeError):
                gs.main()
            report.assert_not_called()

    def test_no_argument_pipeline_exports_two_runs_and_three_report_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(gs, "DEFAULT_OUTPUT", root), patch.object(gs, "sizes", [3]), patch.object(gs, "q_experiments", [.5, 1., 1.2]), patch.object(gs, "DEFAULT_SEEDS", [42]), patch.object(gs, "DEFAULT_NEIGHBORS", []):
                runs = gs.main()
            all_cases = runs["q-variants"]
            subset = runs["non-diag-dominance"]
            self.assertEqual(len(all_cases), 3)
            self.assertEqual(len(subset), 2)
            pd.testing.assert_frame_equal(all_cases[all_cases.q_target >= 1].reset_index(drop=True), subset)
            for mode, expected in [("q-variants", [.5, 1., 1.2]), ("non-diag-dominance", [1., 1.2])]:
                folder = root/mode
                config = json.loads((folder/"config.json").read_text())
                self.assertEqual(config["mode"], mode)
                self.assertEqual(config["q_values"], expected)
                history = pd.read_csv(folder/"convergence_history.csv")
                self.assertEqual(sorted(history.q_target.unique()), expected)
            figures = sorted(path.name for path in (root/"final").glob("*.png"))
            self.assertEqual(figures, ["01_q_vs_iterations.png", "02_error_vs_iterations.png", "03_non_diag_dominance.png"])
            provenance = json.loads((root/"final/provenance.json").read_text())
            self.assertEqual(provenance["total_executions"], 5)
            self.assertEqual(provenance["unique_cases"], 3)
            self.assertEqual(provenance["overlapping_cases_verified_and_counted_once"], 2)
            self.assertEqual(pd.read_csv(root/"final/convergence_by_regime.csv").runs.sum(), 3)

    def test_sparse_degrees_nested_supports_and_q_control(self):
        cases = list(prepare.prepare_experiments(10, q_values=[.5, 1.2], seeds=[42], neighbors=[2, 4, 8]))
        self.assertEqual(len(cases), 8)
        groups = {family: [c for c in cases if c.family == family] for family in ["dense", "sparse_k2", "sparse_k4", "sparse_k8"]}
        dense = groups["dense"][0].A
        masks = []
        for family, pair in groups.items():
            first, second = pair
            off = first.A.copy()
            np.fill_diagonal(off, 0)
            mask = off != 0
            self.assertTrue(np.all(np.count_nonzero(off, axis=1) == first.neighbors))
            self.assertAlmostEqual(first.density, first.neighbors/9)
            self.assertAlmostEqual(first.sparsity, 1-first.density)
            np.testing.assert_array_equal(first.A[mask], dense[mask])
            np.testing.assert_array_equal(first.A[mask], second.A[mask])
            for case in pair:
                self.assertAlmostEqual(prepare.diagonal_dominance_index(case.A), case.q)
                np.testing.assert_allclose(case.b, case.A @ case.x_exact)
            if family != "dense":
                masks.append(mask)
        self.assertTrue(np.all(masks[0] <= masks[1]))
        self.assertTrue(np.all(masks[1] <= masks[2]))

    def test_sparse_sampling_does_not_change_dense_baseline(self):
        old = list(prepare.prepare_experiments(10, q_values=[.7], seeds=[42, 43]))
        extended = list(prepare.prepare_experiments(10, q_values=[.7], seeds=[42, 43], neighbors=[8, 2, 4]))
        dense = [case for case in extended if case.family == "dense"]
        for before, after in zip(old, dense):
            np.testing.assert_array_equal(before.A, after.A)
            np.testing.assert_array_equal(before.b, after.b)
            rng = np.random.default_rng(before.seed)
            original = prepare.apply_diagonal(rng.uniform(-1, 1, (10, 10)), .7)
            np.testing.assert_array_equal(original, after.A)
        reordered = list(prepare.prepare_experiments(10, q_values=[.7], seeds=[42, 43], neighbors=[2, 4, 8]))
        lookup = {(c.seed, c.family): c.A for c in reordered}
        for case in extended:
            np.testing.assert_array_equal(case.A, lookup[case.seed, case.family])

    def test_skip_full_or_impossible_degrees_and_validate_k(self):
        cases = list(prepare.prepare_experiments(3, q_values=[.5], seeds=[42], neighbors=[1, 2, 4, 8]))
        self.assertEqual([c.family for c in cases], ["dense", "sparse_k1"])
        for invalid in [[0], [-1], [2.5], [True], [2, 2]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                list(prepare.prepare_experiments(10, neighbors=invalid))

    def test_summary_keeps_dense_and_sparse_populations_separate(self):
        frame = pd.DataFrame([
            dict(n=10, q_target=.9, family="dense", neighbors=9, density=1., sparsity=0., status="converged", iterations=5, rho=.1, final_error=1e-9),
            dict(n=10, q_target=.9, family="sparse_k2", neighbors=2, density=2/9, sparsity=7/9, status="max_iterations", iterations=20, rho=1.1, final_error=4.),
        ])
        summary = visualization.summarize(frame).set_index("family")
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary.loc["dense", "runs"], 1)
        self.assertEqual(summary.loc["dense", "iterations_median"], 5)
        self.assertEqual(summary.loc["sparse_k2", "convergence_rate"], 0)
        self.assertTrue(np.isnan(summary.loc["sparse_k2", "iterations_median"]))

    def test_sparse_pipeline_exports_family_plots_and_unambiguous_history(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = gs.run_pipeline("q-variants", experiment_sizes=[6], q_values=[.5, 1.2], experiment_seeds=[42], neighbors=[2, 4], max_iteration=30, output_dir=Path(directory), verbose=False)
            self.assertEqual(len(frame), 6)
            self.assertFalse(frame.duplicated(["n", "seed", "family", "q_target"]).any())
            for folder in ["", "sparse_k2", "sparse_k4"]:
                for name in ["q_vs_iterations.png", "q_vs_spectral_radius.png", "error_vs_iterations.png"]:
                    self.assertGreater((Path(directory)/folder/name).stat().st_size, 0)
            self.assertTrue((Path(directory)/"sparsity_comparison_n6.png").exists())
            history = pd.read_csv(Path(directory)/"convergence_history.csv")
            counts = history.groupby(["n", "seed", "family", "q_target"]).size()
            expected = frame.set_index(["n", "seed", "family", "q_target"])["iterations"]+1
            pd.testing.assert_series_equal(counts.sort_index(), expected.sort_index(), check_names=False)

    def test_prepared_cases_share_off_diagonal_values_not_mutable_arrays(self):
        cases = list(prepare.prepare_experiments(4, q_values=[.5, 1.2], seeds=[42]))
        self.assertEqual(len(cases), 2)
        first, second = cases
        mask = ~np.eye(4, dtype=bool)
        np.testing.assert_array_equal(first.A[mask], second.A[mask])
        for case in cases:
            np.testing.assert_allclose(case.b, case.A @ case.x_exact)
            self.assertAlmostEqual(prepare.diagonal_dominance_index(case.A), case.q)
        self.assertFalse(np.shares_memory(first.A, second.A))
        self.assertFalse(np.shares_memory(first.x_exact, second.x_exact))
        self.assertFalse(np.shares_memory(first.x0, second.x0))
        first.x0[:] = 9
        np.testing.assert_array_equal(second.x0, np.zeros(4))
        

    def test_one_sweep_uses_new_lower_components_and_float_copy(self):
        A = np.array([[4, 1], [2, 3]])
        b = np.array([1, 2])
        x0 = np.zeros(2, dtype=int)
        result = solve.solve_gauss_seidel(A, b, x0, tol=1e-30, max_iteration=1)
        np.testing.assert_allclose(result.x, [0.25, 0.5])
        np.testing.assert_array_equal(x0, [0, 0])
        self.assertEqual(result.status, "max_iterations")
        self.assertEqual(result.k, 1)
        self.assertEqual([h[0] for h in result.history], [0, 1])
        

    def test_diagonal_system_and_zero_initial_solution(self):
        result = solve.solve_gauss_seidel(np.diag([2., 4.]), [4, 12], [0, 0], x_exact=[2, 3])
        self.assertTrue(result.success)
        self.assertEqual(result.k, 1)
        self.assertEqual(result.final_error, 0)
        np.testing.assert_allclose(result.x, [2, 3])
        result = solve.solve_gauss_seidel(np.eye(2), [0, 0], [0, 0])
        self.assertEqual(result.k, 0)
        self.assertTrue(result.success)
        

    def test_infinity_norm_and_inclusive_tolerance(self):
        result = solve.solve_gauss_seidel(np.eye(2), [1, 0], [0.5, 0.5], tol=0.5)
        self.assertTrue(result.success)
        self.assertEqual(result.k, 0)
        self.assertEqual(result.residual, 0.5)
        self.assertEqual(solve.error_norm(np.zeros(2), [3, 4]), 4)
        

    def test_known_two_by_two_spectral_radius(self):
        for q in [0.5, 1., 1.2]:
            A = np.array([[1., q], [q, 1.]])
            self.assertAlmostEqual(solve.spectral_radius(A), q*q)
            self.assertAlmostEqual(prepare.diagonal_dominance_index(A), q)
        result = solve.solve_gauss_seidel([[1., .5], [.5, 1.]], [1.5, 1.5], [0., 0.], x_exact=[1, 1])
        self.assertTrue(result.success)
        self.assertLess(result.final_error, 1e-7)
        

    def test_q_above_one_can_converge(self):
        # Lower-triangular nonsingular A gives B_GS = 0 even with q > 1.
        A = np.array([[1., 0.], [2., 1.]])
        self.assertGreater(prepare.diagonal_dominance_index(A), 1)
        self.assertEqual(solve.spectral_radius(A), 0)
        self.assertTrue(solve.solve_gauss_seidel(A, A @ np.ones(2), [0, 0]).success)
        

    def test_finite_divergence_is_not_claimed_on_timeout(self):
        result = solve.solve_gauss_seidel([[1, 1.2], [1.2, 1]], [2.2, 2.2], [0, 0], max_iteration=10)
        self.assertEqual(result.status, "max_iterations")
        self.assertFalse(result.success)
        self.assertGreater(result.history[-1][1], result.history[0][1])
        

    def test_arithmetic_overflow_has_own_status(self):
        result = solve.solve_gauss_seidel([[1e-300]], [1e300], [0])
        self.assertEqual(result.status, "numerical_error")
        self.assertFalse(result.success)
        

    def test_reject_invalid_inputs(self):
        for A, b, x in [([1, 2], [1, 2], [0, 0]), ([[0, 1], [1, 1]], [1, 1], [0, 0]), (np.eye(2), [[1], [1]], [0, 0]), (np.eye(2), [1, np.nan], [0, 0]), (np.eye(2), [1, 1], [0]), ([], [], [])]:
            with self.subTest(A=A), self.assertRaises(ValueError):
                solve.solve_gauss_seidel(A, b, x)
        for kwargs in [dict(tol=0), dict(tol=np.nan), dict(max_iteration=0), dict(max_iteration=1.5), dict(max_iteration=True)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                solve.solve_gauss_seidel(np.eye(2), [1, 1], [0, 0], **kwargs)
                

    def test_diagonal_generation_preserves_off_diagonal_and_input(self):
        base = np.random.default_rng(42).uniform(-1, 1, size=(5, 5))
        original = base.copy()
        mask = ~np.eye(5, dtype=bool)
        for q in [.1, .99, 1.2]:
            A = prepare.apply_diagonal(base, q)
            np.testing.assert_array_equal(A[mask], original[mask])
            np.testing.assert_allclose(prepare.diagonal_dominance_index(A), q)
        np.testing.assert_array_equal(base, original)
        for q in [0, -1, np.inf]:
            with self.assertRaises(ValueError):
                prepare.apply_diagonal(base, q)
        with self.assertRaises(ValueError):
            prepare.apply_diagonal(np.eye(2), .5)
            

    def test_experiments_reproducible_and_records_complete(self):
        kwargs = dict(q_values=[.5, 1.2], seeds=[42, 43], verbose=False)
        first = gs.run_experiments_on_size(3, **kwargs)
        second = gs.run_experiments_on_size(3, **kwargs)
        self.assertEqual([gs.result_record(r) for r in first], [gs.result_record(r) for r in second])
        self.assertEqual(len(first), 4)
        for r in first:
            self.assertAlmostEqual(r.q_actual, r.q)
            self.assertEqual(len(r.history), r.k + 1)
            self.assertIsNotNone(r.final_error)
            

    def test_summary_does_not_count_failures_as_successful_iterations(self):
        frame = pd.DataFrame([
            dict(n=2, q_target=1.2, status="converged", iterations=5, rho=.8, final_error=1e-9),
            dict(n=2, q_target=1.2, status="max_iterations", iterations=100, rho=1.2, final_error=9),
            dict(n=2, q_target=2., status="numerical_error", iterations=20, rho=4., final_error=np.nan),
        ])
        summary = visualization.summarize(frame)
        self.assertEqual(summary.iloc[0].iterations_median, 5)
        self.assertEqual(summary.iloc[0].convergence_rate, .5)
        self.assertTrue(np.isnan(summary.iloc[1].iterations_median))
        self.assertEqual(summary.iloc[1].numerical_error_runs, 1)
        

    def test_pipeline_exports_data_history_config_and_three_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = gs.run_pipeline("q-variants", experiment_sizes=[2, 3], q_values=[.5, 1.2], experiment_seeds=[42, 43], max_iteration=20, output_dir=Path(directory), verbose=False)
            self.assertEqual(len(frame), 8)
            for name in ["results.csv", "summary.csv", "convergence_history.csv", "config.json", "q_vs_iterations.png", "q_vs_spectral_radius.png", "error_vs_iterations.png"]:
                self.assertGreater((Path(directory) / name).stat().st_size, 0)
            history = pd.read_csv(Path(directory) / "convergence_history.csv")
            self.assertEqual(len(history), int((frame.iterations + 1).sum()))
            self.assertEqual(list(pd.read_csv(Path(directory) / "results.csv").columns), gs.RESULT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
