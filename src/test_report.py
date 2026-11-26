import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import program as gs
from report import load_consistent_runs, build_report_bundle


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.q_dir, self.non_dir = self.root/"q-variants", self.root/"non-diag-dominance"
        cases = gs.run_experiments_on_size(3, q_values=[.5, 1., 1.2], seeds=[42], neighbors=[], max_iteration=30, verbose=False)
        all_cases = pd.DataFrame([gs.result_record(result) for result in cases])
        for folder, mode in [(self.q_dir, "q-variants"), (self.non_dir, "non-diag-dominance")]:
            folder.mkdir()
            selected = cases if mode == "q-variants" else [case for case in cases if case.q >= 1]
            pd.DataFrame([gs.result_record(case) for case in selected]).to_csv(folder/"results.csv", index=False)
            config = dict(mode=mode, sizes=[3], seeds=[42], neighbors=[], tol=1e-8, max_iter=30, configured_q_values=[.5, 1., 1.2])
            (folder/"config.json").write_text(json.dumps(config), encoding="utf-8")
            rows = [[3,42,case.q,k,residual,error,case.family,case.neighbors,case.density,case.sparsity] for case in selected for k,residual,error in case.history]
            pd.DataFrame(rows,columns=gs.HISTORY_COLUMNS).to_csv(folder/"convergence_history.csv",index=False)
        self.original = all_cases

    def test_overlap_counted_once_and_sources_unchanged(self):
        before = {p:p.read_bytes() for folder in [self.q_dir, self.non_dir] for p in folder.iterdir()}
        final = build_report_bundle(self.q_dir, self.non_dir, self.root/"final")
        self.assertEqual(len(list(final.glob("*.png"))), 3)
        self.assertEqual(pd.read_csv(final/"convergence_by_regime.csv").runs.sum(), 3)
        self.assertLessEqual(len(pd.read_csv(final/"non_diagonal_examples.csv")), 6)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_conflicting_overlap_is_rejected(self):
        path = self.non_dir/"results.csv"
        frame = pd.read_csv(path)
        frame.loc[0,"rho"] += .1
        frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"do not agree"):
            load_consistent_runs(self.q_dir,self.non_dir)

    def test_missing_case_is_rejected(self):
        path = self.non_dir/"results.csv"
        pd.read_csv(path).iloc[:1].to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"do not agree"):
            load_consistent_runs(self.q_dir,self.non_dir)

    def test_mismatched_settings_are_rejected(self):
        path = self.non_dir/"config.json"
        config = json.loads(path.read_text())
        config["tol"] = 1e-4
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,"different tol"):
            load_consistent_runs(self.q_dir,self.non_dir)

    def test_duplicate_cases_are_rejected(self):
        path = self.non_dir/"results.csv"
        frame = pd.read_csv(path)
        pd.concat([frame,frame.iloc[:1]],ignore_index=True).to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,"unique"):
            load_consistent_runs(self.q_dir,self.non_dir)


if __name__ == "__main__":
    unittest.main()
