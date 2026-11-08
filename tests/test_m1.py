"""Leakage and windowing tests (run: python -m pytest -q tests)."""
import sys
import pathlib

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from wesad.windows import (SUBJECTS, LABEL_KEY, build_window_index, condition_runs,  # noqa: E402
                           load_subject, task_view, windows_from_runs)
from wesad.evaluation import assert_no_subject_leak, loso_splits, run_loso  # noqa: E402
from wesad.windows import DATA_DIR  # noqa: E402

needs_data = pytest.mark.skipif(
    not all((DATA_DIR / f"{s}.npz").exists() for s in SUBJECTS),
    reason="needs the prepared WESAD files in data/wesad_small (run scripts/prepare_wesad.py)")


@pytest.fixture(scope="module")
def win60():
    return build_window_index(60.0, 5.0)


@needs_data
def test_fifteen_subjects_five_runs_each(win60):
    assert sorted(win60.subject.unique(), key=lambda s: int(s[1:])) == SUBJECTS
    runs = win60.groupby("subject").run_id.nunique()
    assert (runs == 5).all()  # baseline, stress, amusement, 2 x meditation


@needs_data
@pytest.mark.parametrize("win_s", [10.0, 60.0])
def test_every_window_has_one_constant_label(win_s):
    """Re-read the 350 Hz label stream and check every sample inside every window."""
    for s in SUBJECTS:
        d = load_subject(s)
        lab, fs = d[LABEL_KEY], float(d["chest_ECG_fs"])
        w = windows_from_runs(condition_runs(lab, fs, s), win_s, 5.0)
        for r in w.itertuples():
            seg = lab[int(round(r.start_s * fs)):int(round(r.end_s * fs))]
            assert len(seg) == int(round(win_s * fs))
            assert (seg == r.label).all(), f"{s} window at {r.start_s}s crosses a label change"


@needs_data
def test_windows_stay_inside_their_run(win60):
    for s in SUBJECTS:
        d = load_subject(s)
        runs = condition_runs(d[LABEL_KEY], float(d["chest_ECG_fs"]), s).set_index("run_id")
        w = win60[win60.subject == s]
        r = runs.loc[w.run_id]
        assert (w.start_s.values >= r.start_s.values - 1e-9).all()
        assert (w.end_s.values <= r.end_s.values + 1e-9).all()


@needs_data
def test_task_views_exclude_meditation_and_transients(win60):
    for task in ("binary", "three_class"):
        t = task_view(win60, task)
        assert not t.label.isin([0, 4, 5, 6, 7]).any()
        assert set(t.y.unique()) == ({0, 1} if task == "binary" else {0, 1, 2})
        # every subject has every class (needed for per-subject macro-F1)
        assert (t.groupby("subject").y.nunique() == t.y.nunique()).all()


@needs_data
def test_loso_folds_are_disjoint_and_complete(win60):
    g = win60.subject.values
    tested = []
    for subj, tr, te in loso_splits(g):
        assert set(g[te]) == {subj}
        assert subj not in set(g[tr])
        assert len(tr) + len(te) == len(g)
        tested.append(subj)
    assert sorted(tested) == sorted(SUBJECTS)


def test_leak_detector_fires():
    g = np.array(["S2", "S2", "S3", "S3"])
    with pytest.raises(AssertionError):
        assert_no_subject_leak(g, np.array([0, 2]), np.array([1, 3]))


class SpyClassifier(BaseEstimator, ClassifierMixin):
    """Remembers the row ids (column 0) it was trained on."""
    seen = []

    def fit(self, X, y):
        SpyClassifier.seen.append(set(np.asarray(X)[:, 0].astype(int)))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


@needs_data
def test_run_loso_never_fits_on_test_rows(win60):
    t = task_view(win60, "binary")
    X = np.c_[np.arange(len(t)), np.zeros(len(t))]
    SpyClassifier.seen = []
    run_loso(SpyClassifier(), X, t.y.values, t.subject.values)
    assert len(SpyClassifier.seen) == 15
    for (subj, tr, te), seen in zip(loso_splits(t.subject.values), SpyClassifier.seen):
        assert seen.isdisjoint(set(te)), f"fold {subj}: model saw test rows"
        assert seen == set(tr)
