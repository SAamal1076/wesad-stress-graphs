"""Leave-one-subject-out (LOSO) evaluation.

Leakage rules enforced here:
* the model is a fresh clone per fold and is fitted on training-subject rows only;
* any scaling / selection / tuning must live INSIDE the estimator (sklearn Pipeline,
  or GridSearchCV with an inner LeaveOneGroupOut on the training subjects);
* metrics are computed per test subject, then summarised over subjects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def loso_splits(groups):
    """Yield (test_subject, train_idx, test_idx) for every subject, with a leakage check."""
    groups = np.asarray(groups)
    for g in pd.unique(groups):
        test_idx = np.flatnonzero(groups == g)
        train_idx = np.flatnonzero(groups != g)
        assert_no_subject_leak(groups, train_idx, test_idx)
        yield g, train_idx, test_idx


def assert_no_subject_leak(groups, train_idx, test_idx):
    groups = np.asarray(groups)
    overlap = set(groups[train_idx]) & set(groups[test_idx])
    if overlap:
        raise AssertionError(f"subject(s) in both train and test: {sorted(overlap)}")
    if len(np.intersect1d(train_idx, test_idx)):
        raise AssertionError("a row is in both train and test")


def run_loso(estimator, X, y, groups, fit_params_fn=None, proba=False):
    """Return out-of-fold predictions (and probabilities) for every row."""
    X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
    y, groups = np.asarray(y), np.asarray(groups)
    pred = np.empty(len(y), dtype=y.dtype)
    prob = np.full((len(y), len(np.unique(y))), np.nan) if proba else None
    for g, tr, te in loso_splits(groups):
        m = clone(estimator)
        Xtr = X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr]
        Xte = X.iloc[te] if isinstance(X, pd.DataFrame) else X[te]
        kw = fit_params_fn(tr) if fit_params_fn else {}
        m.fit(Xtr, y[tr], **kw)
        pred[te] = m.predict(Xte)
        if proba:
            prob[te] = m.predict_proba(Xte)
    return (pred, prob) if proba else pred


def per_subject_metrics(y, pred, groups) -> pd.DataFrame:
    df = pd.DataFrame(dict(y=y, pred=pred, subject=groups))
    rows = []
    for s, d in df.groupby("subject", sort=False):
        rows.append(dict(subject=s, n=len(d),
                         macro_f1=f1_score(d.y, d.pred, average="macro", zero_division=0),
                         balanced_acc=balanced_accuracy_score(d.y, d.pred),
                         acc=accuracy_score(d.y, d.pred)))
    return pd.DataFrame(rows)


def summarise(y, pred, groups) -> dict:
    ps = per_subject_metrics(y, pred, groups)
    return {
        "macro_f1_mean": ps.macro_f1.mean(), "macro_f1_sd": ps.macro_f1.std(ddof=1),
        "bal_acc_mean": ps.balanced_acc.mean(), "bal_acc_sd": ps.balanced_acc.std(ddof=1),
        "worst_subject_f1": ps.macro_f1.min(),
        "pooled_acc": accuracy_score(y, pred),                 # comparable with Schmidt et al.
        "pooled_macro_f1": f1_score(y, pred, average="macro"),
    }
