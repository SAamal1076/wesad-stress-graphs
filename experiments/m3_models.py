"""M3 - classifiers under LOSO and cross-protocol-order evaluation.

Primary: logistic regression with C (and the random-walk lambda) tuned by inner 5-fold
cross-validation grouped by subject on the training subjects only.
Robustness: random forest (5 seeds) for 'levels' and 'levels + rw'.
Out-of-fold probabilities are saved for M4 (test-time graph smoothing).
"""
import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wesad.windows import task_view  # noqa: E402
from wesad.physio import group_columns  # noqa: E402
from wesad.evaluation import assert_no_subject_leak  # noqa: E402

OUT = ROOT / "results" / "m3"
(OUT / "oof").mkdir(parents=True, exist_ok=True)
C_GRID = (0.01, 0.1, 1.0)
LAMBDAS = ("0.005", "0.02", "0.05")
ORDER_A = ["S4", "S5", "S7", "S8", "S10", "S13", "S15", "S17"]
ORDER_B = ["S2", "S3", "S6", "S9", "S11", "S14", "S16"]
SENSORS = {"chest": ["chest"], "wrist": ["wrist"], "both": ["chest", "wrist"]}


def feature_sets(cols, sensor):
    """name -> list of candidate column lists (several candidates = lambda to tune)."""
    cfgs = SENSORS[sensor]
    lev = sum((group_columns(cols, f"{c}_levels") for c in cfgs), [])
    gcols = lambda pref: sum(([c for c in cols if c.startswith(f"g_{pref}_{cfg}_")] for cfg in cfgs), [])  # noqa: E731
    return {
        "levels": [lev],
        "levels+corr": [lev + gcols("corr")],
        "levels+pcorr": [lev + gcols("pcorr")],
        "levels+rw": [lev + gcols(f"rw{lam}") for lam in LAMBDAS],
        "rw_only": [gcols(f"rw{lam}") for lam in LAMBDAS],
    }


def make_lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def make_rf(seed):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(100, min_samples_leaf=5, class_weight="balanced",
                                                n_jobs=1, random_state=seed))


def inner_select(df_tr, y_tr, g_tr, candidates):
    """Pick (candidate index, C) by inner GroupKFold(5) macro-F1 on training subjects."""
    if len(candidates) == 1 and len(C_GRID) == 1:
        return 0, C_GRID[0]
    gkf = GroupKFold(n_splits=min(5, len(np.unique(g_tr))))
    best, best_score = (0, C_GRID[0]), -np.inf
    for k, C in itertools.product(range(len(candidates)), C_GRID):
        X = df_tr[candidates[k]].values
        scores = []
        for itr, iva in gkf.split(X, y_tr, g_tr):
            m = make_lr(C).fit(X[itr], y_tr[itr])
            scores.append(f1_score(y_tr[iva], m.predict(X[iva]), average="macro"))
        if np.mean(scores) > best_score:
            best, best_score = (k, C), np.mean(scores)
    return best


def outer_fold(df, y, g, tr, te, candidates, with_rf):
    assert_no_subject_leak(g, tr, te)
    k, C = inner_select(df.iloc[tr], y[tr], g[tr], candidates)
    cols = candidates[k]
    lr = make_lr(C).fit(df.iloc[tr][cols].values, y[tr])
    res = {"te": te, "lr_prob": lr.predict_proba(df.iloc[te][cols].values), "k": k, "C": C}
    if with_rf:
        probs = [make_rf(s).fit(df.iloc[tr][cols].values, y[tr]).predict_proba(df.iloc[te][cols].values)
                 for s in range(5)]
        res["rf_probs"] = np.stack(probs)
    return res


def outer_splits(g, scheme):
    if scheme == "loso":
        for s in pd.unique(g):
            yield np.flatnonzero(g != s), np.flatnonzero(g == s)
    else:  # cross-order: A -> B and B -> A
        a, b = np.isin(g, ORDER_A), np.isin(g, ORDER_B)
        yield np.flatnonzero(a), np.flatnonzero(b)
        yield np.flatnonzero(b), np.flatnonzero(a)


def run(task, scheme, n_jobs):
    feats = pd.read_csv(ROOT / "results" / "m2" / "features_60s.csv")
    mot = pd.read_csv(ROOT / "results" / "m1" / "motion_features_60s.csv", index_col=0)
    feats = pd.concat([feats, mot.add_prefix("mot_").reset_index(drop=True)], axis=1)
    t = task_view(feats, task)
    y, g = t.y.values, t.subject.values
    n_cls = len(np.unique(y))
    configs = []
    for sensor in ("chest", "wrist", "both"):
        for name, cands in feature_sets(t.columns, sensor).items():
            configs.append((sensor, name, cands, name in ("levels", "levels+rw")))
    configs.append(("motion", "motion", [[c for c in t.columns if c.startswith("mot_")]], True))
    configs.append(("time", "time", [["t_mid_min"]], True))

    meta = []
    for sensor, name, cands, with_rf in configs:
        t0 = time.time()
        folds = Parallel(n_jobs=n_jobs)(delayed(outer_fold)(t, y, g, tr, te, cands, with_rf)
                                        for tr, te in outer_splits(g, scheme))
        lr_prob = np.zeros((len(y), n_cls))
        rf_prob = np.zeros((5, len(y), n_cls)) if with_rf else None
        chosen = []
        for f in folds:
            lr_prob[f["te"]] = f["lr_prob"]
            if with_rf:
                rf_prob[:, f["te"]] = f["rf_probs"]
            chosen.append({"k": int(f["k"]), "C": f["C"]})
        tag = f"{task}__{scheme}__{sensor}__{name}"
        np.savez_compressed(OUT / "oof" / f"{tag}.npz", y=y, subject=g, start_s=t.start_s.values,
                            lr_prob=lr_prob, **({"rf_prob": rf_prob} if with_rf else {}))
        meta.append({"task": task, "scheme": scheme, "sensor": sensor, "set": name,
                     "n_features": len(cands[0]), "chosen": chosen})
        f1 = f1_score(y, lr_prob.argmax(1), average="macro")
        print(f"{tag:55s} pooled LR macro-F1 {f1:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    with open(OUT / f"meta_{task}_{scheme}.json", "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scheme", required=True, choices=["loso", "cross_order"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    run(a.task, a.scheme, a.jobs)
